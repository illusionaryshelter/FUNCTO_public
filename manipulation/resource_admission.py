"""Pre-allocation limits for independently owned host and device resources."""

# BEGIN explicit request limits never depend on an environment override.
DEFAULT_MAX_HOST_RSS_BYTES = 8 * 1024**3
MAX_HOST_RSS_BYTES = 12 * 1024**3
DEFAULT_MAX_DEVICE_USED_BYTES = 6 * 1024**3
MAX_DEVICE_USED_BYTES = 8 * 1024**3
MIN_HOST_AVAILABLE_BYTES = 2 * 1024**3
MIN_DEVICE_FREE_BYTES = 2 * 1024**3
# BEGIN an explicit request may include the retained simulator's process tree.
_HOST_PROCESS_ROOT_IDENTITY = None


def configure_host_process_root(root_pid):
    """Bind this worker's host budget to itself or an explicit live ancestor.

    The standalone default remains this process plus its children. A worker
    launched by a retained simulation instead supplies that simulation's PID,
    so later geometry checks include its parent and sibling workers too. The
    root is never guessed from a shell/session tree. Once explicitly bound,
    this process cannot narrow or replace that scope for a subsequent check.
    """
    import psutil

    global _HOST_PROCESS_ROOT_IDENTITY
    if type(root_pid) is not int or root_pid <= 0:
        raise ValueError("Host process root must be an explicit positive PID")
    current = psutil.Process()
    ancestors = [current, *current.parents()]
    root = next((item for item in ancestors if item.pid == root_pid), None)
    if root is None:
        raise ValueError("Host process root must be this process or one of its live ancestors")
    identity = (root.pid, root.create_time())
    if _HOST_PROCESS_ROOT_IDENTITY is not None and _HOST_PROCESS_ROOT_IDENTITY != identity:
        raise ValueError("The explicitly bound host process root cannot change within a worker")
    _HOST_PROCESS_ROOT_IDENTITY = identity
    return {"pid": identity[0], "create_time": identity[1]}
# END no environment variable can enlarge limits or silently select an ancestor.


class ResourceLimitError(ValueError):
    """A measured capacity refusal, distinct from a model or runtime error."""

    def __init__(self, resource, required_bytes, limit_bytes, *, description="Resource capacity exceeded"):
        self.resource = str(resource)
        self.required_bytes = int(required_bytes)
        self.limit_bytes = int(limit_bytes)
        super().__init__(f"{description} ({self.resource}): required {self.required_bytes} bytes, "
                         f"allowed {self.limit_bytes} bytes")


def require_host_memory(new_bytes, *, pending_existing_bytes, max_rss_bytes):
    """Admit new host storage while honoring every live lazy reservation.

    RSS already contains allocated arrays, native shapes and tree nodes. Add
    only the proposed allocation and still-unmaterialized storage promised to
    existing geometry. In particular, do not subtract a capacity estimate from
    RSS or charge every resident shape twice. The caller counts shared geometry
    once by identity before supplying ``pending_existing_bytes``.
    """
    import psutil

    for name, value in (("new_bytes", new_bytes), ("pending_existing_bytes", pending_existing_bytes)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative byte count")
    if type(max_rss_bytes) is not int or not 0 < max_rss_bytes <= MAX_HOST_RSS_BYTES:
        raise ValueError("Host RSS admission requires an explicit limit no larger than twelve GiB")
    # BEGIN all nested geometry admissions share the explicit request scope.
    if _HOST_PROCESS_ROOT_IDENTITY is None:
        process = psutil.Process()
    else:
        root_pid, created = _HOST_PROCESS_ROOT_IDENTITY
        process = psutil.Process(root_pid)
        if process.create_time() != created:
            raise RuntimeError("The bound host process root exited and its PID was reused")
        # A dead ancestor can remain as a zombie while this worker is reparented.
        # Its unchanged PID/time alone must not admit the now-empty old tree.
        current = psutil.Process()
        if current.pid != root_pid and not any(
                parent.pid == root_pid and parent.create_time() == created for parent in current.parents()):
            raise RuntimeError("The bound host process root no longer contains this worker")
    # END current RSS already includes every previously materialized live map.
    rss = process.memory_info().rss
    # A completed child may disappear between enumeration and reading RSS.
    # Other process/permission failures remain visible capacity-query errors.
    for child in process.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    available = psutil.virtual_memory().available
    additional = new_bytes + pending_existing_bytes
    required = int(rss) + additional
    if required > max_rss_bytes:
        raise ResourceLimitError("host_process_rss", required, max_rss_bytes,
            description="Live host storage and remaining geometry reservations exceed the request RSS limit")
    required = additional + MIN_HOST_AVAILABLE_BYTES
    if required > available:
        raise ResourceLimitError("host_available", required, int(available),
            description="Host memory must retain two GiB after the admitted allocations")
# END no allocation, cleanup, process termination or CUDA initialization occurs here.


# BEGIN native maps and Torch planners share the same whole-device admission.
def require_device_memory(torch, device, *, native_bytes, torch_bytes=0, max_device_used_bytes):
    """Admit new buffers against current whole-device usage and free capacity.

    ``native_bytes`` belongs to Mapper or other native allocators, which cannot
    reuse pages held by Torch's caching allocator. Only ``torch_bytes`` may use
    this process's reserved but currently unallocated Torch pages. All existing
    fields, CUDA contexts, cached pages and other applications already appear
    in driver-reported device usage; none is added again as a new allocation.

    This is a conservative capacity estimate at the allocation boundary, not
    an allocator lock: concurrent allocations and fragmentation can still make
    a later native allocation fail. No cache is cleared or limit increased.
    """
    for name, value in (("native_bytes", native_bytes), ("torch_bytes", torch_bytes)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative byte count")
    if type(max_device_used_bytes) is not int or not 0 < max_device_used_bytes <= MAX_DEVICE_USED_BYTES:
        raise ValueError("Device admission requires an explicit limit no larger than eight GiB")
    free_bytes, total_bytes = map(int, torch.cuda.mem_get_info(device))
    reusable_torch_bytes = (max(0, int(torch.cuda.memory_reserved(device))
                                 -int(torch.cuda.memory_allocated(device))) if torch_bytes else 0)
    if native_bytes > free_bytes:
        raise ResourceLimitError("gpu_native_reserve", native_bytes, free_bytes,
            description="Native allocations cannot borrow pages held by the Torch allocator")
    required_bytes = native_bytes+torch_bytes
    if required_bytes > free_bytes+reusable_torch_bytes:
        raise ResourceLimitError("gpu_allocation_capacity", required_bytes, free_bytes+reusable_torch_bytes,
            description="Current GPU capacity cannot cover the proposed native and Torch allocations")
    new_bytes = native_bytes+max(0, torch_bytes-reusable_torch_bytes)
    projected_used = total_bytes-free_bytes+new_bytes
    if projected_used > max_device_used_bytes:
        raise ResourceLimitError("gpu_device_used", projected_used, max_device_used_bytes,
            description="New allocations exceed the request whole-device usage limit")
    if new_bytes+MIN_DEVICE_FREE_BYTES > free_bytes:
        raise ResourceLimitError("gpu_free_reserve", new_bytes+MIN_DEVICE_FREE_BYTES, free_bytes,
            description="Device memory must retain two GiB after the admitted allocations")
    return new_bytes
# END allocator ownership never changes the explicitly requested physical caps.
