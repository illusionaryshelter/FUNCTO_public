"""Pre-allocation limits for independently owned host and device resources."""

# BEGIN explicit request limits never depend on an environment override.
DEFAULT_MAX_HOST_RSS_BYTES = 8 * 1024**3
MAX_HOST_RSS_BYTES = 10 * 1024**3
DEFAULT_MAX_DEVICE_USED_BYTES = 6 * 1024**3
MAX_DEVICE_USED_BYTES = 8 * 1024**3
MIN_HOST_AVAILABLE_BYTES = 2 * 1024**3
MIN_DEVICE_FREE_BYTES = 2 * 1024**3


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
        raise ValueError("Host RSS admission requires an explicit limit no larger than ten GiB")
    process = psutil.Process()
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
