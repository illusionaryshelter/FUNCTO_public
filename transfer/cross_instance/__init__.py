"""Identity-preserving region transfer between observed object instances.

The public pipeline consumes an already loaded source skill and calibrated,
named RGB-D views.  The array entry replays the identical numerical chain on
prepared evidence; both retain source rows, target elements and optimizer
records.  Perception model execution and tool-reference binding belong to the
callers that provide their respective observations and motion relations.
"""

from .pipeline import run_arrays, run_region_transfer

__all__ = ["run_arrays", "run_region_transfer"]
