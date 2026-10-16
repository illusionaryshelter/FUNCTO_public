"""Independent two-target-view transfer; old and joint single-view paths stay intact.

This package owns the continuous-world-query, raw-score fusion experiment.
It reuses read-only geometry and native registration primitives, but never
dispatches to an old/new region solver as an acceptance fallback.
"""
