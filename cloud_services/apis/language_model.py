"""Use the same s3e-backed Gemini API client as the FUNCTO pipeline."""

# BEGIN one VLM implementation shared by the main and scene-graph entry points.
from local_models.gemini import Gemini

__all__ = ["Gemini"]
# END shared Gemini public interface.
