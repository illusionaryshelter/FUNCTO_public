"""FUNCTO's semantic requests through the existing s3e Gemini gateway backend.

Perception is local; annotated images are sent to the user-authorized Gemini
gateway for semantic point/pose selection. Credentials are read only on use.
"""

import json
import os
from pathlib import Path
import re
import sys

from PIL import Image


class Gemini:
    """A small application adapter; s3e owns the HTTP protocol and validation."""

    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.7-flash-tiered")
        self.base_url = base_url or os.environ.get(
            "SUB2API_BASE_URL", "http://127.0.0.1:18080/antigravity")

    def chat(self, prompt: str, images: list[Image.Image] | None = None,
             *, system_prompt: str = "", json_output: bool = False) -> str:
        """Generate once; transport failures and malformed output are not hidden."""
        key = os.environ.get("SUB2API_API_KEY")
        if not self.base_url or not key:
            raise RuntimeError("Set SUB2API_BASE_URL and SUB2API_API_KEY for the Gemini gateway")
        # Both projects live in the shared recovery workspace. Read the existing
        # backend directly without installing packages or copying its protocol.
        source = Path(__file__).resolve().parents[2] / "s3e"
        if not (source / "s3e" / "backends" / "gemini.py").is_file():
            raise FileNotFoundError(f"Missing local s3e Gemini backend: {source}")
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        from s3e.backends.gemini import GeminiVLM

        backend = GeminiVLM(self.model, base_url=self.base_url, api_key=key, timeout=120)
        try:
            generation = {"responseMimeType": "application/json"} if json_output else {}
            result = backend.query(images or [], prompt, system_prompt=system_prompt or None,
                                   generate=True, **generation)
            return result.text
        finally:
            backend.close()

    def chat_json(self, prompt: str, images: list[Image.Image] | None = None,
                  *, system_prompt: str = "") -> dict:
        """Require a JSON object for FUNCTO's semantic selections."""
        # The gateway does not consistently enforce responseMimeType. State the
        # serialization contract explicitly without changing the task's visual
        # instructions or fields. Reasoning requested by a task stays internal.
        output_rule = ("Return exactly one JSON object. Do not include Markdown fences, "
                       "explanations or text outside that object. Perform any requested "
                       "step-by-step analysis internally and output only the final JSON.")
        system_prompt = f"{system_prompt}\n\n{output_rule}".strip()
        text = self.chat(prompt, images, system_prompt=system_prompt, json_output=True)
        # The configured gateway can wrap an otherwise valid JSON response in
        # a Markdown fence even with responseMimeType set. Accept that complete
        # wrapper only; do not guess a JSON fragment from arbitrary explanation.
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text.strip(),
                              flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            text = fenced.group(1)
        try:
            value = json.loads(text)
        except json.JSONDecodeError as error:
            raise json.JSONDecodeError(
                f"Gemini did not return a JSON object; response starts with {text[:300]!r}",
                error.doc, error.pos) from error
        if not isinstance(value, dict):
            raise ValueError("Gemini must return a JSON object")
        return value
