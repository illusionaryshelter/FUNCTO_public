"""Probe one explicitly chosen model through FUNCTO's existing Gemini client.

One invocation makes one request. Model names are never substituted and HTTP
errors retain their response body. Optional images exercise the same visual
request path used by semantic selection, without running SAM or any GPU model.
Credentials come only from SUB2API_API_KEY and are not written to the report.
"""

import argparse
import json
import os
from pathlib import Path
import time

from PIL import Image

from local_models.gemini import Gemini


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Exact gateway model name")
    parser.add_argument("--base-url", help="Gateway root; defaults to SUB2API_BASE_URL or the existing local gateway")
    parser.add_argument("--image", type=Path, action="append", default=[], help="Optional image; may be repeated")
    parser.add_argument("--prompt", default="Return the word ready.")
    parser.add_argument("--json-output", action="store_true", help="Require a JSON object using the production client")
    parser.add_argument("--output", type=Path, help="Optional JSON report file")
    args = parser.parse_args()
    client = Gemini(model=args.model, base_url=args.base_url)
    report = dict(model=client.model, base_url=client.base_url,
                  image_paths=[str(path.resolve()) for path in args.image],
                  json_output=args.json_output, prompt=args.prompt)
    start = time.monotonic()
    try:
        images = []
        for path in args.image:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        query = client.chat_json if args.json_output else client.chat
        report.update(status="completed", response=query(args.prompt, images))
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        # requests.Response is false for HTTP errors; test identity explicitly
        # so that 404/500 response bodies are not accidentally discarded.
        response = getattr(error, "response", None)
        if response is not None:
            report.update(http_status=response.status_code, response_body=response.text)
    report["elapsed_seconds"] = time.monotonic() - start
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    # A gateway might echo request contents in its error response. Redact the
    # configured credential even though the client uses an Authorization header.
    credential = os.environ.get("SUB2API_API_KEY")
    if credential:
        serialized = serialized.replace(credential, "[REDACTED]")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    raise SystemExit(0 if report["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
