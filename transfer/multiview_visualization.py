"""Render frozen multiview inputs and observed output modes as one PNG per case.

Run after evaluation, using the same --manifest and --output experiment paths.
The output directory must already contain summary.json and every frozen
prediction. Images are saved under output/visualizations. No model, matching,
registration, reprojection, or ground-truth geometry is invoked by this module.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from .multiview_experiment import (ALGORITHMS, _path, load_fusion, load_inputs,
                                    read_manifest, restore_prediction)


# BEGIN read-only multiview result rendering on original observed image pixels.
_BACKGROUND = (235, 239, 245)
_INK = (26, 38, 57)
_MUTED = (78, 91, 111)
_RED = (214, 45, 57)
_GREEN = (35, 185, 105)
_SAM = (0, 176, 211)
_SECONDARY = ((51, 119, 230), (160, 83, 218))
_PANEL_WIDTH, _IMAGE_HEIGHT = 640, 480
_PANEL_HEADER, _PANEL_FOOTER = 58, 82
_GAP, _MARGIN, _TITLE_HEIGHT, _LEGEND_HEIGHT = 18, 24, 90, 145


def _font(size):
    """Use the already available system font; no downloads or font fallback."""
    return ImageFont.truetype("DejaVuSans.ttf", size)


def _read_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def _mask(value, shape, name):
    raw = np.asarray(value)
    if raw.dtype != np.bool_ or raw.shape != shape:
        raise ValueError(f"{name} must be a Boolean mask of shape {shape}")
    return raw


def _overlay(image, mask, color, alpha):
    """Tint only supplied observed pixels and outline their actual pixel mask."""
    mask = _mask(mask, (image.height, image.width), "Image overlay")
    binary = Image.fromarray(mask.astype(np.uint8) * 255)
    layer = Image.new("RGBA", image.size, (*color, 0))
    layer.putalpha(binary.point(lambda value: alpha if value else 0))
    output = Image.alpha_composite(image.convert("RGBA"), layer)
    eroded = np.asarray(binary.filter(ImageFilter.MinFilter(3))) > 0
    boundary = Image.fromarray((mask & ~eroded).astype(np.uint8) * 255)
    layer.putalpha(boundary)
    return Image.alpha_composite(output, layer).convert("RGB")


def _text_lines(draw, text, width, font):
    """Wrap raw English identifiers without silently removing rejection reasons."""
    words = str(text).split()
    lines, current = [], ""
    for word in words:
        proposed = word if not current else current + " " + word
        if draw.textlength(proposed, font=font) <= width:
            current = proposed
            continue
        if current:
            lines.append(current)
            current = ""
        # Raw reason identifiers can be longer than a panel. Split them only
        # for layout, retaining every character and its original spelling.
        for char in word:
            if current and draw.textlength(current + char, font=font) > width:
                lines.append(current)
                current = char
            else:
                current += char
    if current:
        lines.append(current)
    return lines


def _panel(image, title, detail, *, badge=None, badge_color=_MUTED, notice=None):
    """Keep the full original image visible; headers are separate from pixels."""
    height = _PANEL_HEADER + _IMAGE_HEIGHT + _PANEL_FOOTER
    panel = Image.new("RGB", (_PANEL_WIDTH, height), "white")
    draw = ImageDraw.Draw(panel)
    heading, small = _font(22), _font(16)
    draw.text((14, 15), title, font=heading, fill=_INK)
    if badge:
        badge_font = _font(16)
        badge_width = int(draw.textlength(badge, font=badge_font)) + 20
        left = _PANEL_WIDTH - badge_width - 12
        draw.rounded_rectangle((left, 12, _PANEL_WIDTH - 12, 40), radius=5, fill=badge_color)
        draw.text((left + 10, 16), badge, font=badge_font, fill="white")
    contained = ImageOps.contain(image, (_PANEL_WIDTH, _IMAGE_HEIGHT), Image.Resampling.LANCZOS)
    image_left = (_PANEL_WIDTH - contained.width) // 2
    image_top = _PANEL_HEADER + (_IMAGE_HEIGHT - contained.height) // 2
    panel.paste(contained, (image_left, image_top))
    if notice:
        notice_font = _font(18)
        lines = _text_lines(draw, notice, _PANEL_WIDTH - 40, notice_font)
        box_height = 12 + 24 * len(lines)
        top = _PANEL_HEADER + _IMAGE_HEIGHT - box_height - 10
        draw.rounded_rectangle((10, top, _PANEL_WIDTH - 10, top + box_height),
                               radius=5, fill=(248, 250, 253), outline=(207, 215, 226))
        for number, line in enumerate(lines):
            draw.text((20, top + 6 + 24 * number), line, font=notice_font, fill=badge_color)
    lines = _text_lines(draw, detail, _PANEL_WIDTH - 28, small)
    if len(lines) > 3:
        # The panel has a bounded footer, while the complete raw reasons remain
        # in the page-level rejection notes. Do not crop image pixels to fit text.
        lines = lines[:2] + ["Full reasons in notes below."]
    for number, line in enumerate(lines):
        draw.text((14, _PANEL_HEADER + _IMAGE_HEIGHT + 10 + 21 * number),
                  line, font=small, fill=_MUTED)
    draw.rectangle((0, 0, _PANEL_WIDTH - 1, height - 1), outline=(209, 217, 228), width=1)
    return panel


def _observation_domains(bank, targets):
    """Bind frozen global observation indices to exact per-camera point arrays."""
    ids = [target["id"] for target in targets]
    if list(bank["view_ids"]) != ids or len(ids) != 2:
        raise ValueError("Frozen bank must match the two canonical manifest view IDs")
    offsets = np.asarray(bank["view_offsets"])
    observed = np.asarray(bank["observation_xyz"])
    if (offsets.shape != (3,) or offsets.dtype.kind not in "iu"
            or offsets[0] != 0 or offsets[-1] != len(observed)
            or np.any(np.diff(offsets.astype(np.int64)) < 0)):
        raise ValueError("Frozen view_offsets do not delimit the observation bank")
    domains = {}
    for view_index, target in enumerate(targets):
        begin, end = map(int, offsets[view_index:view_index + 2])
        observation = target["observation"]
        if (not np.array_equal(observed[begin:end], observation["xyz"])
                or not np.all(bank["observation_view_index"][begin:end] == view_index)
                or not np.array_equal(bank["observation_local_index"][begin:end], np.arange(end - begin))):
            raise ValueError("Frozen observation provenance disagrees with original view points")
        domains[target["id"]] = {"index": view_index, "begin": begin, "end": end,
                                 "observation": observation}
    return domains


def _mode_pixels(mode, domains, observation_count):
    """Recover only a mode's own camera pixels; cross-view reprojection is forbidden.

    Global masks and sample indices address bank.observation_xyz, not the exact
    XYZ candidate bank. Their complete support must stay within the declared
    view's offset interval. Saved local indices and pixels are cross-checked
    against that interval before anything is drawn.
    """
    view_id = mode["view_id"]
    if view_id not in domains:
        raise ValueError("A frozen mode declares an unknown observing camera")
    domain = domains[view_id]
    begin, end, observation = domain["begin"], domain["end"], domain["observation"]
    if mode["view_index"] != domain["index"]:
        raise ValueError("Frozen mode view_id and view_index disagree")
    mask = _mask(mode["mask"], (observation_count,), "Frozen mode")
    members = np.flatnonzero(mask)
    samples = np.asarray(mode["sample_indices"])
    seed = mode["seed_index"]
    if (samples.ndim != 1 or samples.dtype.kind not in "iu"
            or isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))):
        raise ValueError("Mode samples and seed must be original integer observation indices")
    if (np.any(members < begin) or np.any(members >= end)
            or np.any(samples < begin) or np.any(samples >= end)
            or not begin <= seed < end):
        raise ValueError("A mode crosses camera observation domains; it cannot be plotted in either view")
    if not mask[seed] or not np.all(mask[samples]):
        raise ValueError("Frozen seed/samples must belong to their mode's actual observed mask")
    local_samples = samples.astype(np.int64) - begin
    pixels = observation["pixels"]
    if (mode["local_seed_index"] != seed - begin
            or not np.array_equal(mode["sample_local_indices"], local_samples)
            or not np.array_equal(mode["seed_pixel"], pixels[seed - begin])
            or not np.array_equal(mode["sample_pixels"], pixels[local_samples])):
        raise ValueError("Saved mode pixels do not agree with frozen global-to-local observation identity")
    image_mask = np.zeros(observation["object_mask"].shape, dtype=bool)
    local_members = pixels[members - begin]
    image_mask[local_members[:, 1], local_members[:, 0]] = True
    return image_mask, pixels[local_samples]


def _draw_modes(image, entries):
    output = image.copy()
    for _, _, image_mask, _, color in entries:
        output = _overlay(output, image_mask, color, 95)
    draw = ImageDraw.Draw(output)
    for _, _, _, samples, color in entries:
        for x, y in samples:
            # Marker centers are exact original observed pixels. Their visible
            # circle size is a drawing glyph, not an enlarged inferred region.
            draw.ellipse((int(x) - 3, int(y) - 3, int(x) + 3, int(y) + 3),
                         fill=color, outline=(20, 30, 45), width=1)
            draw.ellipse((int(x) - 1, int(y) - 1, int(x) + 1, int(y) + 1), fill="white")
    return output


def _summary_rows(summary, cases):
    """Require the completed evaluated matrix before rendering any case."""
    if summary.get("protocol", {}).get("truth_loaded_after_all_predictions") is not True:
        raise ValueError("Visualization requires the completed evaluated summary.json")
    expected = {(case["id"], algorithm) for case in cases for algorithm in ALGORITHMS}
    rows = {}
    for row in summary["rows"]:
        key = (row["case"], row["algorithm"])
        if key in rows or "quality" not in row:
            raise ValueError("Evaluation summary contains duplicate or unevaluated matrix cells")
        rows[key] = row
    if set(rows) != expected:
        raise ValueError("Evaluation summary must cover every manifest case and all three algorithms")
    return rows


def _render_case(manifest_path, case, output, rows):
    source, _, _, targets = load_inputs(manifest_path, case)
    bank = load_fusion(output / "fusion" / case["id"])["bank"]
    domains = _observation_domains(bank, targets)
    source_rgb = _read_rgb(_path(manifest_path, case["source_observation_dir"]) / "rgb.png")
    source_evidence = np.load(_path(manifest_path, case["source_input_dir"]) /
                              "source_evidence_mask.npy", allow_pickle=False)
    source_evidence = _mask(source_evidence, (source_rgb.height, source_rgb.width), "Source evidence")
    source_pixels = source["pixels"]
    if not np.array_equal(source_evidence[source_pixels[:, 1], source_pixels[:, 0]], source["region_mask"]):
        raise ValueError("Frozen source evidence mask does not match its observed region pixels")
    records = {record["id"]: record for record in case["targets"]}
    rgb = {target["id"]: _read_rgb(_path(manifest_path,
           records[target["id"]]["observation_dir"]) / "rgb.png") for target in targets}
    panels = [_panel(_overlay(source_rgb, source_evidence, _RED, 130), "Source | evidence",
        f"Input only. Red evidence: {int(source_evidence.sum())} pixels; "
        f"valid-depth evidence: {int(source['region_mask'].sum())} points.", badge="INPUT")]
    for target in targets:
        view_id = target["id"]
        image = rgb[view_id]
        selected = _mask(target["selected_mask"], (image.height, image.width), "Selected SAM")
        missing = target["missing_target"]
        reason = "; ".join(target["missing_target_reasons"])
        detail = (f"Input only. MISSING: {reason}." if missing else
                  f"Input only. Cyan SAM: {int(selected.sum())} pixels; "
                  f"valid-depth observation: {len(target['observation']['xyz'])} points.")
        panels.append(_panel(_overlay(image, selected, _SAM, 70), f"Target {view_id} | SAM",
            detail, badge="MISSING" if missing else "INPUT",
            badge_color=_RED if missing else _MUTED, notice=reason if missing else None))

    predictions, per_view, rejection_notes = {}, {}, []
    for algorithm in ALGORITHMS:
        prediction = restore_prediction(output / case["id"] / algorithm / "prediction")
        row = rows[(case["id"], algorithm)]
        if (prediction["accepted"] != row["accepted"] or prediction["reasons"] != row["reasons"]):
            raise ValueError("Prediction acceptance/reasons differ from the evaluated summary")
        predictions[algorithm] = prediction
        per_view[algorithm] = {target["id"]: [] for target in targets}
        for number, mode in enumerate(prediction["modes"]):
            mask, samples = _mode_pixels(mode, domains, len(bank["observation_xyz"]))
            primary = mode["type"] == "primary"
            color = _GREEN if primary else _SECONDARY[(number - 1) % len(_SECONDARY)]
            per_view[algorithm][mode["view_id"]].append((number, mode, mask, samples, color))
        if not prediction["accepted"]:
            rejection_notes.append(f"{algorithm}: " + "; ".join(prediction["reasons"]))

    for target in targets:
        view_id = target["id"]
        for algorithm in ALGORITHMS:
            prediction = predictions[algorithm]
            entries = per_view[algorithm][view_id]
            observed = _draw_modes(rgb[view_id], entries)
            detail = "; ".join(f"m{number} {'primary' if mode['type'] == 'primary' else 'secondary'}: "
                f"{int(mask.sum())} pixels, {len(samples)} samples" for number, mode, mask, samples, _ in entries)
            if not prediction["accepted"]:
                detail = "Diagnostic output. " + (detail or "; ".join(prediction["reasons"]))
            elif not detail:
                detail = "Accepted run; this camera owns no returned mode."
            notice = None
            if target["missing_target"]:
                notice = "MISSING: " + "; ".join(target["missing_target_reasons"])
            elif not entries:
                notice = "NO MODES RETURNED" if not prediction["modes"] else "NO MODES IN THIS VIEW"
            panels.append(_panel(observed, f"{view_id} | {algorithm}", detail,
                badge=("MISSING" if target["missing_target"] else
                       "RUN ACCEPTED" if prediction["accepted"] else "REJECTED"),
                badge_color=_GREEN if prediction["accepted"] and not target["missing_target"] else _RED,
                notice=notice))

    width = 2 * _MARGIN + 3 * _PANEL_WIDTH + 2 * _GAP
    panel_height = _PANEL_HEADER + _IMAGE_HEIGHT + _PANEL_FOOTER
    basic_height = _TITLE_HEIGHT + 3 * panel_height + 2 * _GAP + _LEGEND_HEIGHT + _MARGIN
    scratch = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    notes = [line for note in rejection_notes for line in _text_lines(scratch, note, width - 2 * _MARGIN, _font(17))]
    canvas = Image.new("RGB", (width, basic_height + len(notes) * 23), _BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((_MARGIN, 18), case["id"] + " | two-view grounding", font=_font(29), fill=_INK)
    draw.text((_MARGIN, 57), "Top: observed inputs. Lower rows: original-pixel outputs owned by each target camera.",
              font=_font(18), fill=_MUTED)
    for number, panel in enumerate(panels):
        row, column = divmod(number, 3)
        canvas.paste(panel, (_MARGIN + column * (_PANEL_WIDTH + _GAP),
                            _TITLE_HEIGHT + row * (panel_height + _GAP)))
    y = _TITLE_HEIGHT + 3 * panel_height + 2 * _GAP + 18
    legend = (("Source evidence", _RED), ("Input SAM", _SAM), ("Primary mode", _GREEN),
              ("Secondary mode", _SECONDARY[0]), ("Secondary mode", _SECONDARY[1]))
    x = _MARGIN
    for label, color in legend:
        draw.rectangle((x, y + 2, x + 16, y + 18), fill=color)
        draw.text((x + 25, y), label, font=_font(17), fill=_INK)
        x += 28 + int(draw.textlength(label, font=_font(17))) + 35
    instructions = (
        "Tint = exact saved pixel mask. White-centered dots = saved samples at observed pixels.",
        "Each mode is drawn only in its recorded camera; no cross-camera projected copies. Ground truth is not drawn.",
        "REJECTED overlays, when present, are diagnostics only. Empty panels never invent a mode or a hidden surface.",
    )
    for number, line in enumerate(instructions):
        draw.text((_MARGIN, y + 30 + 24 * number), line, font=_font(17), fill=_MUTED)
    for number, line in enumerate(notes):
        draw.text((_MARGIN, y + 107 + 23 * number), line, font=_font(17), fill=_RED)
    return canvas


def render_suite(manifest, output):
    """Read completed artifacts and write one new static PNG per declared case."""
    manifest_path, description = read_manifest(manifest)
    output = Path(output).resolve()
    summary = json.loads((output / "summary.json").read_text())
    rows = _summary_rows(summary, description["cases"])
    destination = output / "visualizations"
    filenames = [destination / (case["id"] + ".png") for case in description["cases"]]
    for filename in filenames:
        if filename.exists():
            raise FileExistsError(f"Visualization already exists: {filename}")
    destination.mkdir(parents=True, exist_ok=True)
    for case, filename in zip(description["cases"], filenames):
        image = _render_case(manifest_path, case, output, rows)
        image.save(filename, format="PNG")
        print(json.dumps({"case": case["id"], "visualization": str(filename)}), flush=True)
    return filenames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Evaluated experiment directory containing summary.json")
    args = parser.parse_args()
    render_suite(args.manifest, args.output)


if __name__ == "__main__":
    main()
# END read-only multiview result rendering.
