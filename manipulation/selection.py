"""Task-conditioned selection of observed owners, evidence or contact regions.

The same request builder consumes compiled demonstration semantics and actual
detector candidates for any skill. It generates no prompts with a second LLM,
does not estimate geometry, and never substitutes a part mask for its owner.
"""

# BEGIN generic grounding selection from source evidence and current intent.
import json
import math

import numpy as np
from PIL import Image, ImageDraw

from .grounding import _client, _detection, _image, _mask


_SELECTION_INSTRUCTION = (
    "Select observed candidates for the CURRENT instruction using the demonstrated "
    "functional role and the supplied visual evidence. The source_evidence fields "
    "describe a demonstration, not a command overriding the current instruction. "
    "Source-only position, level, colour, camera direction or world orientation "
    "must not become current-instance requirements unless the current instruction "
    "actually requests them. Preserve the demonstrated functional contact role "
    "for owner and interaction_region. "
    "Do not infer a new target from a skill name.\n"
    "selection_scope states exactly what is being selected: owner means the whole "
    "rigid entity or moving rigid part marked GREEN in the source image; it is not "
    "necessarily the largest enclosing object. interaction_region means the local "
    "functional contact region marked RED. A local part is not its whole owner; "
    "a whole object is not a precise interaction-region mask. Source query descriptions "
    "can mention surrounding geometry used for directions: these are contextual "
    "witnesses, not additional interaction targets and need not fit inside the red mask.\n"
    # BEGIN semantic entities do not imply a single independently moving body.
    "entity means the task-selected semantic object described by object_description. "
    "It may be an assembly containing several independently moving rigid bodies. "
    "The green source entity evidence can be partial; when supplied, a red source "
    "region explains the interaction but is not the entity's complete boundary. "
    "Select the requested entity instance without requiring all its visible parts "
    "to move rigidly together. Region-to-body ownership is a later geometric input, "
    "not a reason to reject a semantically correct assembly mask. If no red region "
    "is provided, do not invent a functional contact region.\n"
    # END existing owner scopes still require one physical rigid owner.
    # BEGIN offline-selected detection references are distinct from contact masks.
    "interaction_reference means the functional COMPONENT named in "
    "source_evidence.interaction_reference.text. The red source contact evidence "
    "helps identify this component but does not define its complete segmentation. "
    "The selected component may extend beyond that contact patch; it must belong "
    "to the rigid owner described by object_description and preserve the demonstrated "
    "interaction_part_description under the current instruction. Do not substitute "
    "an entire enclosing owner or a different independently moving part. Do not "
    "require a complete current owner mask before selecting a visibly supported "
    "component. A component selection does not certify the exact grasp point, "
    "all contact pixels, a grasp pose or geometric feasibility.\n"
    # END explicit component-level grounding reference.
    # BEGIN local visible evidence is neither a whole owner nor a contact mask.
    "owner_evidence instead means visible geometric evidence belonging to ONE "
    "target rigid entity, whose source evidence is marked GREEN. The source green "
    "mask and current candidates may be only partial surfaces of that entity: "
    "neither is required to cover its whole owner or a precise contact region. "
    "Match the owner's demonstrated role to the current instruction. Check visible "
    "membership using scene context; nearby objects and independently moving parts "
    "of an enclosing assembly are different rigid owners. Do not select them merely "
    "because their detector labels sound suitable. A furniture/assembly mask that "
    "includes other independently moving parts is not evidence of one rigid owner. "
    "Do not demand a complete-object segmentation to fill missing evidence.\n"
    # END explicit rigid-owner evidence scope.
    "Each current view has an original image and, when nonempty, numbered candidate "
    "panels. Panels enlarge existing pixels only. Use the original scene to determine "
    "which object or moving part a candidate belongs to. Candidate labels are detector "
    "queries, not proof of identity. Verify even a single candidate. Select only "
    "provided IDs within that view; equal IDs in different views do not establish "
    "physical correspondence. Images from other supplied views may provide context, "
    "but do not invent a mask in a view without a candidate.\n"
    "For entity, owner, interaction_region or interaction_reference, return candidate_ids as: one ID for a uniquely supported match; "
    "multiple IDs only if EVERY listed candidate independently satisfies the current "
    "semantic request, so choosing any is allowed; [] when no supplied candidate "
    "matches; null when evidence cannot resolve which candidate meets the request. "
    "Uncertainty between possible targets is NOT permission to use all of them. "
    # BEGIN evidence sets require an explicit same-owner semantic assertion.
    "For owner_evidence, return one or more IDs ONLY for visible evidence of the "
    "SAME target rigid entity. Multiple masks may provide complementary or "
    "overlapping evidence; they are NOT interchangeable target alternatives. "
    "The reason must identify the visible owner and justify every selected mask's "
    "membership. Return [] if none matches, or null if target identity or owner "
    "membership is unresolved; never combine uncertain instances. This scope "
    "additionally requires same_owner=true for a nonempty ID list, and "
    "same_owner=null for [] or null. This is a semantic assertion, not a proof "
    "of geometric registration or pixel-perfect segmentation.\n"
    # END evidence membership, distinct from per-target alternatives.
    "If a view has no candidates, return [] for it. Do not force an answer from weak "
    "or occluded evidence. Do not merge masks, generate coordinates, geometry, poses "
    "or claim grasp feasibility. This is semantic selection only.\n"
    "Return exactly {\"views\": {view_id: {\"candidate_ids\": [integer_ID, ...] "
    "or null, \"reason\": short_nonempty_string}, ...}}, with exactly the supplied "
    "views; for owner_evidence only, each view decision must also include the "
    "same_owner field specified above. Explain the visible evidence or the reason "
    "for abstaining briefly."
)


# BEGIN optional current-owner identity context for part selection.
_SELECTED_OWNER_INSTRUCTION = (
    "\nSome current views additionally supply selected_owner: this is an already "
    "selected CURRENT instance, separately highlighted GREEN in that view's "
    "owner-context images. Its ID is an opaque association label, not a visual "
    # BEGIN owner association applies to either declared local selection scope.
    "answer. For such a view, select only the local contact region or functional "
    "component requested by selection_scope, belonging to this particular owner "
    "and satisfying the current instruction. Use the original "
    # END explicit local target semantics, without changing owner association.
    "scene, the green owner evidence and each numbered candidate together to "
    "verify membership. A matching part name, mask overlap, adjacency or an "
    "enclosing assembly does not establish that membership. Do not replace the "
    "selected owner with a different instance or its independently moving part. "
    "Return candidate_ids=null when occlusion or uncertain ownership prevents "
    "association, and [] when the candidates visibly belong to other owners or "
    "none matches. The reason must explain this owner-part association or why "
    "it cannot be established. Keep all supplied masks separate; neither merge "
    "them nor clip a part mask to the owner. Views without selected_owner retain "
    "the ordinary instruction-conditioned selection rules.\n"
)

# BEGIN one shared request associates selected views of an entity or component.
_REFERENCE_ASSOCIATION_INSTRUCTION = (
    "\nFor selection_scope=entity or interaction_reference, return exactly views, "
    "same_entity, and same_entity_reason. "
    "Each views record still has only candidate_ids and reason. same_entity is true "
    "only when the unique selected candidate in each selected view depicts the SAME "
    "physical instance of the current semantic task target. For interaction_reference, "
    "this means the same particular component instance, not merely similarly named "
    "components or different components belonging to one enclosing object. This does not assert "
    "that its parts form one rigid body or that it remained still. Return false "
    "if selected views depict different instances, and null if instance identity "
    "cannot be established. A view with alternative candidate IDs is unresolved "
    "for this association; do not pick among those alternatives to force agreement. "
    "If only one view has a unique selected candidate, true means only a single "
    "selected instance in that view, not cross-view verification. If no view has "
    "a unique selected candidate, same_entity must be null. same_entity_reason "
    "must explain the actual visual association or its missing evidence.\n"
)
# END association is semantic evidence, not tracking or rigid-owner certification.


def _selected_owner(record, image, selection_scope):
    """Validate identity evidence without treating pixel overlap as identity."""
    # BEGIN either local contact or component selection may use owner context.
    if selection_scope not in {"interaction_region", "interaction_reference"}:
        raise ValueError("selected_owner is only applicable to interaction_region or interaction_reference selection")
    # END explicit local scope check.
    required = {"owner_id", "mask", "description", "source"}
    if not isinstance(record, dict) or set(record) != required:
        raise ValueError("selected_owner requires exactly owner_id, mask, description and source")
    for name in required - {"mask"}:
        if not isinstance(record[name], str) or not record[name].strip():
            raise ValueError(f"selected_owner.{name} must be a nonempty string")
    mask = _mask(record["mask"], image, "Current selected_owner.mask")
    return {"owner_id": record["owner_id"], "mask": mask,
            "description": record["description"].strip(), "source": record["source"].strip()}
# END current-owner context, separate from candidate selection or mask geometry.


def _box(mask):
    """Compute display bounds from an existing mask, without changing its pixels."""
    yy, xx = np.nonzero(mask)
    return (int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1)


def _highlight(image, mask, colour, *, outline=True):
    panel = Image.composite(Image.new("RGB", image.size, colour), image,
                            Image.fromarray(mask.astype(np.uint8) * 85))
    if outline:
        ImageDraw.Draw(panel).rectangle(_box(mask), outline=colour, width=1)
    return panel


def _context_crop(image, box):
    """Enlarge small candidate evidence for presentation, never for detection."""
    x0, y0, x1, y1 = box
    margin = max(10, math.ceil(max(x1 - x0, y1 - y0) * .6))
    crop = image.crop((max(0, math.floor(x0) - margin), max(0, math.floor(y0) - margin),
                       min(image.width, math.ceil(x1) + margin),
                       min(image.height, math.ceil(y1) + margin)))
    scale = min(256 / crop.width, 256 / crop.height)
    return crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                       Image.Resampling.NEAREST)


def _panels(image, candidates):
    """Keep candidate masks independent and show a local crop of each one."""
    columns = min(3, len(candidates))
    rows = math.ceil(len(candidates) / columns)
    sheet = Image.new("RGB", (columns * 272, rows * 290), (22, 28, 35))
    for index, candidate in enumerate(candidates):
        panel = _highlight(image, candidate["mask"], (255, 70, 40))
        # The detector box need only overlap its mask. Show the entire actual
        # mask so an outlying component cannot be hidden from semantic review.
        crop = _context_crop(panel, _box(candidate["mask"]))
        left, top = index % columns * 272, index // columns * 290
        sheet.paste(crop, (left + (272 - crop.width) // 2, top + 26 + (256 - crop.height) // 2))
        # Only IDs are rendered; natural-language labels remain in JSON.
        ImageDraw.Draw(sheet).text((left + 8, top + 6), str(candidate["candidate_id"]), fill="white")
    return sheet


def _prepare(source, instruction, views, selection_scope):
    """Allowlist semantic evidence; source poses and simulator IDs are not sent."""
    # BEGIN component references retain a scope distinct from exact contacts.
    if selection_scope not in {"entity", "owner", "interaction_region", "interaction_reference", "owner_evidence"}:
        raise ValueError("Unknown semantic selection scope")
    # END declared detection-reference selection scope.
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("A current instruction is required")
    # BEGIN an entity can be selected without a functional-contact region.
    required_source = {"image", "description", "owner_mask" if selection_scope == "entity" else "mask"}
    if not isinstance(source, dict) or not required_source <= source.keys():
        raise ValueError("Source requires image, description and evidence for its selected scope")
    # END no contact region is manufactured for a contact-free reference.
    description = source["description"]
    queries = source.get("query_descriptions", [])
    if not isinstance(description, str) or not description.strip():
        raise ValueError("Source description must be nonempty")
    if (not isinstance(queries, list)
            or any(not isinstance(query, str) or not query.strip() for query in queries)):
        raise ValueError("Source query_descriptions must be a list of nonempty strings")
    # BEGIN copy only declared offline semantics into the fixed runtime request.
    source_evidence = {"description": description.strip(), "query_descriptions": queries}
    semantic_fields = {"object_description", "interaction_part_description", "interaction_reference"}
    if source.keys() & semantic_fields:
        if not semantic_fields <= source.keys():
            raise ValueError("Offline interaction semantics require object, part and detection reference")
        for key in semantic_fields - {"interaction_reference"}:
            # BEGIN contact-free entity evidence has no required contact-part name.
            empty_part_allowed = selection_scope in {"entity", "owner_evidence"} and key == "interaction_part_description"
            if not isinstance(source[key], str) or (not empty_part_allowed and not source[key].strip()):
                raise ValueError(f"Source {key} must be a nonempty string")
            # END entity identity remains mandatory; no part description is invented.
        reference = source["interaction_reference"]
        if (not isinstance(reference, dict) or set(reference) != {"text", "scope"}
                or reference["scope"] not in {"object", "part"}
                or not isinstance(reference["text"], str) or not reference["text"].strip()):
            raise ValueError("Source interaction_reference requires text and scope object or part")
        source_evidence.update({key: source[key] for key in semantic_fields})
    if (selection_scope == "interaction_reference"
            and source_evidence.get("interaction_reference", {}).get("scope") != "part"):
        raise ValueError("Component selection requires an offline part interaction_reference")
    # END offline semantic input; poses and provenance are not model inputs.
    source_image = _image(source["image"])
    # BEGIN evidence-only sources need no manufactured interaction/owner mask.
    # BEGIN source entity masks and region-owner poses have separate meanings.
    entity_scope = selection_scope == "entity"
    evidence_scope = selection_scope == "owner_evidence" or (entity_scope and "mask" not in source)
    source_region = _mask(source["owner_mask"] if entity_scope and "mask" not in source else source["mask"], source_image,
                          "Source owner evidence mask" if evidence_scope else "Source interaction mask")
    # END original owner scopes keep their former evidence requirements.
    owner = None
    if "owner_mask" in source:
        owner = _mask(source["owner_mask"], source_image, "Source owner mask")
    if selection_scope == "owner" and owner is None:
        raise ValueError("Owner selection requires the source owner_mask")
    panel = _highlight(source_image, source_region,
                       (30, 225, 110) if evidence_scope else (255, 65, 65), outline=False)
    if owner is not None:
        # An outline gives source context without occluding the red contact area.
        ImageDraw.Draw(panel).rectangle(_box(owner), outline=(30, 225, 110), width=2)
    source_extent = owner if owner is not None else source_region
    images = [panel, _context_crop(panel, _box(source_extent))]
    image_order = [("source scene: green visible owner evidence, optional green owner bounds"
                    if evidence_scope else
                    "source scene: red interaction region, green owner bounds when available"),
                   "source detail: same annotation, enlarged existing pixels"]
    # BEGIN display semantic-object evidence without declaring physical rigidity.
    if entity_scope:
        image_order[0] = ("source scene: green semantic entity evidence; no contact region supplied"
                          if evidence_scope else "source scene: red supplied region, green semantic entity context")
    # END physical region ownership is never inferred from this green mask.
    # END scope-specific source annotation; current candidate masks stay separate.
    if not isinstance(views, dict) or not views:
        raise ValueError("views must be a nonempty view-ID mapping")
    prepared, descriptions = {}, {}
    has_selected_owner = False
    for name, view in views.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("View IDs must be nonempty strings")
        if not isinstance(view, dict) or not {"image", "candidates"} <= view.keys():
            raise ValueError("Each view requires image and candidates")
        image = _image(view["image"])
        raw = view["candidates"]
        if not isinstance(raw, list):
            raise ValueError("Candidates must be a list of actual detections")
        if any(not isinstance(item, dict) or not isinstance(item.get("label"), str)
               or not item["label"].strip() for item in raw):
            raise ValueError("Each candidate needs a nonempty detector label")
        labels = list(dict.fromkeys(item["label"] for item in raw))
        candidates, used = [], set()
        for index, item in enumerate(raw, 1):
            number = item.get("candidate_id", index)
            if type(number) is not int or number < 1 or number in used:
                raise ValueError("Candidate IDs must be distinct positive integers within a view")
            used.add(number)
            candidates.append(_detection(item, image, labels, number))
        prepared[name] = candidates
        descriptions[name] = {
            "allowed_ids": [item["candidate_id"] for item in candidates],
            "candidates": [{"candidate_id": item["candidate_id"], "label": item["label"],
                            "bbox": item["bbox"]} for item in candidates],
        }
        images.append(image)
        image_order.append(f"view {name}: original current scene")
        # BEGIN show the actual already-selected owner, never an inferred mask union.
        if "selected_owner" in view:
            selected_owner = _selected_owner(view["selected_owner"], image, selection_scope)
            descriptions[name]["selected_owner"] = {
                key: selected_owner[key] for key in ("owner_id", "description", "source")}
            owner_panel = _highlight(image, selected_owner["mask"],
                                     (30, 225, 110), outline=False)
            images.extend((owner_panel, _context_crop(owner_panel, _box(selected_owner["mask"]))))
            image_order.extend((f"view {name}: selected current owner highlighted GREEN",
                                f"view {name}: same selected owner, enlarged existing pixels"))
            has_selected_owner = True
        # END no candidate is removed, intersected or associated by mask overlap.
        if candidates:
            images.append(_panels(image, candidates))
            image_order.append(f"view {name}: numbered candidate details")
    context = {"selection_scope": selection_scope, "current_instruction": instruction.strip(),
               "source_evidence": source_evidence,
               "views": descriptions,
               "image_order": [{"image": index + 1, "content": content}
                               for index, content in enumerate(image_order)]}
    # BEGIN absent current-owner context leaves the original request unchanged.
    prompt = (_SELECTION_INSTRUCTION
              + (_SELECTED_OWNER_INSTRUCTION if has_selected_owner else "")
              + (_REFERENCE_ASSOCIATION_INSTRUCTION
                 if selection_scope in {"entity", "interaction_reference"} else "")
              + "\nInput data:\n" + json.dumps(context, ensure_ascii=False))
    # END optional contextual selection, without skill-name prompt branches.
    return prompt, images, prepared


def select_grounding_candidates(source, instruction, views, *, selection_scope, client=None):
    """Select semantic candidates using a fixed, automatically assembled prompt.

    ``entity`` selects a semantic object instance, which may contain multiple
    physical bodies. It requires source owner_mask as visible entity context;
    a source contact mask is optional. This scope does not bind a region to a
    body and does not replace the existing physical owner selection scopes.

    ``source`` consumes existing ``image``, interaction ``mask``, ``description``
    and optional ``query_descriptions``. ``selection_scope='owner'`` additionally
    requires ``source.owner_mask``. These are compiled demonstration evidence;
    only ``instruction`` describes the current request. No source file path,
    skill-name branch, source pose or current oracle geometry is used.

    ``selection_scope='interaction_reference'`` selects the functional component
    named in the offline ``source.interaction_reference`` (scope ``part``), with
    its ``object_description`` and ``interaction_part_description``. The source
    red contact mask locates the demonstrated interaction; it does not require
    the selected component to have exactly the same contact-patch extent. This
    scope needs no complete current owner mask and never manufactures one.

    ``selection_scope='owner_evidence'`` interprets ``source.mask`` as visible
    evidence of the rigid entity described by the source, not as a contact mask.
    Its optional ``owner_mask`` provides wider source context only. Neither
    source nor current evidence must cover a complete owner. A nonempty current
    selection requires an explicit VLM ``same_owner=True`` assertion: all masks
    are complementary/overlapping evidence of one target, never alternatives
    between instances. Empty/unresolved decisions require ``same_owner=None``.
    This semantic membership claim does not establish measured motion rigidity.

    Each ``views[view_id]`` contains an image and detector candidates. Explicit
    positive integer candidate IDs are preserved; otherwise the input index
    starting at one is used. A normal single-view call supplies one mapping
    entry. For ``entity`` and ``interaction_reference``, the same request also
    reports ``same_entity`` as true, false or unknown for the uniquely selected
    instances across views. Component references must identify the same actual
    component, not merely components of one owner. This explicit semantic
    assertion does not establish rigidity or geometric registration. Other
    scopes share context only; their local IDs establish no association.

    For ``interaction_region`` or ``interaction_reference``, a view may supply ``selected_owner`` with
    exactly owner_id, mask, description and source. That actual current mask is
    shown independently in green, and the same request must verify that each
    selected part belongs to that instance. Spatial overlap alone is not used
    to associate, select, clip or merge masks. Unclear membership remains null.
    Omitting this record preserves the original request and behavior.

    Each result has a status: selected, alternatives (all semantically allowed),
    unresolved, no_match, or no_candidates. ``regions`` retains the separate
    selected masks. Owner and local interaction scopes are explicit in the
    returned record; this function never silently constructs an owner_mask.
    Geometry, correspondence, accessibility and grasp feasibility remain the
    responsibilities of the existing downstream consumers.

    Owner-evidence selections use ``selected`` for one or several masks, carry
    the semantic ``same_owner`` assertion, and never merge their pixels or
    upgrade them to a complete ``owner_mask``. Entity/component-reference
    selections retain ``alternatives`` but must report unknown association
    when any view offers interchangeable targets instead of one instance.
    """
    prompt, images, candidates = _prepare(source, instruction, views, selection_scope)
    called = any(candidates.values())
    if called:
        response = _client(client).chat_json(prompt, images=images)
    else:
        response = {"views": {name: {"candidate_ids": [], "reason": "No detector candidates were supplied."}
                              for name in candidates}}
        if selection_scope in {"entity", "interaction_reference"}:
            response.update(same_entity=None, same_entity_reason="No observed candidates to associate")
        # BEGIN the no-candidate fast path obeys the same evidence schema.
        if selection_scope == "owner_evidence":
            for decision in response["views"].values():
                decision["same_owner"] = None
        # END local empty decision, with no fabricated membership assertion.
    response_fields = ({"views", "same_entity", "same_entity_reason"}
                       if selection_scope in {"entity", "interaction_reference"} else {"views"})
    if (not isinstance(response, dict) or set(response) != response_fields
            or not isinstance(response["views"], dict) or set(response["views"]) != set(candidates)):
        raise ValueError("Selection must return exactly the supplied view IDs")
    result = {}
    for name, items in candidates.items():
        decision = response["views"][name]
        # BEGIN same-owner evidence is an explicit, scope-specific contract.
        evidence_scope = selection_scope == "owner_evidence"
        required = {"candidate_ids", "reason", "same_owner"} if evidence_scope else {"candidate_ids", "reason"}
        if not isinstance(decision, dict) or set(decision) != required:
            raise ValueError(f"Each view decision requires exactly {', '.join(sorted(required))}")
        ids, reason = decision["candidate_ids"], decision["reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Selection reason must be a nonempty string")
        by_id = {item["candidate_id"]: item for item in items}
        if ids is not None:
            if (not isinstance(ids, list) or any(type(number) is not int or number not in by_id for number in ids)
                    or len(set(ids)) != len(ids)):
                raise ValueError("Selected IDs must be distinct integers from this view's candidates")
        if evidence_scope and ((ids and decision["same_owner"] is not True)
                               or (not ids and decision["same_owner"] is not None)):
            raise ValueError("Owner evidence requires same_owner=true for selected IDs, null otherwise")
        if not items:
            if ids != []:
                raise ValueError("A view with no candidates must return an empty list")
            status = "no_candidates"
        elif ids is None:
            status = "unresolved"
        elif not ids:
            status = "no_match"
        else:
            status = "selected" if evidence_scope or len(ids) == 1 else "alternatives"
        result[name] = {"status": status, "candidate_ids": ids, "reason": reason.strip(),
                        "regions": [by_id[number] for number in ids] if ids else []}
        if evidence_scope:
            result[name]["same_owner"] = decision["same_owner"]
        # END same-owner evidence; legacy scopes keep their independent alternatives.
    output = {"selection_scope": selection_scope, "views": result, "vlm_used": called}
    # BEGIN preserve cross-view entity evidence instead of assigning a shared ID.
    if selection_scope in {"entity", "interaction_reference"}:
        association, reason = response["same_entity"], response["same_entity_reason"]
        unique = [name for name, row in result.items() if row["status"] == "selected"]
        ambiguous = any(row["status"] == "alternatives" for row in result.values())
        if ((association is not None and type(association) is not bool)
                or not isinstance(reason, str) or not reason.strip()
                or ((not unique or ambiguous) and association is not None)):
            raise ValueError("Entity association needs bool/null evidence for uniquely selected candidates")
        output.update(same_entity=association, same_entity_reason=reason,
                      association_scope="cross_view" if len(unique) > 1 else "single_selected_view" if unique else "no_unique_selection",
                      associated_views=unique)
    # END uncertain associations stay explicit; their masks are never merged.
    return output
# END generic grounding selection.
