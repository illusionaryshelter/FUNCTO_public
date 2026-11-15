"""Joint semantic selection and source/current instance relation per binding.

All supplied bindings whose image/candidate inputs are ready share one VLM
request. Candidate IDs remain local to their view and binding. The result
distinguishes current-view association from source/current identity; neither
field represents metric geometry, physical rigidity or task feasibility.
"""
from __future__ import annotations

import json

from .grounding import _client
from .selection import prepare_selection_evidence, parse_selection_response


INSTANCE_RELATIONS = frozenset({"same_instance", "same_category_different_instance",
                                "different_category", "unresolved"})

_RELATION_INSTRUCTION = """
Select the current candidates and identify the source/current instance relation
for EACH supplied binding in ONE response. Binding IDs and entity IDs are
opaque references supplied by the caller, not evidence of physical identity.
The source pictures depict the demonstrated entity/component and current
pictures depict the instruction's candidate instances. Do not infer instance
identity from a category or matching label. source_target_relation is one of:
same_instance (the demonstrated physical instance),
same_category_different_instance (another instance of the same task category),
different_category, unresolved. Explain the source/current visual evidence in
relation_reason; use unresolved when the evidence does not establish a relation.
same_entity instead associates the uniquely selected CURRENT candidates across
views. It does not compare a current candidate with the demonstration. An
alternative target list remains alternatives even if its category is known.
Use exactly this response schema, with the supplied binding and view IDs:
{"bindings": {binding_id: {
  "views": {view_id: {"candidate_ids": [integer_ID, ...] or null,
                       "reason": nonempty_string}},
  "same_entity": true or false or null,
  "same_entity_reason": nonempty_string,
  "source_target_relation": enum_above,
  "relation_reason": nonempty_string}}}.
For owner_evidence scope, each view also includes the required same_owner field.
Return no metric coordinates, poses or physical acceptance judgments.
"""


def select_bindings(bindings, instruction, *, client=None):
    """Return joint semantic evidence for explicit source/current bindings.

    Each mapping value provides ``source``, ``views`` with real candidates,
    ``selection_scope``, ``source_owner_id``, ``current_entity_id`` and
    ``owner_role``. Images from every binding enter this one model call. The
    same parser resolves the actual masks used by local semantic selection.
    Empty candidate sets produce explicit missing evidence without inference.
    """
    if not bindings:
        raise ValueError("Joint selection requires explicit entity bindings")
    images, contexts, prepared, rules = [], {}, {}, []
    for binding_id, binding in bindings.items():
        if not isinstance(binding_id, str) or not binding_id.strip():
            raise ValueError("Binding IDs must be nonempty strings")
        rule, pictures, candidates, context = prepare_selection_evidence(binding["source"], instruction,
            binding["views"], binding["selection_scope"])
        offset = len(images)
        context["image_order"] = [{**item, "image": item["image"] + offset}
                                  for item in context["image_order"]]
        context.update({key: binding[key] for key in
                       ("source_owner_id", "current_entity_id", "owner_role")})
        contexts[binding_id] = context
        prepared[binding_id] = candidates
        images.extend(pictures)
        rules.append(f"Binding {binding_id} selection rules:\n{rule}")
    called = any(any(views.values()) for views in prepared.values())
    prompt = ("\n".join(rules) + _RELATION_INSTRUCTION + "\nInput data:\n" +
              json.dumps({"current_instruction": instruction, "bindings": contexts}, ensure_ascii=False))
    if called:
        response = _client(client).chat_json(prompt, images=images)
    else:
        response = {"bindings": {}}
        for binding_id, candidates in prepared.items():
            evidence = bindings[binding_id]["selection_scope"] == "owner_evidence"
            response["bindings"][binding_id] = {
                "views": {name: {"candidate_ids": [], "reason": "No detector candidates supplied.",
                                  **({"same_owner": None} if evidence else {})}
                          for name in candidates},
                "same_entity": None, "same_entity_reason": "No current instance selected.",
                "source_target_relation": "unresolved",
                "relation_reason": "No current candidate supplies source/current evidence."}
    if (not isinstance(response, dict) or set(response) != {"bindings"}
            or not isinstance(response["bindings"], dict)
            or set(response["bindings"]) != set(bindings)):
        raise ValueError("Joint selection must return exactly the supplied binding IDs")
    results = {}
    fields = {"views", "same_entity", "same_entity_reason", "source_target_relation", "relation_reason"}
    for binding_id, candidates in prepared.items():
        row = response["bindings"][binding_id]
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError("A binding response requires selection, association and instance relation")
        relation, reason = row["source_target_relation"], row["relation_reason"]
        if relation not in INSTANCE_RELATIONS or not isinstance(reason, str) or not reason.strip():
            raise ValueError("Source/current relation requires its enum and nonempty evidence")
        selected = parse_selection_response(
            {key: row[key] for key in ("views", "same_entity", "same_entity_reason")}, candidates,
            selection_scope=bindings[binding_id]["selection_scope"], vlm_used=called, associate=True)
        results[binding_id] = {**selected, "binding_id": binding_id,
            "source_target_relation": relation, "relation_reason": reason.strip(),
            "relation_evidence": {key: bindings[binding_id][key] for key in
                                  ("source_owner_id", "current_entity_id", "owner_role")}}
    return {"bindings": results, "vlm_used": called, "raw_response": response}


def region_strategy(relation):
    """Dispatch an established semantic relation without changing its evidence."""
    if relation not in INSTANCE_RELATIONS:
        raise ValueError("Unknown source/current instance relation")
    return {"same_instance": "same_instance",
            "same_category_different_instance": "cross_instance",
            "different_category": None, "unresolved": None}[relation]
