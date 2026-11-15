"""Joint semantic routing contracts independent of model recognition quality.

Synthetic image evidence exercises actual candidate preparation and parsing.
The two entity bindings intentionally have different source/current relations
while current-view association and alternative targets remain independent.
"""
from copy import deepcopy
import json
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest

from manipulation.instance_relation import select_bindings


def binding():
    mask = np.zeros((8, 10), bool)
    mask[2:6, 3:7] = True
    image = Image.new("RGB", (10, 8), (45, 90, 120))
    return dict(source=dict(image=image, owner_mask=mask, description="Demonstrated entity"),
        views={name: dict(image=image, candidates=[dict(candidate_id=number, label="entity",
            score=.6, bbox=[3, 2, 7, 6], mask=mask)])
            for name, number in (("west", 7), ("east", 19))},
        selection_scope="entity", source_owner_id="source_owner", current_entity_id="current_owner",
        owner_role="carrier")


def decision(relation, *, same_entity=True):
    return dict(views={name: dict(candidate_ids=[number], reason="Visible current candidate")
                for name, number in (("west", 7), ("east", 19))},
        same_entity=same_entity, same_entity_reason="Explicit current-view association",
        source_target_relation=relation, relation_reason="Source/current visual identity evidence")


def test_two_roles_share_one_model_call_without_overwriting_identity_relations():
    bindings = {"held": binding(), "destination": binding()}
    bindings["destination"]["owner_role"] = "reference"
    raw = {"bindings": {"held": decision("same_instance", same_entity=False),
                        "destination": decision("same_category_different_instance")}}
    client = Mock(chat_json=Mock(return_value=raw))
    result = select_bindings(bindings, "Perform the declared relation", client=client)
    client.chat_json.assert_called_once()
    assert result["bindings"]["held"]["same_entity"] is False
    assert result["bindings"]["held"]["source_target_relation"] == "same_instance"
    assert result["bindings"]["destination"]["source_target_relation"] == "same_category_different_instance"
    prompt = client.chat_json.call_args.args[0]
    data = json.loads(prompt.split("\nInput data:\n", 1)[1])
    indices = [entry["image"] for item in data["bindings"].values() for entry in item["image_order"]]
    assert indices == list(range(1, len(client.chat_json.call_args.kwargs["images"]) + 1))
    assert data["bindings"]["destination"]["owner_role"] == "reference"
    assert "Return exactly {\"views\"" not in prompt


def test_category_relation_does_not_resolve_alternative_current_targets():
    selected = binding()
    extra = deepcopy(selected["views"]["west"]["candidates"][0])
    extra["candidate_id"] = 23
    selected["views"]["west"]["candidates"].append(extra)
    raw = decision("same_category_different_instance", same_entity=None)
    raw["views"]["west"]["candidate_ids"] = [7, 23]
    result = select_bindings({"target": selected}, "Choose the requested instance",
        client=Mock(chat_json=Mock(return_value={"bindings": {"target": raw}})))
    assert result["bindings"]["target"]["views"]["west"]["status"] == "alternatives"
    assert result["bindings"]["target"]["same_entity"] is None


@pytest.mark.parametrize("mutation", ["binding_id", "view_id", "foreign_candidate", "relation"])
def test_joint_protocol_rejects_foreign_ids_at_their_actual_scope(mutation):
    raw = {"bindings": {"target": decision("same_instance")}}
    if mutation == "binding_id":
        raw["bindings"]["other"] = raw["bindings"].pop("target")
    elif mutation == "view_id":
        raw["bindings"]["target"]["views"]["north"] = raw["bindings"]["target"]["views"].pop("west")
    elif mutation == "foreign_candidate":
        raw["bindings"]["target"]["views"]["west"]["candidate_ids"] = [19]
    else:
        raw["bindings"]["target"]["source_target_relation"] = "category_name"
    with pytest.raises(ValueError):
        select_bindings({"target": binding()}, "Use the declared entity",
                        client=Mock(chat_json=Mock(return_value=raw)))
