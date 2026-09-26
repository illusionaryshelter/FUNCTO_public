"""CPU contract checks; SAM3.1 and Gemini calls are replaced at their boundaries.

These tests verify instance association and prompt forwarding. They do not claim
to test model quality or contact any model service.
"""

# BEGIN regression checks for the local scene-graph migration.
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from cloud_services import scene_graph, scene_graph_new
from cloud_services.graph_format import parse_graph_assignments


class SceneGraphContractTest(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (8, 8))
        self.masks = [np.zeros((8, 8), dtype=bool) for _ in range(3)]
        for index, mask in enumerate(self.masks):
            mask[index, index] = True
        self.predictions = [
            {"label": name, "score": 0.9, "bbox": [0, 0, 8, 8], "mask": mask}
            for name, mask in zip(("bowl", "orange", "orange"), self.masks)
        ]

    # BEGIN regress the real Gemini response without contacting any service.
    def test_real_fenced_graph_preserves_every_predicate_in_both_interfaces(self):
        response = (Path(__file__).parent / "fixtures" / "gemini_scene_graph.txt").read_text()
        raw_nodes, raw_edges = parse_graph_assignments(response)
        self.assertEqual(raw_nodes[0], ("table1", ""))
        self.assertEqual(raw_nodes[-1], ("orange4", "Fruits, with_mark"))
        self.assertEqual(len(raw_edges), 5)
        expected = [("bowl1", "wood")] + [
            (f"orange{index}", "Fruits" if index < 4 else "Fruits, with_mark")
            for index in range(1, 5)
        ]
        for module in (scene_graph, scene_graph_new):
            with self.subTest(module=module.__name__):
                nodes, edges, classes = module.SceneGraph.parse_scene_graph(response)
                if module is scene_graph_new:
                    self.assertEqual(nodes[-1].predicates, ["Fruits", "with_mark"])
                    self.assertEqual(nodes[-1].object_type, "orange")
                    nodes = [(node.name, ", ".join(node.predicates)) for node in nodes]
                self.assertEqual(nodes, expected)
                self.assertEqual(edges, [])
                self.assertEqual(set(classes), {"bowl", "orange"})

    def test_literal_node_variants_and_relations_keep_original_semantics(self):
        response = (
            "nodes = [('item1',), ('item2', None, 'red', 'marked'), ('bin1', 'wood'), ('Chair1',)]\n"
            "edges = [('INSIDE', 'item1', 'bin1'), ('ON', 'item2', 'Chair1')]"
        )
        for module in (scene_graph, scene_graph_new):
            with self.subTest(module=module.__name__):
                nodes, edges, classes = module.SceneGraph.parse_scene_graph(response)
                if module is scene_graph_new:
                    nodes = [(node.name, ", ".join(node.predicates)) for node in nodes]
                    edges = [(edge.relationship, edge.source, edge.destination) for edge in edges]
                self.assertEqual(nodes, [("item1", ""), ("item2", "red, marked"), ("bin1", "wood")])
                self.assertEqual(edges, [("INSIDE", "item1", "bin1")])
                self.assertEqual(set(classes), {"item", "bin"})

    def test_graph_rejects_nonliteral_code_and_malformed_graphs(self):
        responses = [
            "nodes = []\nedges = []\nimport os",
            "nodes = list()\nedges = []",
            "nodes = []\nedges = []\nextra = 1",
            "nodes = []\nnodes = []\nedges = []",
            "nodes = []",
            "nodes = [('item1', 2)]\nedges = []",
            "nodes = [('item1',), ('item1',)]\nedges = []",
            "nodes = []\nedges = [('ON', 'item1')]",
            "Explanation:\n```python\nnodes=[]\nedges=[]\n```",
        ]
        for response in responses:
            with self.subTest(response=response), self.assertRaises((ValueError, SyntaxError)):
                parse_graph_assignments(response)

    def test_alignment_accepts_complete_fence_and_multiple_predicates(self):
        response = '```python\n[(2, ("orange1", "Fruits", "with_mark")), (1, ("orange2",))]\n```'
        self.assertEqual(scene_graph_new.SceneGraph().parse_alignments(response), [
            (2, "orange1"), (1, "orange2"),
        ])

    def test_alignment_rejects_ambiguous_or_malformed_pairs(self):
        responses = [
            "{1: ('orange1',)}",
            "[(True, ('orange1',))]",
            "[('1', ('orange1',))]",
            "[(1, ('orange1',)), (1, ('orange2',))]",
            "[(1, ('orange1',)), (2, ('orange1',))]",
            "[(1, (None,))]",
            "[(1, ('orange1', 2))]",
            "[(1, ('orange1',), 'extra')]",
        ]
        for response in responses:
            with self.subTest(response=response), self.assertRaises(ValueError):
                scene_graph_new.SceneGraph().parse_alignments(response)
    # END real response and literal contract regressions.

    def test_both_graphs_keep_all_instances_and_corresponding_masks(self):
        for module in (scene_graph, scene_graph_new):
            with self.subTest(module=module.__name__), patch.object(
                module, "detect", return_value=self.predictions
            ) as detect:
                graph = module.SceneGraph()
                result = graph.detect_and_segment(self.image, ["bowl", "orange"])
                detect.assert_called_once_with(self.image, ["bowl", "orange"])
                masks, names = result[:2]
                self.assertEqual(names, ["bowl", "orange", "orange"])
                for expected, actual in zip(self.masks, masks):
                    self.assertIs(actual["segmentation"], expected)
                if module is scene_graph_new:
                    self.assertEqual(result[2], ["orange"])

    def test_empty_graph_does_not_launch_a_perception_worker(self):
        for module in (scene_graph, scene_graph_new):
            with self.subTest(module=module.__name__), patch.object(module, "detect") as detect:
                result = module.SceneGraph().detect_and_segment(self.image, [])
                self.assertTrue(all(value == [] for value in result))
                detect.assert_not_called()

    def test_graph_prompt_preserves_system_instruction_and_image(self):
        for module in (scene_graph, scene_graph_new):
            with self.subTest(module=module.__name__):
                graph = module.SceneGraph()
                prompt, system_prompt = graph.get_scene_graph_prompt("Fruits")
                with patch.object(graph.gemini, "chat", return_value="nodes=[]\nedges=[]") as chat:
                    self.assertEqual(
                        graph.get_scene_graph_string(self.image, "Fruits"),
                        "nodes=[]\nedges=[]",
                    )
                    chat.assert_called_once_with(
                        prompt, images=[self.image], system_prompt=system_prompt
                    )

    def test_numbered_ambiguous_subset_uses_one_based_indices(self):
        graph = scene_graph_new.SceneGraph()
        data = scene_graph_new.SceneGraphData(nodes=[
            scene_graph_new.SceneNode("bowl1", "bowl"),
            scene_graph_new.SceneNode("orange1", "orange"),
            scene_graph_new.SceneNode("orange2", "orange"),
        ])
        # The singleton bowl precedes the ambiguous subset. This catches both
        # the former full-list indexing bug and its one-based offset error.
        masks = [{"segmentation": mask} for mask in self.masks]
        with patch.object(graph, "get_annotated_segmentation", return_value=self.image), patch.object(
            graph, "get_alignments_string",
            return_value="[(2, ('orange1', 'fruits')), (1, ('orange2', 'fruits'))]",
        ):
            result, _ = graph.assign_masks_to_nodes(
                self.image, data, masks, ["bowl", "orange", "orange"],
                ["orange"], "graph context",
            )
        self.assertIs(result.nodes[0].mask, self.masks[0])
        self.assertIs(result.nodes[1].mask, self.masks[2])
        self.assertIs(result.nodes[2].mask, self.masks[1])

    def test_invalid_visual_label_fails_instead_of_selecting_another_mask(self):
        graph = scene_graph_new.SceneGraph()
        data = scene_graph_new.SceneGraphData(nodes=[
            scene_graph_new.SceneNode("orange1", "orange"),
        ])
        with patch.object(graph, "get_annotated_segmentation", return_value=self.image), patch.object(
            graph, "get_alignments_string", return_value="[(0, ('orange1', 'fruits'))]",
        ):
            with self.assertRaisesRegex(ValueError, "Invalid annotated mask number"):
                graph.assign_masks_to_nodes(
                    self.image, data, [{"segmentation": self.masks[0]}],
                    ["orange"], ["orange"], "graph context",
                )

    # BEGIN reject incomplete or cross-class alignments before mask assignment.
    def test_alignment_requires_all_requested_names_and_matching_mask_classes(self):
        responses = [
            "[(1, ('unknown1',)), (2, ('bowl1',))]",
            "[(1, ('orange1',))]",
            "[(3, ('orange1',)), (2, ('bowl1',))]",
            "[(2, ('orange1',)), (1, ('bowl1',))]",
        ]
        graph = scene_graph_new.SceneGraph()
        for response in responses:
            data = scene_graph_new.SceneGraphData(nodes=[
                scene_graph_new.SceneNode("orange1", "orange"),
                scene_graph_new.SceneNode("bowl1", "bowl"),
            ])
            with self.subTest(response=response), patch.object(
                graph, "get_annotated_segmentation", return_value=self.image
            ), patch.object(graph, "get_alignments_string", return_value=response):
                with self.assertRaises(ValueError):
                    graph.assign_masks_to_nodes(
                        self.image, data,
                        [{"segmentation": mask} for mask in self.masks[:2]],
                        ["orange", "bowl"], ["orange", "bowl"], "graph context",
                    )
                self.assertTrue(all(node.mask is None for node in data.nodes))
    # END complete mapping validation.

    # BEGIN prevent one detected instance from being reused for missing objects.
    def test_single_mask_cannot_satisfy_multiple_requested_instances(self):
        graph = scene_graph_new.SceneGraph()
        data = scene_graph_new.SceneGraphData(nodes=[
            scene_graph_new.SceneNode("orange1", "orange"),
            scene_graph_new.SceneNode("orange2", "orange"),
        ])
        with patch.object(graph, "get_annotated_segmentation") as annotate, patch.object(
            graph, "get_alignments_string"
        ) as align:
            with self.assertRaisesRegex(ValueError, "requested 2, detected 1"):
                graph.assign_masks_to_nodes(
                    self.image, data, [{"segmentation": self.masks[0]}],
                    ["orange"], [], "graph context",
                )
            annotate.assert_not_called()
            align.assert_not_called()
            self.assertTrue(all(node.mask is None for node in data.nodes))

    def test_missing_class_fails_before_assigning_other_nodes(self):
        graph = scene_graph_new.SceneGraph()
        data = scene_graph_new.SceneGraphData(nodes=[
            scene_graph_new.SceneNode("bowl1", "bowl"),
            scene_graph_new.SceneNode("orange1", "orange"),
        ])
        with patch.object(graph, "get_annotated_segmentation") as annotate, patch.object(
            graph, "get_alignments_string"
        ) as align:
            with self.assertRaisesRegex(ValueError, "'orange': requested 1, detected 0"):
                graph.assign_masks_to_nodes(
                    self.image, data, [{"segmentation": self.masks[0]}],
                    ["bowl"], [], "graph context",
                )
            annotate.assert_not_called()
            align.assert_not_called()
            self.assertTrue(all(node.mask is None for node in data.nodes))
    # END missing instance regressions.


if __name__ == "__main__":
    unittest.main()
# END regression checks for the local scene-graph migration.
