from PIL import Image
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Any
# BEGIN local perception and the shared Gemini API integration.
from cloud_services.apis.language_model import Gemini
from cloud_services.image_utils import annotate_masks
from local_models.sam31 import detect
# END local perception and the shared Gemini API integration.
# BEGIN shared literal-only scene graph parser and prompt.
from cloud_services.graph_format import GRAPH_INSTRUCTION, parse_graph_assignments, strip_literal_fence
# END shared literal-only scene graph parser and prompt.
from collections import Counter
import ast

@dataclass
class SceneNode:
    name: str
    object_type: Optional[str]
    predicates: List[str] = field(default_factory=list)
    mask: Optional[Any] = None

    def __str__(self):
        return f"( {self.name}, {', '.join(self.predicates)})"



@dataclass
class SceneEdge:
    relationship: str
    source: str
    destination: str


@dataclass
class SceneGraphData:
    nodes: List[SceneNode] = field(default_factory=list)
    edges: List[SceneEdge] = field(default_factory=list)

    def add_node(self, node: SceneNode):
        self.nodes.append(node)

    def add_edge(self, edge: SceneEdge):
        self.edges.append(edge)


class SceneGraph:
    def __init__(self):
        # BEGIN use the same Gemini client as the main FUNCTO pipeline.
        self.gemini = Gemini()
        # END shared Gemini client.

    @staticmethod
    def get_scene_graph_prompt(predicate: str) -> Tuple[str, str]:
        prompt = f"The additional predicate is [{predicate}]"
        # BEGIN require a complete literal graph with every matching predicate.
        meta_prompt = GRAPH_INSTRUCTION
        # END complete graph response contract.
        return prompt, meta_prompt

    # BEGIN specify the literal format and one-based visual labels to Gemini.
    @staticmethod
    def get_alignments_prompt(nodes: str) -> str:
        prompt = f'''Align the number in the marked image with the following nodes: 
{nodes}

Return only a Python literal list of (number, (node_name, predicate1, predicate2))
tuples, without Markdown fences or explanation. Include all predicate fields for
each node, or (node_name,) if it has none. Use the image's labels, which start at
1, and preserve the requested node names exactly. Match every requested node
exactly once, never reuse an image number, and include no unrequested nodes.
        '''
        return prompt
    # END alignment response contract.

    @staticmethod
    def parse_scene_graph(scene_graph: str) -> Tuple[List[SceneNode], List[SceneEdge], List[str]]:
        # BEGIN parse all legal literal quotes and predicate fields safely.
        node_values, edge_values = parse_graph_assignments(scene_graph)
        nodes = [SceneNode(name=obj, object_type=re.sub(r'\d+', '', obj), predicates=ot.split(', ') if ot else []) 
                 for obj, ot in node_values]
        edges = [SceneEdge(relationship=rel, source=src, destination=dest) 
                 for rel, src, dest in edge_values]
        # END shared graph parsing.

        filtered_nodes = [node for node in nodes if not ('table' in node.name.lower() or 'chair' in node.name.lower())]
        # BEGIN apply the existing exclusion to either endpoint consistently.
        filtered_edges = [edge for edge in edges if not any(
            'table' in element.lower() or 'chair' in element.lower()
            for element in [edge.source, edge.destination]
        )]
        # END endpoint exclusion.

        object_classes = list(set(re.sub(r'\d+', '', node.name) for node in filtered_nodes))

        return filtered_nodes, filtered_edges, object_classes

    def parse_alignments(self, alignments_string: str):
        # BEGIN accept only complete literal responses with unambiguous mapping.
        data_list = ast.literal_eval(strip_literal_fence(alignments_string))
        if not isinstance(data_list, list):
            raise ValueError("Alignments must be a literal list")
        match_relation, used_numbers, used_names = [], set(), set()
        for pair in data_list:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError(f"Invalid alignment pair: {pair!r}")
            number, node = pair
            if type(number) is not int or number < 1:
                raise ValueError(f"Invalid annotated mask number: {number!r}")
            fields = (node,) if isinstance(node, str) else node
            if not (isinstance(fields, (tuple, list)) and fields
                    and isinstance(fields[0], str) and fields[0].strip()
                    and all(field is None or isinstance(field, str) for field in fields[1:])):
                raise ValueError(f"Invalid alignment node: {node!r}")
            name = fields[0]
            if number in used_numbers or name in used_names:
                raise ValueError("Alignments must use unique node names and mask numbers")
            used_numbers.add(number)
            used_names.add(name)
            match_relation.append((number, name))
        return match_relation
        # END validated literal alignment mapping.

    def get_scene_graph_string(self, image: Image.Image, predicate: str) -> str:
        prompt, meta_prompt = self.get_scene_graph_prompt(predicate)
        # BEGIN retain the graph prompt and image while changing the VLM provider.
        return self.gemini.chat(prompt, images=[image], system_prompt=meta_prompt)
        # END shared Gemini graph request.

    def get_alignments_string(self, image: Image.Image, nodes: str, scene_graph_string: str) -> str:
        prompt = self.get_alignments_prompt(nodes)
        # BEGIN include graph context and the numbered image in the Gemini request.
        return self.gemini.chat(prompt, images=[image], system_prompt=scene_graph_string)
        # END shared Gemini alignment request.

    # BEGIN use all local SAM3.1 instances, preserving same-class disambiguation.
    def detect_and_segment(self, image: Image.Image, object_classes: List[str]) -> Tuple[List[Any], List[str], List[str]]:
        if not object_classes:
            return [], [], []
        detected_objects = detect(image, list(object_classes))
        box_names = [obj["label"] for obj in detected_objects]
        missing_objects = set(object_classes) - set(box_names)
        if missing_objects:
            print(f"Missing objects that were not detected: {', '.join(sorted(missing_objects))}")
        # Each mask comes from the same prediction as its label and pixel bbox;
        # running a second segmentor would break that one-to-one association.
        masks = [{"segmentation": obj["mask"]} for obj in detected_objects]

        # Count occurrences of each box_name
        box_count = Counter(box_names)

        # Filter out box_names that appear more than once
        multiple_box_names = [name for name, count in box_count.items() if count > 1]
        return masks, box_names, multiple_box_names
    # END local SAM3.1 instance mapping.

    def get_annotated_segmentation(self, image: Image.Image, masks: List[Any]) -> Image.Image:
        segment_img = annotate_masks(
            image, 
            masks=[anno["segmentation"] for anno in masks],
            label_mode="1",
            alpha=0.05,
            draw_mask=True, 
            draw_mark=True, 
            draw_box=True,
            mark_position='top_left'
        )
        return segment_img

    def assign_masks_to_nodes(self, image, scene_graph_data: SceneGraphData, masks: List[Any], box_names: List[str], multiple_box_names: List[str], scene_graph_string):
        # BEGIN reject missing instances before assigning masks or requesting alignment.
        if len(masks) != len(box_names):
            raise ValueError("Masks and detected class names must have the same length")
        requested_counts = Counter(node.object_type for node in scene_graph_data.nodes)
        detected_counts = Counter(box_names)
        for object_type, requested_count in requested_counts.items():
            detected_count = detected_counts[object_type]
            if detected_count < requested_count:
                raise ValueError(
                    f"Insufficient detected instances for {object_type!r}: "
                    f"requested {requested_count}, detected {detected_count}"
                )
        # END instance coverage validation; extra detections remain available for alignment.
        # print("multiple names:", multiple_box_names)
        masks_muiltiple = []
        # BEGIN retain class association for the numbered ambiguous subset.
        ambiguous_labels = []
        # END ambiguous subset class association.
        # Adding masks to the nodes
        for mask, box_name in zip(masks, box_names):
            if box_name not in multiple_box_names:
                print(f"Box name {box_name} not found in multiple_box_names")
                for node in scene_graph_data.nodes:
                    # Check if the box_name matches the object_type of the node
                    if box_name == node.object_type:
                        node.mask = mask["segmentation"]
                        print(f"Added mask to node: {node.name}")
            else:
                masks_muiltiple.append(mask)
                # BEGIN keep each subset mask paired with its detected class.
                ambiguous_labels.append(box_name)
                # END ambiguous subset class association.
        

        unmatched_nodes = [str(node) for node in scene_graph_data.nodes if node.object_type in multiple_box_names]  # Using the __str__ method of each node
        print("Unmatched nodes:", unmatched_nodes)

        annotated_img = self.get_annotated_segmentation(image, masks)

        if len(unmatched_nodes):
            annotated_img_mutiple = self.get_annotated_segmentation(image, masks_muiltiple)

            get_alignments_string = self.get_alignments_string(annotated_img_mutiple, str(unmatched_nodes), scene_graph_string)    
            parse_alignments = self.parse_alignments(get_alignments_string)
            # for 
            print("alignments:", parse_alignments)
            
            # BEGIN validate the complete mapping before assigning any ambiguous mask.
            requested_nodes = {
                node.name: node for node in scene_graph_data.nodes
                if node.object_type in multiple_box_names
            }
            if {name for _, name in parse_alignments} != set(requested_nodes):
                raise ValueError("Alignments must match exactly the requested node names")
            for number, name in parse_alignments:
                if not 1 <= number <= len(masks_muiltiple):
                    raise ValueError(f"Invalid annotated mask number: {number}")
                if ambiguous_labels[number - 1] != requested_nodes[name].object_type:
                    raise ValueError(f"Annotated mask class does not match node {name}")
            # annotate_masks numbers only the ambiguous subset, starting at one.
            for number, name in parse_alignments:
                requested_nodes[name].mask = masks_muiltiple[number - 1]["segmentation"]
            # END validated visual-label to mask mapping.
        return scene_graph_data, annotated_img

if __name__ == "__main__":
    scene_graph_processor = SceneGraph()
    # BEGIN run the example with real Gemini semantics and local SAM3.1 masks.
    file_path = Path(__file__).resolve().parent / "images" / "1.png"
    image = Image.open(file_path).convert("RGB")
    predicate = ['Fruits', 'wood', 'with_mark']
    scene_graph_string = scene_graph_processor.get_scene_graph_string(image, predicate)
    # END real model example.

    nodes, edges, object_classes = scene_graph_processor.parse_scene_graph(scene_graph_string)
    print("Nodes:", nodes)
    print("Edges:", edges)
    print("Object classes:", object_classes)

    # Creating a scene graph data structure
    scene_graph_data = SceneGraphData()
    for node in nodes:
        scene_graph_data.add_node(node)
    for edge in edges:
        scene_graph_data.add_edge(edge)

    masks, box_names, multiple_box_names = scene_graph_processor.detect_and_segment(image, object_classes)

    ## Adding masks to the nodes

    scene_graph_data, annotated_img = scene_graph_processor.assign_masks_to_nodes(image, scene_graph_data, masks, box_names, multiple_box_names, scene_graph_string)
    print(scene_graph_data.nodes)
    # BEGIN support a headless local run without opening an external image viewer.
    output_path = Path.cwd() / "scene_graph_instances.png"
    annotated_img.save(output_path)
    print(f"Annotated masks saved to {output_path}")
    # END headless example output.
