import re
from pathlib import Path
from PIL import Image
# BEGIN local perception and the shared Gemini API integration.
from cloud_services.apis.language_model import Gemini
from cloud_services.image_utils import annotate_masks
from local_models.sam31 import detect
# END local perception and the shared Gemini API integration.
# BEGIN shared literal-only scene graph parser and prompt.
from cloud_services.graph_format import GRAPH_INSTRUCTION, parse_graph_assignments
# END shared literal-only scene graph parser and prompt.


class SceneGraph:
    def __init__(self):
        # BEGIN use the same Gemini client as the main FUNCTO pipeline.
        self.gemini = Gemini()
        # END shared Gemini client.

    @staticmethod
    def get_scene_graph_prompt(predicate):
        prompt = f'''The additional predicate is [{predicate}]'''
        # BEGIN require a complete literal graph with every matching predicate.
        meta_prompt = GRAPH_INSTRUCTION
        # END complete graph response contract.
        return prompt, meta_prompt

    @staticmethod
    def parse_scene_graph(scene_graph):
        # BEGIN parse all legal literal quotes and predicate fields safely.
        nodes, edges = parse_graph_assignments(scene_graph)
        # END shared graph parsing.

        # Filter nodes that include 'table' or 'chair' and collect remaining nodes
        filtered_nodes = [node for node in nodes if not ('table' in node[0].lower() or 'chair' in node[0].lower())]

        # Filter edges, ensuring neither source nor destination is a filtered-out object
        # BEGIN apply the existing exclusion to either endpoint consistently.
        filtered_edges = [edge for edge in edges if not any(
            'table' in element.lower() or 'chair' in element.lower() for element in edge[1:]
        )]
        # END endpoint exclusion.

        # Use a set to collect unique object classes from filtered nodes, removing numbers
        object_classes_set = set(re.sub(r'\d+', '', obj) for obj, _ in filtered_nodes)

        # Convert set to list to finalize the object classes
        object_classes = list(object_classes_set)

        return filtered_nodes, filtered_edges, object_classes

    def get_scene_graph_string(self, image, predicate):
        prompt, meta_prompt = self.get_scene_graph_prompt(predicate)
        # BEGIN retain the graph prompt and image while changing the VLM provider.
        return self.gemini.chat(prompt, images=[image], system_prompt=meta_prompt)
        # END shared Gemini request.

    def detect_and_segment(self, image, object_classes):
        # BEGIN keep every SAM3.1 instance and its associated label/mask together.
        # SAM3.1 already supplies masks, so a second box-prompted segmentation
        # pass would duplicate computation and could select a different instance.
        if not object_classes:
            return [], []
        detected_objects = detect(image, list(object_classes))
        box_names = [obj["label"] for obj in detected_objects]
        missing_objects = set(object_classes) - set(box_names)
        if missing_objects:
            print(f"Missing objects that were not detected: {', '.join(sorted(missing_objects))}")
        masks = [{"segmentation": obj["mask"]} for obj in detected_objects]
        return masks, box_names
        # END local SAM3.1 instance mapping.
    
    def get_annotated_segmentation(self, image, masks):
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

if __name__ == "__main__":
    scene_graph_processor = SceneGraph()
    # BEGIN run the example with real Gemini semantics and local SAM3.1 masks.
    file_path = Path(__file__).resolve().parent / "images" / "1.png"
    image = Image.open(file_path).convert("RGB")
    predicate = "Fruits"
    scene_graph_string = scene_graph_processor.get_scene_graph_string(image, predicate)
    # END real model example.
    nodes, edges, object_classes = scene_graph_processor.parse_scene_graph(scene_graph_string)

    print("Nodes:", nodes)
    print("Edges:", edges)
    print("Object classes:", object_classes)
    masks, box_names = scene_graph_processor.detect_and_segment(image, object_classes)
    print("Box names:", box_names)

    annotated_img = scene_graph_processor.get_annotated_segmentation(image, masks)
    # BEGIN support a headless local run without opening an external image viewer.
    output_path = Path.cwd() / "scene_graph_segmentation.png"
    annotated_img.save(output_path)
    print(f"Annotated masks saved to {output_path}")
    # END headless example output.
