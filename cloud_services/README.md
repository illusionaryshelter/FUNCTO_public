# Scene graph helpers

Object detection and segmentation run locally through
`local_models.sam31.detect(image, labels)`. Each prediction contains its label,
score, pixel-coordinate `xyxy` box, and full-image boolean mask. The scene graph
keeps every predicted instance; it does not run a second segmentation model on
the returned boxes.

Semantic descriptions and assignment of same-class instances use
`local_models.gemini.Gemini`, which calls the Gemini API through the local `s3e`
integration. This is an API request, not local VLM inference. Both scene graph
implementations use the same Gemini settings and credentials as `main.py`.

`apis/language_model.py` exports this shared `Gemini` interface:

```python
from cloud_services.apis.language_model import Gemini

text = Gemini().chat("Describe the objects in this image.", images=[image])
```

`scene_graph.py` produces a graph and annotated masks. `scene_graph_new.py` also
assigns instance masks to graph nodes, asking Gemini to distinguish repeated
object classes. Both use `graph_format.py` to parse only literal `nodes` and
`edges` assignments with Python's `ast` module; model-generated code is never
executed. A complete Python code fence is accepted, all matching predicates are
retained, and the existing table/chair exclusion and object-class names remain.
Instance alignment validates unique node names and mask numbers, the complete
requested node set, and each selected mask's class before assigning masks.
Insufficient detected instances fail before assignment or another API request;
extra detections remain available for the existing instance-alignment step.
Run examples from the `FUNCTO_public` directory after configuring the local model
paths and Gemini credentials for the main pipeline:

```bash
PYTHONNOUSERSITE=1 python -m cloud_services.scene_graph
PYTHONNOUSERSITE=1 python -m cloud_services.scene_graph_new
```

The examples use `images/1.png`, request real Gemini descriptions, and save
`scene_graph_segmentation.png` or `scene_graph_instances.png` in the working
directory. They consume GPU resources and API requests only when explicitly run.
Importing the modules does not execute either example.

`image_utils.py` and `visualizer.py` remain the common mask/point rendering
utilities. The former OWLv2/SAM HTTP clients, GPT/LLaVA adapters, deployment and
installation scripts, and remote-service example notebooks have been removed.

CPU contract tests use an independent copy of an actual Gemini graph response
and mock model boundaries; they make no GPU or API calls:

```bash
PYTHONNOUSERSITE=1 MPLBACKEND=Agg python -m unittest cloud_services.test_scene_graph_contract -v
```
