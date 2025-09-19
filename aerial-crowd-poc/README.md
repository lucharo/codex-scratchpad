# AeroCrowd PoC

Minimal yet robust aerial crowd counting proof-of-concept with switchable local and remote backends.

## Quickstart

```bash
# Create environment (Python 3.9+ recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\\Scripts\\activate`

pip install -r requirements.txt
# Optional: `uv pip install -r requirements.txt` if you prefer uv tooling.

python app.py
```

Gradio will print a local URL (typically http://127.0.0.1:7860). Open it to load the UI.

> **Apple Silicon note:** install PyTorch following the [official selector](https://pytorch.org/get-started/locally/) before running `pip install ultralytics`. After PyTorch is installed, rerun `pip install ultralytics`.

## Configuring remote Hugging Face models

Remote Spaces or Inference Endpoints are defined in `models.yaml`:

```yaml
hf_spaces:
  space1:
    url: "https://huggingface.co/spaces/ORG/SPACE/api/predict"
    type: "json_base64"
  density_endpoint:
    url: "https://api.endpoints.huggingface.cloud/.../predict"
    type: "multipart"
    timeout: 60
```

Set `HF_TOKEN` in your environment to authorize private endpoints:

```bash
export HF_TOKEN=hf_your_api_token
```

Click **"Reload models.yaml"** in the UI to pick up changes without restarting.

Supported payload formats:
- `json_base64`: sends `{"image": "<base64>"}` and expects JSON containing `count`, `points`, or `density` fields.
- `multipart`: uploads an `image` file via multipart/form-data.

## Features

- Gradio UI with **Image** and **Video** tabs.
- Multi-select model comparison (local YOLOv8 + configurable HF endpoints).
- Adjustable detection threshold, tiling size/overlap, frame stride.
- Optional stratified sampling helper with 95% CI estimates.
- Per-tile breakdowns, runtime stats, and annotated previews.
- CSV downloads: aggregate results, per-tile counts, per-frame counts.
- CLI mirroring the UI controls.

```
+------------------------------+
|  AeroCrowd Preview           |
|  [annotated image placeholder]
+------------------------------+
```

## CLI examples

```bash
# Image inference with YOLOv8n and a remote density model
python -m aerocrowd --image assets/sample.jpg --models local:yolov8n,hf:space1 --tile-size 1024 --overlap 128

# Video inference every 10th frame with YOLOv8n
python -m aerocrowd --video crowd.mp4 --models local:yolov8n --frame-stride 10
```

Outputs are stored under `runs/<timestamp>/` with annotated imagery, CSV summaries, and per-frame charts.

## Architecture overview

- `app.py`: Gradio interface wiring inputs → model pipelines → downloads.
- `aerocrowd/__init__.py`: model registry, YOLOv8 backend, Hugging Face HTTP backend, shared analysis routines, CLI.
- `aerocrowd/utils.py`: tiling utilities, density map stitching, overlays, stratified CI helpers (with inline tests via `python aerocrowd/utils.py`).
- `models.yaml`: stub configuration for remote endpoints.
- `assets/sample.jpg`: placeholder — replace with a real aerial crowd image for demos.

## Caveats

- YOLOv8 detectors can under-count dense crowds due to occlusion; consider density models for high-density aerial scenes.
- Remote inference latency depends on your Hugging Face Space/Endpoint performance.
- Density maps returned by remote models are naively blended; calibrate scaling for your specific model outputs.
- Stratified sampling estimates rely on heuristic texture stratification and randomly selected tiles; treat results as rough guidance.

Happy counting!
