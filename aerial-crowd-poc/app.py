"""Gradio application for aerial crowd counting comparison."""
from __future__ import annotations

from typing import List, Optional

import gradio as gr
import pandas as pd

from aerocrowd import ModelRegistry, analyze_image, analyze_video

registry = ModelRegistry()


def _model_choices() -> List[str]:
    return registry.keys()


def _default_selection(choices: List[str]) -> List[str]:
    if "local:yolov8n" in choices:
        return ["local:yolov8n"]
    return choices[:1] if choices else []


def refresh_models() -> gr.CheckboxGroup:
    global registry
    registry = ModelRegistry()
    choices = _model_choices()
    return gr.CheckboxGroup.update(choices=choices, value=_default_selection(choices))


def handle_image(
    image,
    model_keys: List[str],
    threshold: float,
    tile_size: int,
    overlap: int,
    compare: bool,
    enable_sampling: bool,
    samples_per_stratum: int,
) -> tuple:
    if image is None:
        empty_df = pd.DataFrame(columns=["model", "count", "runtime_s", "notes"])
        return None, empty_df, pd.DataFrame(), "Upload an image to start.", None, None, None
    if not model_keys:
        return image, pd.DataFrame(), pd.DataFrame(), "Select at least one model.", None, None, None
    sampling = samples_per_stratum if enable_sampling else 0
    result = analyze_image(
        image,
        model_keys,
        registry=registry,
        threshold=threshold,
        tile_size=tile_size,
        overlap=overlap,
        compare=compare,
        stratified_samples=sampling,
    )
    preview = result.get("preview")
    summary = result.get("summary", pd.DataFrame())
    tile_table = result.get("tile_table") or pd.DataFrame()
    sampling_note = result.get("sampling_note", "")
    files = result.get("files", {})
    results_csv = files.get("results_csv")
    annotated = files.get("annotated_image")
    tile_csv = files.get("tile_counts_csv")
    return (
        preview,
        summary,
        tile_table,
        sampling_note or "",
        str(results_csv) if results_csv else None,
        str(annotated) if annotated else None,
        str(tile_csv) if tile_csv else None,
    )


def handle_video(
    video_path: Optional[str],
    model_keys: List[str],
    threshold: float,
    tile_size: int,
    overlap: int,
    frame_stride: int,
    enable_sampling: bool,
    samples_per_stratum: int,
) -> tuple:
    if not video_path:
        empty_df = pd.DataFrame(columns=["model", "count", "runtime_s", "fps", "notes"])
        return None, empty_df, [], None, None
    if not model_keys:
        return None, pd.DataFrame(), [], None, None
    sampling = samples_per_stratum if enable_sampling else 0
    result = analyze_video(
        video_path,
        model_keys,
        registry=registry,
        threshold=threshold,
        tile_size=tile_size,
        overlap=overlap,
        frame_stride=frame_stride,
        stratified_samples=sampling,
    )
    preview = result.get("preview")
    summary = result.get("summary", pd.DataFrame())
    files = result.get("files", {})
    charts = files.get("charts", {})
    gallery_items = [str(path) for path in charts.values() if path]
    results_csv = files.get("results_csv")
    frame_csv = files.get("combined_frame_csv")
    return (
        preview,
        summary,
        gallery_items,
        str(results_csv) if results_csv else None,
        str(frame_csv) if frame_csv else None,
    )


def build_interface() -> gr.Blocks:
    choices = _model_choices()
    default_models = _default_selection(choices)
    with gr.Blocks(title="AeroCrowd PoC") as demo:
        gr.Markdown(
            """
            # AeroCrowd — Aerial Crowd Counting PoC
            Select one or more models to estimate crowd counts from high-resolution aerial imagery or video.
            Configure thresholds, tiling, and optional stratified sampling to stress-test models on dense scenes.
            """
        )
        with gr.Row():
            model_selector = gr.CheckboxGroup(
                choices=choices,
                value=default_models,
                label="Models",
                info="Mix local YOLO and remote Hugging Face endpoints",
            )
            refresh_btn = gr.Button("Reload models.yaml", variant="secondary")
            refresh_btn.click(refresh_models, outputs=model_selector)
        with gr.Tabs():
            with gr.Tab("Image"):
                image_input = gr.Image(type="numpy", label="Aerial image")
                with gr.Row():
                    threshold = gr.Slider(0.05, 0.8, value=0.25, label="Detection threshold")
                    tile_size = gr.Slider(256, 2048, value=1024, step=64, label="Tile size")
                    overlap = gr.Slider(0, 512, value=128, step=32, label="Tile overlap")
                with gr.Row():
                    compare = gr.Checkbox(value=False, label="Compare selected models")
                    enable_sampling = gr.Checkbox(value=False, label="Enable stratified sampling")
                    samples = gr.Slider(0, 5, value=2, step=1, label="Samples per stratum")
                image_button = gr.Button("Run image analysis", variant="primary")
                preview_out = gr.Image(label="Annotated preview")
                summary_out = gr.Dataframe(label="Model summary")
                tiles_out = gr.Dataframe(label="Per-tile counts (first model)")
                sampling_out = gr.Textbox(label="Sampling estimate", lines=2)
                with gr.Row():
                    results_dl = gr.File(label="results.csv")
                    annotated_dl = gr.File(label="annotated_image.png")
                    tiles_dl = gr.File(label="tile_counts.csv")
                image_button.click(
                    handle_image,
                    inputs=[
                        image_input,
                        model_selector,
                        threshold,
                        tile_size,
                        overlap,
                        compare,
                        enable_sampling,
                        samples,
                    ],
                    outputs=[
                        preview_out,
                        summary_out,
                        tiles_out,
                        sampling_out,
                        results_dl,
                        annotated_dl,
                        tiles_dl,
                    ],
                )
            with gr.Tab("Video"):
                video_input = gr.Video(label="Video", sources=["upload"], type="filepath")
                with gr.Row():
                    frame_stride = gr.Slider(1, 30, value=10, step=1, label="Frame stride")
                    threshold_v = gr.Slider(0.05, 0.8, value=0.25, label="Detection threshold")
                    tile_size_v = gr.Slider(256, 2048, value=1024, step=64, label="Tile size")
                    overlap_v = gr.Slider(0, 512, value=128, step=32, label="Tile overlap")
                with gr.Row():
                    enable_sampling_v = gr.Checkbox(value=False, label="Enable stratified sampling")
                    samples_v = gr.Slider(0, 5, value=2, step=1, label="Samples per stratum")
                video_button = gr.Button("Run video analysis", variant="primary")
                preview_video = gr.Image(label="Preview (sample frame)")
                summary_video = gr.Dataframe(label="Model summary")
                chart_gallery = gr.Gallery(label="Per-model frame counts")
                with gr.Row():
                    video_results_dl = gr.File(label="results.csv")
                    frame_counts_dl = gr.File(label="per_frame_counts.csv")
                video_button.click(
                    handle_video,
                    inputs=[
                        video_input,
                        model_selector,
                        threshold_v,
                        tile_size_v,
                        overlap_v,
                        frame_stride,
                        enable_sampling_v,
                        samples_v,
                    ],
                    outputs=[
                        preview_video,
                        summary_video,
                        chart_gallery,
                        video_results_dl,
                        frame_counts_dl,
                    ],
                )
        gr.Markdown(
            """
            ### Tips
            - Configure remote Hugging Face endpoints via `models.yaml` and reload the interface.
            - Stratified sampling estimates rely on per-tile counts; dense regions may require tighter overlap.
            - CLI usage: `python -m aerocrowd --image path.jpg --models local:yolov8n`
            """
        )
    return demo


def main() -> None:
    demo = build_interface()
    demo.launch()


if __name__ == "__main__":
    main()
