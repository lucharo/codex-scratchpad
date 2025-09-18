"""Core models and pipeline utilities for AeroCrowd."""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import requests
import yaml

from . import utils


ModelKind = Literal["detector", "density", "remote"]


@dataclass
class BaseCrowdModel:
    name: str
    kind: ModelKind

    def predict_image(self, image: np.ndarray, **kwargs) -> Dict:
        raise NotImplementedError

    def predict_video(self, path: str, frame_stride: int = 10, **kwargs) -> Dict:
        start = time.time()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
        frame_index = 0
        frame_counts: List[Tuple[int, int]] = []
        sample_preview: Optional[Dict] = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % frame_stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.predict_image(rgb, **kwargs)
                count = int(result.get("count", 0))
                frame_counts.append((frame_index, count))
                if sample_preview is None:
                    sample_preview = {
                        "frame_index": frame_index,
                        "image": rgb,
                        "result": result,
                    }
            frame_index += 1
        cap.release()
        runtime = time.time() - start
        total_count = int(sum(c for _, c in frame_counts))
        meta = {
            "runtime_s": runtime,
            "frame_counts": frame_counts,
            "fps": (len(frame_counts) / runtime) if runtime > 0 else 0.0,
            "notes": f"Processed {len(frame_counts)} frames (stride={frame_stride})",
        }
        if sample_preview is not None:
            meta["sample_preview"] = sample_preview
        return {
            "count": total_count,
            "points": None,
            "bboxes": None,
            "density": None,
            "meta": meta,
        }


class LocalYoloCrowdModel(BaseCrowdModel):
    def __init__(
        self,
        name: str = "local:yolov8n",
        model_path: str = "yolov8n.pt",
        device: str = "cpu",
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> None:
        super().__init__(name=name, kind="detector")
        from ultralytics import YOLO  # Lazy import

        self.model_path = model_path
        self.device = device
        self.default_conf = conf
        self.iou = iou
        self._model = YOLO(model_path)

    def _run_detector(self, image_bgr: np.ndarray, conf: float) -> Tuple[List[Tuple[float, float, float, float, float]], List[Tuple[float, float]]]:
        results = self._model.predict(
            image_bgr,
            conf=conf,
            iou=self.iou,
            classes=[0],
            device=self.device,
            verbose=False,
        )
        bboxes: List[Tuple[float, float, float, float, float]] = []
        points: List[Tuple[float, float]] = []
        for res in results:
            if res.boxes is None:
                continue
            for box in res.boxes:
                cls = int(box.cls[0]) if box.cls is not None else 0
                if cls != 0:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf[0]) if box.conf is not None else 0.0
                bboxes.append((x1, y1, x2, y2, score))
                points.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
        return bboxes, points

    def predict_image(
        self,
        image: np.ndarray,
        tile_size: int = 1024,
        overlap: int = 128,
        conf: Optional[float] = None,
        stratified_samples: int = 0,
        rng: Optional[np.random.Generator] = None,
        **_: Dict,
    ) -> Dict:
        start = time.time()
        rgb = utils.ensure_rgb(image)
        conf = conf if conf is not None else self.default_conf
    tiles = utils.tile_image(rgb, tile_size=tile_size, overlap=overlap)
        all_bboxes: List[Tuple[float, float, float, float, float]] = []
        all_points: List[Tuple[float, float]] = []
        tile_counts: List[int] = []
        tile_origins: List[Tuple[int, int, int, int]] = []
        textures: List[float] = []
        for tile in tiles:
            tile_bgr = cv2.cvtColor(tile.image, cv2.COLOR_RGB2BGR)
            bboxes, points = self._run_detector(tile_bgr, conf=conf)
            tile_count = len(bboxes)
            tile_counts.append(tile_count)
            h, w = tile.image.shape[:2]
            tile_origins.append((tile.origin[0], tile.origin[1], w, h))
            gray = cv2.cvtColor(tile.image, cv2.COLOR_RGB2GRAY)
            textures.append(float(gray.std()))
            if bboxes:
                all_bboxes.extend(utils.offset_bboxes(bboxes, tile.origin[0], tile.origin[1]))
            if points:
                all_points.extend([(x + tile.origin[0], y + tile.origin[1]) for x, y in points])
        runtime = time.time() - start
        meta = {
            "runtime_s": runtime,
            "notes": f"conf={conf}, tiles={len(tile_counts)}",
            "tile_counts": tile_counts,
            "tile_origins": tile_origins,
            "textures": textures,
        }
        if stratified_samples > 0 and tile_counts:
            labels = utils.classify_texture_strata(textures)
            sample_dict, strata_sizes = utils.random_sample_by_strata(labels, tile_counts, stratified_samples)
            estimate, lower, upper = utils.compute_stratified_ci(sample_dict, strata_sizes)
            total_tiles = len(tile_counts)
            meta["sampling"] = {
                "estimate_per_tile": estimate,
                "ci": (lower, upper),
                "total_tiles": total_tiles,
                "estimate_total": estimate * total_tiles,
            }
        return {
            "count": int(sum(tile_counts)),
            "points": all_points,
            "bboxes": all_bboxes,
            "density": None,
            "meta": meta,
        }


class HuggingFaceCrowdModel(BaseCrowdModel):
    def __init__(self, name: str, config: Dict) -> None:
        super().__init__(name=name, kind="remote")
        self.config = config
        self.url = config.get("url")
        if not self.url:
            raise ValueError(f"Missing URL for model {name}")
        self.req_type = config.get("type", "json_base64")
        self.timeout = float(config.get("timeout", 30))
        self.headers = {}
        token = os.environ.get("HF_TOKEN")
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    def _encode_image(self, image: np.ndarray) -> Tuple[Dict, Dict, Optional[bytes]]:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not success:
            raise RuntimeError("Failed to encode image for remote call")
        data = buffer.tobytes()
        if self.req_type == "json_base64":
            payload = {"image": base64.b64encode(data).decode("utf-8")}
            return payload, {}, None
        elif self.req_type == "multipart":
            files = {"image": ("frame.jpg", data, "image/jpeg")}
            return {}, files, data
        else:
            raise ValueError(f"Unknown request type: {self.req_type}")

    def _parse_response(self, response: requests.Response) -> Dict:
        response.raise_for_status()
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON response: {exc}")
        payload = data
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], dict):
            payload = {**data, **data["data"]}
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            # Merge list of dicts
            merged: Dict = {}
            for item in data["data"]:
                if isinstance(item, dict):
                    merged.update(item)
            payload = {**data, **merged}
        count = payload.get("count")
        if isinstance(count, (float, int)):
            count = int(round(float(count)))
        else:
            count = None
        points = payload.get("points")
        if isinstance(points, list):
            parsed_points = []
            for pt in points:
                if isinstance(pt, dict) and {"x", "y"} <= set(pt.keys()):
                    parsed_points.append((float(pt["x"]), float(pt["y"])))
                elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    parsed_points.append((float(pt[0]), float(pt[1])))
            points = parsed_points
        else:
            points = None
        density = payload.get("density")
        if density is not None:
            density = np.array(density, dtype=np.float32)
            if density.ndim == 3 and density.shape[0] in (1, 3):
                density = np.mean(density, axis=0)
            if count is None:
                count = int(round(float(density.sum())))
        if count is None and points is not None:
            count = len(points)
        if count is None:
            raise RuntimeError("Remote response did not contain count/points/density")
        return {
            "count": int(count),
            "points": points,
            "density": density,
            "raw": payload,
        }

    def predict_image(
        self,
        image: np.ndarray,
        tile_size: int = 1024,
        overlap: int = 128,
        stratified_samples: int = 0,
        rng: Optional[np.random.Generator] = None,
        **_: Dict,
    ) -> Dict:
        start = time.time()
        rgb = utils.ensure_rgb(image)
        tiles = utils.tile_image(rgb, tile_size=tile_size, overlap=overlap)
        tile_counts: List[int] = []
        tile_origins: List[Tuple[int, int, int, int]] = []
        textures: List[float] = []
        density_tiles: List[Tuple[np.ndarray, Tuple[int, int]]] = []
        all_points: List[Tuple[float, float]] = []
        notes: List[str] = []
        for tile in tiles:
            try:
                payload, files, _ = self._encode_image(tile.image)
                headers = dict(self.headers)
                if self.req_type == "json_base64":
                    response = requests.post(
                        self.url,
                        headers={**headers, "Content-Type": "application/json"},
                        json=payload,
                        timeout=self.timeout,
                    )
                else:
                    response = requests.post(
                        self.url,
                        headers=headers,
                        files=files,
                        timeout=self.timeout,
                    )
                parsed = self._parse_response(response)
                count = int(parsed["count"])
                tile_counts.append(count)
                h, w = tile.image.shape[:2]
                tile_origins.append((tile.origin[0], tile.origin[1], w, h))
                gray = cv2.cvtColor(tile.image, cv2.COLOR_RGB2GRAY)
                textures.append(float(gray.std()))
                if parsed.get("density") is not None:
                    density_tiles.append((np.asarray(parsed["density"], dtype=np.float32), tile.origin))
                if parsed.get("points"):
                    all_points.extend(
                        [(x + tile.origin[0], y + tile.origin[1]) for x, y in parsed["points"]]
                    )
            except Exception as exc:  # noqa: BLE001
                notes.append(f"tile@{tile.origin} failed: {exc}")
        runtime = time.time() - start
        total_count = int(sum(tile_counts))
        density = None
        if density_tiles:
            density = utils.stitch_density_map(density_tiles, rgb.shape[:2])
        meta = {
            "runtime_s": runtime,
            "notes": "; ".join(notes) if notes else f"tiles={len(tile_counts)}",
            "tile_counts": tile_counts,
            "tile_origins": tile_origins,
            "textures": textures,
        }
        if stratified_samples > 0 and tile_counts:
            labels = utils.classify_texture_strata(textures)
            sample_dict, strata_sizes = utils.random_sample_by_strata(labels, tile_counts, stratified_samples)
            estimate, lower, upper = utils.compute_stratified_ci(sample_dict, strata_sizes)
            total_tiles = len(tile_counts)
            meta["sampling"] = {
                "estimate_per_tile": estimate,
                "ci": (lower, upper),
                "total_tiles": total_tiles,
                "estimate_total": estimate * total_tiles,
            }
        return {
            "count": total_count,
            "points": all_points if all_points else None,
            "bboxes": None,
            "density": density,
            "meta": meta,
        }


def _default_local_factories() -> Dict[str, Callable[[], BaseCrowdModel]]:
    return {
        "local:yolov8n": lambda: LocalYoloCrowdModel(name="local:yolov8n", model_path="yolov8n.pt"),
    }


class ModelRegistry:
    def __init__(self, config_path: str = "models.yaml") -> None:
        self.config_path = config_path
        self.factories: Dict[str, Callable[[], BaseCrowdModel]] = _default_local_factories()
        self.instances: Dict[str, BaseCrowdModel] = {}
        self._load_remote_factories()

    def _load_remote_factories(self) -> None:
        if not os.path.exists(self.config_path):
            return
        with open(self.config_path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
        for name, cfg in (config.get("hf_spaces") or {}).items():
            key = f"hf:{name}"
            cfg_copy = dict(cfg)
            self.factories[key] = lambda cfg=cfg_copy, key=key: HuggingFaceCrowdModel(name=key, config=cfg)

    def keys(self) -> List[str]:
        return sorted(self.factories.keys())

    def get(self, key: str) -> BaseCrowdModel:
        if key not in self.factories:
            raise KeyError(f"Unknown model key: {key}")
        if key not in self.instances:
            self.instances[key] = self.factories[key]()
        return self.instances[key]


def _color_for_index(index: int) -> Tuple[int, int, int]:
    palette = [
        (46, 204, 113),
        (52, 152, 219),
        (231, 76, 60),
        (155, 89, 182),
        (241, 196, 15),
        (26, 188, 156),
    ]
    return palette[index % len(palette)]


def analyze_image(
    image: np.ndarray,
    model_keys: Sequence[str],
    registry: Optional[ModelRegistry] = None,
    threshold: float = 0.25,
    tile_size: int = 1024,
    overlap: int = 128,
    compare: bool = False,
    stratified_samples: int = 0,
) -> Dict:
    registry = registry or ModelRegistry()
    base = utils.ensure_rgb(image)
    overlays: List[Tuple[str, np.ndarray]] = []
    rows: List[Dict] = []
    sampling_rows: List[str] = []
    tile_table = None
    first_tile_export = None
    merged_frame_counts: Optional[pd.DataFrame] = None
    for idx, key in enumerate(model_keys):
        model = registry.get(key)
        try:
            result = model.predict_image(
                base,
                conf=threshold,
                tile_size=tile_size,
                overlap=overlap,
                stratified_samples=stratified_samples,
            )
        except Exception as exc:  # noqa: BLE001
            rows.append({
                "model": model.name,
                "count": 0,
                "runtime_s": 0.0,
                "notes": f"error: {exc}",
            })
            continue
        meta = result.get("meta", {})
        notes = meta.get("notes", "")
        if meta.get("sampling"):
            sampling = meta["sampling"]
            sampling_rows.append(
                f"{model.name}: est={sampling['estimate_total']:.1f} (per tile {sampling['estimate_per_tile']:.2f})"
                f" CI[{sampling['ci'][0]:.2f}, {sampling['ci'][1]:.2f}]"
            )
        rows.append(
            {
                "model": model.name,
                "count": result.get("count", 0),
                "runtime_s": round(meta.get("runtime_s", 0.0), 3),
                "notes": notes,
            }
        )
        overlay_img = base.copy()
        if result.get("density") is not None:
            overlay_img = utils.overlay_density_heatmap(overlay_img, result["density"])
        color = _color_for_index(idx)
        if result.get("bboxes"):
            overlay_img = utils.draw_bboxes(overlay_img, result["bboxes"], color=color, label=model.name)
        if result.get("points"):
            overlay_img = utils.draw_points(overlay_img, result["points"], color=color)
        overlays.append((model.name, overlay_img))
        if tile_table is None and meta.get("tile_counts"):
            tile_records = []
            for i, (count, origin) in enumerate(zip(meta["tile_counts"], meta["tile_origins"])):
                x, y, w, h = origin
                tile_records.append({
                    "tile": i,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "count": count,
                })
            tile_table = pd.DataFrame(tile_records)
            first_tile_export = tile_table
    preview = utils.annotate_preview(base, overlays, compare=compare)
    summary_df = pd.DataFrame(rows)
    sampling_note = "\n".join(sampling_rows) if sampling_rows else ""
    files: Dict[str, Path] = {}
    run_dir = Path("runs")
    run_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = run_dir / f"image-{timestamp}"
    out_dir.mkdir(exist_ok=True)
    preview_bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
    preview_path = out_dir / "annotated_image.png"
    cv2.imwrite(str(preview_path), preview_bgr)
    files["annotated_image"] = preview_path
    results_path = out_dir / "results.csv"
    summary_df.to_csv(results_path, index=False)
    files["results_csv"] = results_path
    if first_tile_export is not None:
        tiles_path = out_dir / "tile_counts.csv"
        first_tile_export.to_csv(tiles_path, index=False)
        files["tile_counts_csv"] = tiles_path
    if sampling_note:
        note_path = out_dir / "sampling.txt"
        note_path.write_text(sampling_note, encoding="utf-8")
        files["sampling_txt"] = note_path
    return {
        "preview": preview,
        "summary": summary_df,
        "tile_table": tile_table,
        "files": files,
        "sampling_note": sampling_note,
    }


def analyze_video(
    path: str,
    model_keys: Sequence[str],
    registry: Optional[ModelRegistry] = None,
    threshold: float = 0.25,
    tile_size: int = 1024,
    overlap: int = 128,
    frame_stride: int = 10,
    stratified_samples: int = 0,
) -> Dict:
    registry = registry or ModelRegistry()
    rows: List[Dict] = []
    overlays: List[Tuple[str, np.ndarray]] = []
    per_frame_exports: Dict[str, pd.DataFrame] = {}
    charts: Dict[str, Path] = {}
    run_dir = Path("runs")
    run_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = run_dir / f"video-{timestamp}"
    out_dir.mkdir(exist_ok=True)

    for idx, key in enumerate(model_keys):
        model = registry.get(key)
        try:
            result = model.predict_video(
                path,
                frame_stride=frame_stride,
                conf=threshold,
                tile_size=tile_size,
                overlap=overlap,
                stratified_samples=stratified_samples,
            )
        except Exception as exc:  # noqa: BLE001
            rows.append({
                "model": model.name,
                "count": 0,
                "runtime_s": 0.0,
                "fps": 0.0,
                "notes": f"error: {exc}",
            })
            continue
        meta = result.get("meta", {})
        frame_counts = meta.get("frame_counts", [])
        df = pd.DataFrame(frame_counts, columns=["frame", "count"])
        per_frame_exports[model.name] = df
        chart_path = out_dir / f"{model.name.replace(':', '_')}_counts.png"
        if not df.empty:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(4, 2.5))
            plt.plot(df["frame"], df["count"], marker="o")
            plt.xlabel("Frame")
            plt.ylabel("Count")
            plt.title(model.name)
            plt.tight_layout()
            plt.savefig(chart_path)
            plt.close()
            charts[model.name] = chart_path
        notes = meta.get("notes", "")
        rows.append(
            {
                "model": model.name,
                "count": result.get("count", 0),
                "runtime_s": round(meta.get("runtime_s", 0.0), 3),
                "fps": round(meta.get("fps", 0.0), 2),
                "notes": notes,
            }
        )
        sample = meta.get("sample_preview")
        if sample and sample.get("result"):
            frame_rgb = sample["image"]
            overlay_img = frame_rgb.copy()
            color = _color_for_index(idx)
            sample_result = sample["result"]
            if sample_result.get("density") is not None:
                overlay_img = utils.overlay_density_heatmap(overlay_img, sample_result["density"])
            if sample_result.get("bboxes"):
                overlay_img = utils.draw_bboxes(overlay_img, sample_result["bboxes"], color=color, label=model.name)
            if sample_result.get("points"):
                overlay_img = utils.draw_points(overlay_img, sample_result["points"], color=color)
            overlays.append((model.name, overlay_img))
    preview = overlays[0][1] if overlays else None
    if len(overlays) > 1:
        preview = utils.annotate_preview(overlays[0][1], overlays, compare=True)
    summary_df = pd.DataFrame(rows)
    preview_path = None
    if preview is not None:
        preview_path = out_dir / "preview.png"
        cv2.imwrite(str(preview_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    results_path = out_dir / "results.csv"
    summary_df.to_csv(results_path, index=False)
    frame_csv_paths: Dict[str, Path] = {}
    for name, df in per_frame_exports.items():
        path_csv = out_dir / f"{name.replace(':', '_')}_frames.csv"
        df.to_csv(path_csv, index=False)
        frame_csv_paths[name] = path_csv
        renamed = df.rename(columns={"count": name})
        if merged_frame_counts is None:
            merged_frame_counts = renamed
        else:
            merged_frame_counts = pd.merge(merged_frame_counts, renamed, on="frame", how="outer")
    combined_frame_path = None
    if merged_frame_counts is not None:
        merged_frame_counts = merged_frame_counts.sort_values("frame").fillna(0)
        combined_frame_path = out_dir / "per_frame_counts.csv"
        merged_frame_counts.to_csv(combined_frame_path, index=False)
    return {
        "preview": preview,
        "summary": summary_df,
        "files": {
            "results_csv": results_path,
            "frame_csv": frame_csv_paths,
            "charts": charts,
            "preview": preview_path,
            "combined_frame_csv": combined_frame_path,
        },
    }


def run_cli(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Aerial crowd counting CLI")
    parser.add_argument("--image", type=str, help="Path to an image")
    parser.add_argument("--video", type=str, help="Path to a video")
    parser.add_argument("--models", type=str, default="local:yolov8n", help="Comma-separated model keys")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--stratified-samples", type=int, default=0)
    args = parser.parse_args(argv)
    registry = ModelRegistry()
    model_keys = [key.strip() for key in args.models.split(",") if key.strip()]
    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            raise SystemExit(f"Failed to read image: {args.image}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = analyze_image(
            rgb,
            model_keys,
            registry=registry,
            threshold=args.threshold,
            tile_size=args.tile_size,
            overlap=args.overlap,
            compare=True,
            stratified_samples=args.stratified_samples,
        )
        print(result["summary"].to_string(index=False))
        if result.get("sampling_note"):
            print("Sampling:\n" + result["sampling_note"])
        for name, path in result["files"].items():
            print(f"Saved {name}: {path}")
    elif args.video:
        result = analyze_video(
            args.video,
            model_keys,
            registry=registry,
            threshold=args.threshold,
            tile_size=args.tile_size,
            overlap=args.overlap,
            frame_stride=args.frame_stride,
            stratified_samples=args.stratified_samples,
        )
        print(result["summary"].to_string(index=False))
        for key, path in result["files"].items():
            print(f"Saved {key}: {path}")
    else:
        parser.error("Provide either --image or --video")


__all__ = [
    "BaseCrowdModel",
    "LocalYoloCrowdModel",
    "HuggingFaceCrowdModel",
    "ModelRegistry",
    "analyze_image",
    "analyze_video",
    "run_cli",
]
