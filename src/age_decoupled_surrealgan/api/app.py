from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nibabel import Nifti1Image, load as load_nifti

from ..analysis_artifacts import build_run_analysis_artifacts
from ..config import ProjectConfig
from ..inference import generate_single_row, infer_subject_defaults
from ..reporting import format_duration
from ..utils import ensure_dir

try:
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover - runtime dependency
    BaseModel = object  # type: ignore[assignment]
    Field = lambda default=None, **_: default  # type: ignore[assignment]


class InferRequest(BaseModel):
    run_name: str | None = None
    split_name: str = Field(default="application")
    row_index: int | float | str | None = None
    feature_values: dict[str, float] | None = None
    age_years: float | None = None
    process_latents: list[float] | None = None


class SubjectDefaultsRequest(BaseModel):
    run_name: str
    split_name: str = Field(default="application")
    row_index: int | float | str


def create_app(config: ProjectConfig):
    try:
        from fastapi import Body, FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("FastAPI and Pydantic must be installed to run the API.") from exc

    app = FastAPI(title="Age-Decoupled SurrealGAN API", version="0.1.0")
    debug_mode = os.getenv("AGE_DECOUPLED_SURREALGAN_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[config.app.allow_origin],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    runs_dir = Path(config.paths.runs_dir)
    processed_dir = Path(config.paths.processed_dir)
    overlay_dir = ensure_dir(runs_dir / "_web_cache")
    segmentation_img = load_nifti(config.paths.atlas_segmentation)
    segmentation_data = np.asanyarray(segmentation_img.dataobj)

    def run_dir_for_name(run_name: str) -> Path:
        path = runs_dir / run_name
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Run not found: {run_name}")
        return path

    def run_summary(run_name: str) -> dict[str, Any]:
        path = run_dir_for_name(run_name) / "run_summary.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="run_summary.json not found")
        return json.loads(path.read_text())

    def analysis_manifest(run_name: str) -> dict[str, Any]:
        run_dir = run_dir_for_name(run_name)
        path = run_dir / "analysis" / "manifest.json"
        if not path.exists():
            return build_run_analysis_artifacts(run_dir, config, force=False, device=config.training.device)
        return json.loads(path.read_text(encoding="utf-8"))

    def roi_metadata() -> pd.DataFrame:
        path = processed_dir / "roi_metadata.csv"
        if not path.exists():
            raise HTTPException(status_code=404, detail="ROI metadata not found. Run prepare-data first.")
        return pd.read_csv(path)

    def split_frame(split_name: str) -> pd.DataFrame:
        path = processed_dir / f"{split_name}.csv"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Split not found: {split_name}")
        frame = pd.read_csv(path, low_memory=False)
        row_index = pd.Series(range(len(frame)), name="row_index")
        return pd.concat([row_index, frame.reset_index(drop=True)], axis=1, copy=False)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/runs")
    def list_runs() -> list[dict[str, Any]]:
        items = []
        for path in sorted(runs_dir.glob("*"), reverse=True):
            summary_path = path / "run_summary.json"
            if summary_path.exists():
                payload = json.loads(summary_path.read_text())
                items.append(
                    {
                        "run_name": path.name,
                        "selected_checkpoint": payload.get("selected_checkpoint"),
                        "selected_repetition": payload.get("selected_repetition"),
                    }
                )
        return items

    @app.get("/runs/{run_name}/metadata")
    def get_run_metadata(run_name: str) -> dict[str, Any]:
        payload = run_summary(run_name)
        checkpoint = Path(payload["selected_checkpoint"])
        checkpoint_payload = json.loads(json.dumps(torch_load_metadata(checkpoint)))
        return {
            "run_name": run_name,
            "summary": payload,
            "n_processes": checkpoint_payload["n_processes"],
            "feature_columns": checkpoint_payload["feature_columns"],
            "analysis": payload.get("analysis"),
        }

    @app.get("/runs/{run_name}/splits/{split_name}")
    def get_split_subjects(run_name: str, split_name: str) -> list[dict[str, Any]]:
        _ = run_summary(run_name)
        frame = split_frame(split_name)
        columns = ["row_index", "subject_id", "study", "age", "sex", "diagnosis_raw", "diagnosis_group", "cohort_bucket"]
        return frame[columns].to_dict(orient="records")

    def parse_row_index(value: int | float | str | None) -> int:
        if value is None or value == "":
            raise HTTPException(status_code=400, detail="row_index is required.")
        try:
            row_index = int(float(value))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"Invalid row_index: {value}") from None
        return row_index

    def overlay_url_for_percent_change(percent_change: list[float], roi_df: pd.DataFrame) -> str:
        overlay_data = np.zeros(segmentation_data.shape, dtype=np.float32)
        lookup = roi_df.set_index("roi_id")["percent_change"].to_dict()
        for roi_id, value in lookup.items():
            if pd.isna(roi_id):
                continue
            overlay_data[segmentation_data == int(roi_id)] = float(value)
        overlay_path = overlay_dir / f"overlay_{uuid.uuid4().hex}.nii.gz"
        overlay_img = Nifti1Image(overlay_data, segmentation_img.affine, segmentation_img.header)
        overlay_img.to_filename(str(overlay_path))
        return f"/overlays/{overlay_path.name}"

    @app.post("/defaults")
    def subject_defaults(request: SubjectDefaultsRequest = Body(...)) -> dict[str, Any]:
        request_timer_start = time.perf_counter()
        payload = run_summary(request.run_name)
        checkpoint_path = Path(payload["selected_checkpoint"])
        checkpoint_meta = torch_load_metadata(checkpoint_path)
        row_index = parse_row_index(request.row_index)
        frame = split_frame(request.split_name)
        if row_index < 0 or row_index >= len(frame):
            raise HTTPException(status_code=400, detail="row_index out of bounds")
        selected = frame.iloc[row_index]
        metadata = selected[
            ["subject_id", "study", "age", "sex", "diagnosis_raw", "diagnosis_group", "cohort_bucket"]
        ].to_dict()
        try:
            defaults = infer_subject_defaults(
                checkpoint_path=checkpoint_path,
                row=selected,
                feature_columns=checkpoint_meta["feature_columns"],
                device=config.training.device,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        default_age_years = float(selected["age"]) if pd.notna(selected["age"]) else 20.0
        if debug_mode:
            print(
                "[api/defaults]",
                {
                    "run_name": request.run_name,
                    "split_name": request.split_name,
                    "row_index": row_index,
                    "subject_id": metadata.get("subject_id"),
                    "age_years": default_age_years,
                    "n_processes": defaults["n_processes"],
                    "process_head": defaults["process_latents"][: min(5, len(defaults["process_latents"]))],
                    "duration": format_duration(time.perf_counter() - request_timer_start),
                },
            )
        return {
            "metadata": metadata,
            "defaults": {
                "n_processes": defaults["n_processes"],
                "age_years": float(min(max(default_age_years, 20.0), 100.0)),
                "age_latent": defaults["age_latent"],
                "process_latents": defaults["process_latents"],
            },
        }

    @app.post("/infer")
    def infer(request: InferRequest = Body(...)) -> dict[str, Any]:
        request_timer_start = time.perf_counter()
        if request.run_name is None or not str(request.run_name).strip():
            raise HTTPException(status_code=400, detail="run_name is required.")
        payload = run_summary(request.run_name)
        checkpoint_path = Path(payload["selected_checkpoint"])
        checkpoint_meta = torch_load_metadata(checkpoint_path)
        feature_columns = checkpoint_meta["feature_columns"]
        reference_template = pd.read_csv(processed_dir / "reference_template.csv").set_index("feature_name")["value"]

        if request.feature_values is not None:
            row = pd.Series(request.feature_values)
            for column in feature_columns:
                if column not in row.index:
                    raise HTTPException(status_code=400, detail=f"Missing feature: {column}")
            row = row.reindex(feature_columns)
            metadata = {
                "subject_id": "manual",
                "study": "manual",
                "age": request.age_years,
                "sex": "U",
                "diagnosis_raw": "manual",
                "diagnosis_group": "manual",
                "cohort_bucket": "manual",
            }
        else:
            row_index = parse_row_index(request.row_index)
            frame = split_frame(request.split_name)
            if row_index < 0 or row_index >= len(frame):
                raise HTTPException(status_code=400, detail="row_index out of bounds")
            selected = frame.iloc[row_index]
            row = selected
            metadata = selected[
                ["subject_id", "study", "age", "sex", "diagnosis_raw", "diagnosis_group", "cohort_bucket"]
            ].to_dict()

        inference_timer_start = time.perf_counter()
        try:
            inference = generate_single_row(
                checkpoint_path=checkpoint_path,
                row=row,
                feature_columns=feature_columns,
                age_years=request.age_years,
                process_latents=request.process_latents,
                device=config.training.device,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        inference_seconds = time.perf_counter() - inference_timer_start
        roi_df = roi_metadata().copy()
        roi_df["baseline_value"] = inference["baseline_target"]
        roi_df["predicted_value"] = inference["synthetic_target"]
        roi_df["delta"] = inference["synthetic_delta"]
        roi_df["percent_change"] = inference["percent_change"]
        roi_df["age_delta"] = inference["age_delta"]
        for idx in range(inference["n_processes"]):
            roi_df[f"process_{idx + 1}_delta"] = [values[idx] for values in zip(*inference["process_deltas"])]
        roi_df["abs_percent_change"] = roi_df["percent_change"].abs()
        top_changes = roi_df.sort_values("abs_percent_change", ascending=False).head(25)
        overlay_timer_start = time.perf_counter()
        overlay_url = overlay_url_for_percent_change(inference["percent_change"], roi_df)
        overlay_seconds = time.perf_counter() - overlay_timer_start
        if debug_mode:
            print(
                "[api/infer]",
                {
                    "run_name": request.run_name,
                    "split_name": request.split_name,
                    "row_index": request.row_index,
                    "subject_id": metadata.get("subject_id"),
                    "requested_age_years": inference["requested_age_years"],
                    "age_latent": inference["age_latent"],
                    "default_age_latent": inference.get("default_age_latent"),
                    "process_head": inference["process_latents"][: min(5, len(inference["process_latents"]))],
                    "default_process_head": inference.get("default_process_latents", [])[
                        : min(5, len(inference.get("default_process_latents", [])))
                    ],
                    "delta_abs_max": float(roi_df["delta"].abs().max()),
                    "delta_abs_mean": float(roi_df["delta"].abs().mean()),
                    "pct_abs_max": float(roi_df["percent_change"].abs().max()),
                    "pct_abs_mean": float(roi_df["percent_change"].abs().mean()),
                    "inference_duration": format_duration(inference_seconds),
                    "overlay_duration": format_duration(overlay_seconds),
                    "total_duration": format_duration(time.perf_counter() - request_timer_start),
                    "generation_debug": inference.get("debug", {}),
                },
            )
        return {
            "metadata": metadata,
            "inference": inference,
            "overlay_image_url": overlay_url,
            "roi_table": roi_df.to_dict(orient="records"),
            "top_changes": top_changes.to_dict(orient="records"),
            "timing": {
                "inference_seconds": inference_seconds,
                "overlay_seconds": overlay_seconds,
                "total_seconds": time.perf_counter() - request_timer_start,
            },
            "debug": inference.get("debug") if debug_mode else None,
        }

    @app.get("/runs/{run_name}/population-patterns")
    def get_population_patterns(run_name: str) -> dict[str, Any]:
        manifest = analysis_manifest(run_name)
        population = manifest.get("population_patterns", {})
        patterns = []
        for item in population.get("patterns", []):
            row = dict(item)
            overlay_filename = row.pop("overlay_filename", None)
            if overlay_filename:
                row["overlay_image_url"] = f"/runs/{run_name}/analysis/population-patterns/overlays/{overlay_filename}"
            patterns.append(row)
        return {
            **population,
            "patterns": patterns,
        }

    @app.get("/runs/{run_name}/population-patterns/{pattern_key}")
    def get_population_pattern(run_name: str, pattern_key: str) -> dict[str, Any]:
        _ = analysis_manifest(run_name)
        pattern_path = run_dir_for_name(run_name) / "analysis" / "population_patterns" / f"{pattern_key}.json"
        if not pattern_path.exists():
            raise HTTPException(status_code=404, detail=f"Population pattern not found: {pattern_key}")
        payload = json.loads(pattern_path.read_text(encoding="utf-8"))
        overlay_filename = payload.pop("overlay_filename", None)
        if overlay_filename:
            payload["overlay_image_url"] = f"/runs/{run_name}/analysis/population-patterns/overlays/{overlay_filename}"
        return payload

    @app.get("/runs/{run_name}/analysis/population-patterns/overlays/{overlay_name}")
    def population_overlay(run_name: str, overlay_name: str):
        path = run_dir_for_name(run_name) / "analysis" / "population_patterns" / overlay_name
        if not path.exists():
            raise HTTPException(status_code=404, detail="Population overlay not found.")
        return FileResponse(path)

    @app.get("/atlas/manifest")
    def atlas_manifest() -> dict[str, str]:
        return {
            "atlas_image_url": "/atlas/image.nii.gz",
            "atlas_segmentation_url": "/atlas/segmentation.nii.gz",
            "roi_metadata_url": "/atlas/roi-metadata.json",
        }

    @app.get("/atlas/image.nii.gz")
    def atlas_image():
        path = Path(config.paths.atlas_image)
        if not path.exists():
            raise HTTPException(status_code=404, detail="atlas.nii.gz not found")
        return FileResponse(path)

    @app.get("/atlas/segmentation.nii.gz")
    def atlas_segmentation():
        path = Path(config.paths.atlas_segmentation)
        if not path.exists():
            raise HTTPException(status_code=404, detail="atlas_segmentation.nii.gz not found")
        return FileResponse(path)

    @app.get("/atlas/roi-metadata.json")
    def atlas_roi_metadata() -> list[dict[str, Any]]:
        return roi_metadata().to_dict(orient="records")

    @app.get("/overlays/{overlay_name}")
    def generated_overlay(overlay_name: str):
        path = overlay_dir / overlay_name
        if not path.exists():
            raise HTTPException(status_code=404, detail="Generated overlay not found.")
        return FileResponse(path)

    return app


def torch_load_metadata(checkpoint_path: Path) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return {
        "n_processes": int(checkpoint["n_processes"]),
        "feature_columns": checkpoint["feature_columns"],
    }


def run_api(config: ProjectConfig) -> None:
    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("Uvicorn must be installed to run the API.") from exc

    app = create_app(config)
    uvicorn.run(app, host=config.app.api_host, port=config.app.api_port)
