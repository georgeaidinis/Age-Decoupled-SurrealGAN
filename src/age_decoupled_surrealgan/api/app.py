from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import ProjectConfig
from ..inference import infer_single_row


def create_app(config: ProjectConfig):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse
        from pydantic import BaseModel, Field
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("FastAPI and Pydantic must be installed to run the API.") from exc

    class InferRequest(BaseModel):
        run_name: str
        split_name: str = Field(default="application")
        row_index: int | None = None
        feature_values: dict[str, float] | None = None

    app = FastAPI(title="Age-Decoupled SurrealGAN API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[config.app.allow_origin],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    runs_dir = Path(config.paths.runs_dir)
    processed_dir = Path(config.paths.processed_dir)

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

    def roi_metadata() -> pd.DataFrame:
        path = processed_dir / "roi_metadata.csv"
        if not path.exists():
            raise HTTPException(status_code=404, detail="ROI metadata not found. Run prepare-data first.")
        return pd.read_csv(path)

    def split_frame(split_name: str) -> pd.DataFrame:
        path = processed_dir / f"{split_name}.csv"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Split not found: {split_name}")
        frame = pd.read_csv(path)
        frame = frame.reset_index(drop=True)
        frame.insert(0, "row_index", frame.index)
        return frame

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
        }

    @app.get("/runs/{run_name}/splits/{split_name}")
    def get_split_subjects(run_name: str, split_name: str) -> list[dict[str, Any]]:
        _ = run_summary(run_name)
        frame = split_frame(split_name)
        columns = ["row_index", "subject_id", "study", "age", "sex", "diagnosis_raw", "diagnosis_group", "cohort_bucket"]
        return frame[columns].to_dict(orient="records")

    @app.post("/infer")
    def infer(request: InferRequest) -> dict[str, Any]:
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
                "age": None,
                "sex": "U",
                "diagnosis_raw": "manual",
                "diagnosis_group": "manual",
                "cohort_bucket": "manual",
            }
        else:
            if request.row_index is None:
                raise HTTPException(status_code=400, detail="row_index is required when feature_values is not provided.")
            frame = split_frame(request.split_name)
            if request.row_index < 0 or request.row_index >= len(frame):
                raise HTTPException(status_code=400, detail="row_index out of bounds")
            selected = frame.iloc[int(request.row_index)]
            row = selected
            metadata = selected[
                ["subject_id", "study", "age", "sex", "diagnosis_raw", "diagnosis_group", "cohort_bucket"]
            ].to_dict()

        inference = infer_single_row(
            checkpoint_path=checkpoint_path,
            row=row,
            feature_columns=feature_columns,
            reference_template=reference_template,
            device=config.training.device,
        )
        roi_df = roi_metadata().copy()
        roi_df["delta"] = inference["synthetic_delta"]
        roi_df["age_delta"] = inference["age_delta"]
        for idx in range(inference["n_processes"]):
            roi_df[f"process_{idx + 1}_delta"] = [values[idx] for values in zip(*inference["process_deltas"])]
        roi_df["abs_delta"] = roi_df["delta"].abs()
        top_changes = roi_df.sort_values("abs_delta", ascending=False).head(25)
        return {
            "metadata": metadata,
            "inference": inference,
            "roi_table": roi_df.to_dict(orient="records"),
            "top_changes": top_changes.to_dict(orient="records"),
        }

    @app.get("/atlas/manifest")
    def atlas_manifest() -> dict[str, str]:
        return {
            "atlas_image_url": "/atlas/image",
            "atlas_segmentation_url": "/atlas/segmentation",
            "roi_metadata_url": "/atlas/roi-metadata",
        }

    @app.get("/atlas/image")
    def atlas_image():
        path = Path(config.paths.atlas_image)
        if not path.exists():
            raise HTTPException(status_code=404, detail="atlas.nii.gz not found")
        return FileResponse(path)

    @app.get("/atlas/segmentation")
    def atlas_segmentation():
        path = Path(config.paths.atlas_segmentation)
        if not path.exists():
            raise HTTPException(status_code=404, detail="atlas_segmentation.nii.gz not found")
        return FileResponse(path)

    @app.get("/atlas/roi-metadata")
    def atlas_roi_metadata() -> list[dict[str, Any]]:
        return roi_metadata().to_dict(orient="records")

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
