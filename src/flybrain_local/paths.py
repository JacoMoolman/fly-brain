from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    project: Path
    assets: Path
    dataset: Path
    prepared: Path
    results: Path

    @classmethod
    def discover(cls) -> "ProjectPaths":
        project = Path(
            os.environ.get("FLYBRAIN_PROJECT_ROOT", Path(__file__).resolve().parents[2])
        ).resolve()
        config = json.loads((project / "config" / "paths.json").read_text())
        assets = Path(os.environ.get("FLYBRAIN_ASSET_ROOT", config["assetRoot"]))
        dataset = Path(os.environ.get("FLYBRAIN_DATASET_ROOT", config["datasetRoot"]))
        paths = cls(
            project=project,
            assets=assets,
            dataset=dataset,
            prepared=assets / "prepared" / "malecns-v1.0",
            results=Path(config["resultRoot"]),
        )
        paths.validate()
        return paths

    def validate(self) -> None:
        for path in (self.project, self.assets, self.dataset, self.prepared, self.results):
            resolved = path.resolve()
            if resolved.drive.upper() == "C:":
                raise RuntimeError(f"Project storage guard rejected C drive path: {resolved}")
        for path in (self.assets, self.dataset, self.prepared, self.results):
            path.mkdir(parents=True, exist_ok=True)

