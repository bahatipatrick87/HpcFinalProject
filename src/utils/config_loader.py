import os
from typing import Any, Dict

import yaml


def load_pipeline_config(config_path: str | None = None) -> Dict[str, Any]:
    """
    Load the YAML configuration for the pipeline.

    If no path is provided, defaults to `config/pipeline.yaml` at the project root.
    This is the single source of truth for the "user-customisable workflow":
    every MPI rank calls this independently (no broadcast needed, since the
    file is read identically by every process) to learn which stages to run,
    in what order, and whether task-parallel prefetch is enabled.
    """
    if config_path is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        config_path = os.path.join(project_root, "config", "pipeline.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg or {}


def get_dataset_root(cfg: Dict[str, Any]) -> str:
    """
    Resolve the dataset root directory.

    The configuration may choose between:
      - data/raw/eurosat (preferred structure)
      - data/EuroSAT      (legacy layout already used in this project)
    This helper ensures backward compatibility.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # 1. If an explicit `root` is given, honor it relative to project root
    dataset_cfg = cfg.get("dataset", {}) if (cfg := cfg) else {}
    root_rel = dataset_cfg.get("root", "data/raw/eurosat")
    root_abs = os.path.join(project_root, root_rel)

    if os.path.exists(root_abs):
        return root_abs

    # 2. Fallback to legacy EuroSAT path: the dataset was originally unpacked
    # under data/EuroSAT before the config-driven layout was introduced, and
    # the dataset is large enough that re-downloading/reorganising it just to
    # match the new convention was not worthwhile.
    legacy = os.path.join(project_root, "data", "EuroSAT")
    if os.path.exists(legacy):
        return legacy

    # 3. Neither path exists yet: return the configured path anyway so the
    # caller's own "no files found" handling (simulated-data fallback) can
    # kick in, rather than raising here.
    return root_abs

