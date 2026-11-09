from typing import List

from mpi4py import MPI

from pipeline.task_graph import Task, TaskGraph
from stages.io_stage import io_load_and_partition
from stages.fft_stage import fft_frequency_stage
from stages.filter_stage import filter_stage
from stages.feature_stage import feature_extraction_stage


# Registry of stage names -> (task_name, fn, requires, produces) for the
# user-customizable workflow (proposal 0.5.5). Mapping stage IDs through this
# dict rather than calling functions directly by name is what lets
# config/pipeline.yaml's `stages` list reorder or drop stages without the
# config file needing to reference Python identifiers. `requires`/`produces`
# are the context keys each stage reads and writes; TaskGraph derives the
# dependency edges from them.
STAGE_REGISTRY = {
    "io": ("io_stage", io_load_and_partition,
           frozenset(), frozenset({"input_data"})),
    "fft": ("fft_stage", fft_frequency_stage,
            frozenset({"input_data"}), frozenset({"fft_data", "processed_data"})),
    "filter": ("filter_stage", filter_stage,
               frozenset({"fft_data"}), frozenset({"processed_data"})),
    "feature": ("feature_stage", feature_extraction_stage,
                frozenset({"processed_data"}), frozenset({"features"})),
}

# Context keys that are pipeline results (gathered by main_mpi.py), so the
# stage producing them is never reported as having an unused output.
FINAL_KEYS = frozenset({"features"})


def build_pipeline_from_config(comm: MPI.Comm, config: dict) -> TaskGraph:
    """
    Build and validate the task graph from config (user-customizable workflow).

    Nodes follow pipeline.stages in config; edges come from each stage's
    declared inputs and outputs. Stages can be enabled, disabled, or
    reordered by editing the YAML file, with no recompilation; an order in
    which a stage's input is not produced earlier, or an unknown stage name,
    raises ValueError on every rank before any data is processed.
    """
    graph = TaskGraph()
    pipeline_cfg = config.get("pipeline", {})
    stages = pipeline_cfg.get("stages", ["io", "fft", "filter", "feature"])

    for stage_id in stages:
        stage_id = str(stage_id).strip().lower()
        if stage_id not in STAGE_REGISTRY:
            raise ValueError(
                f"unknown pipeline stage '{stage_id}' in config "
                f"(valid: {', '.join(STAGE_REGISTRY)})")
        name, fn, requires, produces = STAGE_REGISTRY[stage_id]
        graph.add_task(Task(name=name, fn=fn, requires=requires, produces=produces))
    graph.validate()
    return graph


def pipeline_warnings(graph: TaskGraph) -> List[str]:
    """Human-readable warnings for a valid but questionable stage order."""
    return [f"stage '{name}' runs but none of its outputs is used by a later stage"
            for name in graph.unused_outputs(FINAL_KEYS)]


def build_default_pipeline(comm: MPI.Comm) -> TaskGraph:
    """
    Build the default four-stage pipeline (proposal DAG: I/O -> FFT -> Filtering -> Feature).
    """
    return build_pipeline_from_config(comm, {"pipeline": {"stages": ["io", "fft", "filter", "feature"]}})
