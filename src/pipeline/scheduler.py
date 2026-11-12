from typing import Any, Dict

from .task_graph import TaskGraph, Context


def run_pipeline(graph: TaskGraph, context: Dict[str, Any] | None = None) -> Context:
    """
    Execute all tasks in the given TaskGraph in list order (linear DAG).

    Within each MPI rank, each stage processes the rank's whole share of the
    images before the next stage starts, so the stages never overlap. This is
    the "sequential mode" baseline (task_parallel_prefetch: false),
    used as the reference point to quantify the task-parallel pipeline's
    speedup in the report.
    Parallelism comes from: (1) data parallelism---MPI ranks and SIMD/OpenMP
    inside kernels; (2) functional decomposition---stages mapped to CPU vs GPU.
    Optional ``task_parallel_prefetch`` (see ``batched_pipeline``) uses a
    three-thread queue pipeline to overlap load, GPU FFT/filter, and CPU features
    across successive batches.
    """
    if context is None:
        context = {}
    graph.run(context)
    return context

