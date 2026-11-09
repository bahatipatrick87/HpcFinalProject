"""
Tests for the pipeline DAG (src/pipeline/task_graph.py, workflow_builder.py).

Checks that dependency edges are derived from each stage's declared inputs
and outputs, that invalid stage lists from config/pipeline.yaml are rejected
before any data is processed, and that tasks run in topological order.

Run from project root:
    python -m pytest tests/test_task_graph.py -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from pipeline.task_graph import Task, TaskGraph
from pipeline.workflow_builder import (STAGE_REGISTRY, build_pipeline_from_config,
                                       pipeline_warnings)


def build(stages):
    return build_pipeline_from_config(None, {"pipeline": {"stages": stages}})


class TestPipelineDAG(unittest.TestCase):

    def test_default_pipeline_edges(self):
        g = build(["io", "fft", "filter", "feature"])
        names = [t.name for t in g.tasks]
        edges = {(names[u], names[v]) for u, v in g.edges}
        self.assertEqual(edges, {("io_stage", "fft_stage"),
                                 ("fft_stage", "filter_stage"),
                                 ("filter_stage", "feature_stage")})
        self.assertEqual(g.topological_order(), [0, 1, 2, 3])
        self.assertEqual(pipeline_warnings(g), [])

    def test_filter_can_be_removed(self):
        # Without the filter, features are computed from the FFT output.
        g = build(["io", "fft", "feature"])
        names = [t.name for t in g.tasks]
        self.assertIn(("fft_stage", "feature_stage"),
                      {(names[u], names[v]) for u, v in g.edges})
        self.assertEqual(pipeline_warnings(g), [])

    def test_missing_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "'fft_stage' needs 'input_data'"):
            build(["fft", "filter", "feature"])

    def test_unknown_stage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown pipeline stage 'fltr'"):
            build(["io", "fft", "fltr", "feature"])

    def test_filter_after_feature_is_flagged(self):
        # Valid graph, but the filter's output is never used.
        g = build(["io", "fft", "feature", "filter"])
        self.assertEqual(pipeline_warnings(g),
                         ["stage 'filter_stage' runs but none of its outputs "
                          "is used by a later stage"])

    def test_cycle_is_rejected(self):
        g = TaskGraph()
        g.add_task(Task("a", lambda c: None, frozenset(), frozenset({"x"})))
        g.add_task(Task("b", lambda c: None, frozenset({"x"}), frozenset({"y"})))
        g.edges.append((1, 0))  # b -> a closes the cycle a -> b -> a
        with self.assertRaisesRegex(ValueError, "cycle"):
            g.topological_order()

    def test_run_follows_dependencies(self):
        order = []
        g = TaskGraph()
        for name, req, prod in [("load", set(), {"a"}), ("gpu", {"a"}, {"b"}),
                                ("cpu", {"b"}, {"c"})]:
            g.add_task(Task(name, lambda c, n=name: order.append(n),
                            frozenset(req), frozenset(prod)))
        g.run({})
        self.assertEqual(order, ["load", "gpu", "cpu"])

    def test_registry_covers_all_yaml_stage_names(self):
        self.assertEqual(set(STAGE_REGISTRY), {"io", "fft", "filter", "feature"})


if __name__ == "__main__":
    unittest.main()
