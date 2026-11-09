from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Tuple


Context = Dict[str, Any]


@dataclass
class Task:
    """
    One pipeline stage.

    Each task receives a mutable `context` dictionary. `requires` lists the
    context keys the task reads and `produces` the keys it writes; the graph
    derives its dependency edges from these, so the data flow between stages
    is explicit rather than implied by list position.
    """

    name: str
    fn: Callable[[Context], None]
    requires: FrozenSet[str] = frozenset()
    produces: FrozenSet[str] = frozenset()

    def run(self, context: Context) -> None:
        self.fn(context)


@dataclass
class TaskGraph:
    """
    Directed acyclic graph of pipeline tasks.

    Nodes are added in the order given by config/pipeline.yaml. An edge
    u -> v is created when v requires a key whose most recent producer
    before v is u, i.e. v consumes data that u writes. `validate()` checks
    that every required key has a producer, then orders the tasks with
    Kahn's topological sort, which also rejects cycles. `run()` executes
    the tasks in that topological order.
    """

    tasks: List[Task] = field(default_factory=list)
    edges: List[Tuple[int, int]] = field(default_factory=list)

    def add_task(self, task: Task) -> None:
        """Append a task and connect it to the earlier tasks it depends on."""
        v = len(self.tasks)
        for key in task.requires:
            producer = next((u for u in range(v - 1, -1, -1)
                             if key in self.tasks[u].produces), None)
            if producer is not None and (producer, v) not in self.edges:
                self.edges.append((producer, v))
        self.tasks.append(task)

    def missing_inputs(self) -> List[Tuple[str, str]]:
        """(task, key) pairs whose required key no earlier task produces."""
        missing = []
        for v, task in enumerate(self.tasks):
            for key in sorted(task.requires):
                if not any(key in self.tasks[u].produces for u in range(v)):
                    missing.append((task.name, key))
        return missing

    def unused_outputs(self, final_keys: FrozenSet[str]) -> List[str]:
        """Tasks none of whose outputs is consumed later or is a final result."""
        consumed = {u for u, _ in self.edges}
        return [t.name for i, t in enumerate(self.tasks)
                if i not in consumed and not (t.produces & final_keys)]

    def topological_order(self) -> List[int]:
        """Kahn's algorithm; raises ValueError if the graph has a cycle."""
        indeg = [0] * len(self.tasks)
        succ: Dict[int, List[int]] = {i: [] for i in range(len(self.tasks))}
        for u, v in self.edges:
            succ[u].append(v)
            indeg[v] += 1
        ready = deque(i for i in range(len(self.tasks)) if indeg[i] == 0)
        order = []
        while ready:
            u = ready.popleft()
            order.append(u)
            for v in succ[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    ready.append(v)
        if len(order) != len(self.tasks):
            raise ValueError("pipeline graph contains a cycle")
        return order

    def validate(self) -> List[int]:
        """Raise ValueError on a missing input or a cycle; return the run order."""
        missing = self.missing_inputs()
        if missing:
            detail = ", ".join(f"'{t}' needs '{k}'" for t, k in missing)
            raise ValueError(f"no earlier stage produces the required input: {detail}")
        return self.topological_order()

    def run(self, context: Context) -> None:
        for i in self.validate():
            self.tasks[i].run(context)
