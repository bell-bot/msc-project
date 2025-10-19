import ast
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Dict
import graphviz

from circuits.neurons.core import Signal, const
from circuits.utils.misc import OrderedSet


@dataclass
class VisualisationNode:

    signal_id: int
    activation: bool | float
    parents: Dict["VisualisationNode", float] = field(default_factory=dict)
    bias: float = 0.0
    metadata: Dict[str, str] = field(default_factory=dict)
    is_input: bool = False
    is_output: bool = False
    is_constant: bool = False

    def add_parent(self, parent: "VisualisationNode", weight: float):
        """Add a parent node with associated weight"""
        self.parents[parent] = weight

    def __hash__(self):
        return self.signal_id

    def __eq__(self, other):
        return isinstance(other, VisualisationNode) and self.signal_id == other.signal_id


class CircuitVisualiser:

    def __init__(self, name: str = "circuit_graph"):
        self.name = name
        self.graph = graphviz.Digraph(name, comment=f"{name} computation graph")
        self.graph.attr(rankdir="TB")  # top to bottom layout to match load_nodes()-method

    def visualise_method(
        self, fn: Callable[..., list[Signal]], input_len: int, filename: str, input: str | None = None, **kwargs: Any
    ):
        if input:
            inp = const(input)
        else:
            inp = const("0" * input_len)

        out = fn(inp, **kwargs)

        return self.load_nodes_and_visualise(inp, out, filename)

    def create_node_from_signal(self, signal: Signal) -> VisualisationNode:
        node = VisualisationNode(
            signal_id=id(signal),
            activation=signal.activation,
            metadata=dict(signal.metadata) if hasattr(signal, "metadata") else {},
        )

        return node

    def load_nodes_and_visualise(
        self, inp_signals: list[Signal], out_signals: list[Signal], filename: str
    ) -> graphviz.Digraph:
        """
        Create nodes from signals using the exact same algorithm as load_nodes,
        then visualize the resulting graph structure.
        """

        inp_nodes = [self.create_node_from_signal(s) for s in inp_signals]
        out_nodes = [self.create_node_from_signal(s) for s in out_signals]

        for node in inp_nodes:
            node.is_input = True
        for node in out_nodes:
            node.is_output = True

        inp_set = OrderedSet(inp_nodes)
        out_set = OrderedSet(out_nodes)

        nodes = {k: v for k, v in zip(inp_signals + out_signals, inp_nodes + out_nodes)}
        signals = {v: k for k, v in nodes.items()}

        seen: OrderedSet[VisualisationNode] = OrderedSet()
        frontier = out_nodes
        disconnected = True
        constants: OrderedSet[VisualisationNode] = OrderedSet()

        for i, inp in enumerate(inp_nodes):
            if inp.metadata.get("name") is None:
                inp.metadata["name"] = f"i{i}"

        # Go backwards from output nodes to record all connections
        while frontier:
            new_frontier: OrderedSet["VisualisationNode"] = OrderedSet()
            seen.update(frontier)
            for child in frontier:

                # Stop at inputs, they could have parents
                if child in inp_set:
                    disconnected = False
                    continue

                # Record parents of frontier nodes
                neuron = signals[child].source
                child.bias = neuron.bias
                for i, p in enumerate(neuron.incoming):
                    if p not in nodes:
                        nodes[p] = self.create_node_from_signal(p)
                        signals[nodes[p]] = p
                    parent = nodes[p]
                    if parent not in seen:
                        new_frontier.add(parent)
                    child.add_parent(parent, weight=neuron.weights[i])

                if len(child.parents) == 0:
                    constants.add(child)
                    child.is_constant = True

            frontier = list(new_frontier)

        for c in constants:
            if c not in inp_set and c not in out_set:
                c.is_constant = True

        print(f"Graph analysis complete:")
        print(f"  - Input nodes: {len(inp_nodes)}")
        print(f"  - Output nodes: {len(out_nodes)}")
        print(f"  - Total nodes: {len(nodes)}")
        print(f"  - Constants: {len(constants)}")
        print(f"  - Disconnected: {disconnected}")

        return self._visualize_node_graph(list(nodes.values()), inp_set, out_set, constants, filename)

    def _visualize_node_graph(
        self,
        all_nodes: list[VisualisationNode],
        inp_set: OrderedSet,
        out_set: OrderedSet,
        constants: OrderedSet,
        filename: str,
    ) -> graphviz.Digraph:
        """Create visual representation of the node graph"""

        # Add all nodes to the graph
        for node in all_nodes:
            node_id = f"node_{node.signal_id}"

            # Determine node styling based on type
            if node.is_input or node in inp_set:
                # Input nodes
                name = node.metadata.get("name", f"Input_{node.signal_id}")
                label = f"Input {name}\\nval={node.activation}"
                self.graph.node(node_id, label, shape="ellipse", style="filled", fillcolor="lightblue")

            elif node.is_output or node in out_set:
                # Output nodes
                label = f"Output"
                if node.bias != 0:
                    label += f"\\nbias={node.bias}"
                label += f"\\n\\nval={node.activation}"
                self.graph.node(node_id, label, shape="ellipse", style="filled", fillcolor="lightcoral")

            elif node.is_constant or node in constants:
                # Constant nodes
                label = f"Const\\nval={node.activation}"
                self.graph.node(
                    node_id, label, shape="diamond", style="filled", fillcolor="lightyellow"
                )

            else:
                # Internal nodes (neurons/gates)
                label = f"Gate"
                if node.bias != 0:
                    label += f"\\nbias={node.bias}"
                label += f"\\n\\nval={node.activation}"
                self.graph.node(node_id, label, shape="box", style="filled", fillcolor="darkseagreen1")

        # Add edges based on parent-child relationships
        for node in all_nodes:
            child_id = f"node_{node.signal_id}"

            for parent, weight in node.parents.items():
                parent_id = f"node_{parent.signal_id}"

                # Create edge from parent to child with weight label
                weight_label = str(weight) if weight != 1 else ""
                self.graph.edge(parent_id, child_id, label=weight_label)

        # Render if filename provided
        if filename:
            self.graph.unflatten(stagger=5).render(filename, format="pdf", cleanup=True)
            print(f"Graph saved as {filename}.pdf")

        return self.graph
