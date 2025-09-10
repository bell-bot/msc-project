from collections.abc import Callable
from typing import Any
from circuits.neurons.core import Signal, const
from circuits.sparse.compile import Graph, Node
from circuits.utils.misc import OrderedSet
from msc_project.circuits_custom.custom_logic_gates import get_positive_laplace_weights, get_random_identity_params
from numpy.random import uniform, RandomState

def custom_compiled_from_io(inputs: list[Signal], outputs: list[Signal], rs = None) -> Graph:
    """Compiles a graph for function f using dummy input and output=f(input)."""
    return CustomGraph(inputs, outputs, rs=rs)


def custom_compiled(function: Callable[..., list[Signal]], input_len: int, rs = None, **kwargs: Any) -> Graph:
    """Compiles a function into a graph."""
    inp = const("0" * input_len)
    if rs is not None:
        kwargs['rs'] = rs
    out = function(inp, **kwargs)
    return custom_compiled_from_io(inp, out, rs)


class CustomGraph(Graph):

    def __init__(self, inputs: list[Signal], outputs: list[Signal], rs = None) -> None:
        self.rs = rs
        
        super().__init__(inputs, outputs)

    def ensure_adjacent_parents(self, layers: list[list[Node]]) -> list[list[Node]]: # type: ignore
        """Copy signals to next layers, ensuring child.depth==parent.depth+1"""
        copies_by_layer: list[list[Node]] = [[] for _ in range(len(layers))]
        for layer_idx, layer in enumerate(layers):
            for node in layer:

                # Stop at outputs
                if len(node.children) == 0:
                    continue

                max_child_depth = max([c.depth for c in node.children])
                n_missing_layers = max_child_depth - (layer_idx + 1)
                if n_missing_layers <= 0:
                    continue

                # Create chain of copies
                copy_chain: list[Node] = []
                prev = node
                prev_name = prev.metadata.get('name', 'Node')
                for depth in range(layer_idx + 1, layer_idx + n_missing_layers + 1):
                    curr = Node(prev.metadata)
                    curr.depth = depth
                    (weight, bias) = get_random_identity_params(rs=self.rs)
                    curr.bias = bias
                    curr.add_parent(prev, weight=weight)
                    copy_chain.append(curr)
                    curr.metadata['name'] = f"{prev_name}" + "`"
                    prev = curr

                # Redirect children to appropriate copies
                for child in list(node.children):
                    if child.depth == -1:
                        raise ValueError("Child depth must be set")
                    elif child.depth <= layer_idx + 1:
                        continue
                    new_parent = copy_chain[child.depth - layer_idx - 2]
                    child.replace_parent(node, new_parent)

                # Add copies to their respective layers
                for i, copy_node in enumerate(copy_chain):
                    copies_by_layer[layer_idx + 1 + i].append(copy_node)

        # Add copies and record indices
        for i, layer in enumerate(layers):
            layer.extend(copies_by_layer[i])
            for j, node in enumerate(layer):
                node.column = j

        return layers