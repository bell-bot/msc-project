from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from circuits.neurons.core import Neuron, Signal

@dataclass
class BundledBit:

    bundle: tuple[Signal, Signal, Signal]

def bundled_gate(incoming: list[BundledBit], weights: list[int], threshold: int) -> BundledBit:

    output_signals = []

    for i in range(len(incoming)):
        line_inputs = [bundle.bundle[i] for bundle in incoming]

        line_output = Neuron(tuple(line_inputs), tuple(weights), -threshold, step).outgoing

        output_signals.append(line_output)

    return BundledBit(tuple(output_signals))