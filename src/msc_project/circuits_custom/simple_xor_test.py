# Create a new file named simple_xor_test.py

from circuits.utils.format import Bits
from circuits.neurons.core import const, Bit
from msc_project.circuits_custom.custom_stepmlp import RandomisedStepMLP
from msc_project.circuits_custom.custom_compile import custom_compiled
from msc_project.circuits_custom.custom_logic_gates import custom_xor

# 1. Define the simple circuit to be compiled
def xor_circuit(x: list[Bit]):
    # We are testing a single, robust XOR gate
    return [custom_xor(x)]

# 2. Compile the function using your custom compiler
# No random state (rs) is needed yet, as we've made the gates deterministic
print("Compiling deterministic XOR circuit...")
graph = custom_compiled(xor_circuit, input_len=5)
mlp = RandomisedStepMLP.from_graph(graph)
print("Compilation complete.")

# 3. Test the resulting MLP
test_input = Bits("10110")
output = mlp.infer_bits(test_input)

# The XOR of 1+0+1+1+0 = 3 (odd), so the result should be 1.
expected_output = "1"

print(f"\nInput:    {test_input.bitstr}")
print(f"Output:   {output.bitstr}")
print(f"Expected: {expected_output}")

assert output.bitstr == expected_output
print("\n✅ Step 1 Passed: The deterministic robust XOR circuit is logically correct!")