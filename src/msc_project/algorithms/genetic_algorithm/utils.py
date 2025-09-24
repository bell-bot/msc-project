from circuits.utils.format import Bits

def verify_ga_optimised_stepml(mlp_template, formatted_message: Bits, expected_output: Bits):
    actual_output = mlp_template.infer_bits(formatted_message)
    actual_output_text = actual_output.bitstr
    print(f"Expected output: {expected_output.bitstr}")
    print(f"Actual output: {actual_output_text}")
    
    if actual_output_text == expected_output.bitstr:
        print("GA-optimised StepMLP produces the expected output.")
    else:
        print("GA-optimised StepMLP does NOT produce the expected output.")
        raise ValueError("Output mismatch after GA optimisation.")