"""
Test module for the SelfAttention implementation.
Includes separate tests for forward pass and step function.
"""
from test_cases.test_utils import (
    compare_tensors, 
    load_test_data, 
    setup_attention_with_test_data,
    run_test,
    ATTENTION_TEST_DATA_PATH
)

def test_attention_forward():
    """Test the SelfAttention forward pass implementation."""
    test_data = load_test_data(ATTENTION_TEST_DATA_PATH)
    attention = setup_attention_with_test_data(test_data)
    
    # Get input sequence and run forward pass
    sequence = test_data["sequence"]
    output = attention(sequence)

    # Compare with expected outputs
    compare_tensors(
        output, 
        test_data["expected_output"], 
        label="SelfAttention output", 
        atol=1e-5
    )

def test_attention_step():
    """Test the SelfAttention step function."""
    test_data = load_test_data(ATTENTION_TEST_DATA_PATH)
    attention = setup_attention_with_test_data(test_data)
    
    # Test step function
    partial_sequence = test_data["partial_sequence"]
    partial_output = attention.step(partial_sequence)

    compare_tensors(
        partial_output,
        test_data["expected_step_output"],
        label="SelfAttention step output",
        atol=1e-5,
    )

def test_attention():
    """Run all SelfAttention tests."""
    run_test(test_attention_forward, "SelfAttention forward pass")
    run_test(test_attention_step, "SelfAttention step function")