"""
Test module for the RNN implementation.
"""
from test_cases.test_utils import (
    compare_tensors, load_test_data, setup_rnn_with_test_data, run_test, RNN_TEST_DATA_PATH
)

def test_rnn_forward():
    """Test the RNN forward pass implementation."""
    test_data = load_test_data(RNN_TEST_DATA_PATH)
    rnn = setup_rnn_with_test_data(test_data)
    
    # Get input sequence and run forward pass
    input_sequence = test_data["input_sequence"]
    hidden_states, output_states = rnn(input_sequence)

    # Compare with expected outputs
    compare_tensors(
        hidden_states, test_data["expected_hidden"], 
        label="RNN hidden states", atol=1e-5
    )
    compare_tensors(
        output_states, test_data["expected_output"], 
        label="RNN output states", atol=1e-5
    )

def test_rnn_step():
    """Test the RNN step function."""
    test_data = load_test_data(RNN_TEST_DATA_PATH)
    rnn = setup_rnn_with_test_data(test_data)
    
    # Test step function
    step_input = test_data["step_input"]
    step_hidden = test_data["step_hidden"]
    next_hidden, next_output = rnn.step(step_input, step_hidden)

    # Compare with expected outputs
    compare_tensors(
        next_hidden, test_data["expected_step_hidden"], 
        label="RNN.step hidden", atol=1e-5
    )
    compare_tensors(
        next_output, test_data["expected_step_output"], 
        label="RNN.step output", atol=1e-5
    )

def test_rnn():
    """Run all RNN tests."""
    run_test(test_rnn_forward, "RNN forward pass")
    run_test(test_rnn_step, "RNN step function")