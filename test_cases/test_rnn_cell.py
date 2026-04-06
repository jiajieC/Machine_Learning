"""
Test module for the RNNCell implementation.
"""
from test_cases.test_utils import (
    compare_tensors, 
    load_test_data, 
    setup_rnncell_with_test_data,
    run_test,
    RNNCELL_TEST_DATA_PATH
)

def test_rnn_cell():
    """Test the RNNCell implementation against reference implementation."""
    test_data = load_test_data(RNNCELL_TEST_DATA_PATH)
    cell = setup_rnncell_with_test_data(test_data)
    
    # Get inputs
    dummy_input = test_data["dummy_input"]
    dummy_hidden = test_data["dummy_hidden"]

    # Run forward pass
    output = cell(dummy_input, dummy_hidden)

    # Compare with expected output using our helper
    compare_tensors(
        output, 
        test_data["expected_output"], 
        label="RNNCell output", 
        atol=1e-5
    )

def test_cell():
    """Run RNNCell test with formatting."""
    run_test(test_rnn_cell, "RNNCell")