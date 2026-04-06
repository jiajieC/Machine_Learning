"""
Main test runner for RNN implementation.
This file orchestrates the tests but delegates the actual test logic to separate modules.
"""
import warnings
import os
from torch.cuda import is_available
from test_cases.test_utils import compare_tensors
from test_cases.test_rnn_cell import test_rnn_cell
from test_cases.test_rnn import test_rnn
from test_cases.test_attention import test_attention
from test_cases.test_hybrid_language_model import test_hybrid_language_model
from test_cases.test_train import test_train
from test_cases.test_generation import test_generation
from test_cases.test_validate import test_validation

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

def run_tests():
    """Run all tests for the RNN implementation."""

    tests = [
        ("RNNCell", test_rnn_cell),
        ("RNN", test_rnn),
        ("SelfAttention", test_attention),
        ("HybridLanguageModel", test_hybrid_language_model),
        ("train", test_train),
        ("validate", test_validation),
        ("HybridLanguageModel.generate", test_generation),
    ]

    passed = 0
    total = len(tests)

    print("\n===== Starting RNN Implementation Tests =====\n")
    print(
        "IMPORTANT NOTE: Tests are independent. If a later test passes but an earlier test fails,"
    )
    print(
        "it does NOT mean the earlier component is correct. For example, if 'HybridLanguageModel.generate'"
    )
    print(
        "passes but 'HybridLanguageModel' fails, your language model is still incorrect and needs fixing."
    )
    print("You must pass ALL tests for your implementation to be fully correct.\n")
    if is_available():
        print("===== Test Results Summary =====")
        print("You are currently on a gpu. If you are on colab, please switch to a cpu to run these tests.")
        print("If you are running this locally, manually set DEVICE=\"cpu\" on line 17. Don't forget to change this back when running your empiricals.")
        return
    for test_name, test_func in tests:
        print(f"Running test: {test_name}...")
        try:
            test_func()
            passed += 1
            print(f"{test_name}: ✓ PASSED\n")
        except Exception as e:
            print(f" ✗ FAILED")
            print(f"Error details:\n{str(e)}\n")

    print("===== Test Results Summary =====")
    print(f"Passed: {passed}/{total} tests")

    if passed == total:
        print("🎉 All tests passed! Your implementation is correct.")
    else:
        print(f"❌ {total - passed} test(s) failed. Please review your implementation.")
        print(
            "Tip: Look at the detailed error messages above to see how far off your outputs were."
        )
        print(
            "Remember: Tests are independent, so passing a later test doesn't guarantee correctness of earlier components."
        )

    print("\n====================================")


if __name__ == "__main__":
    run_tests()