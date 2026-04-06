import torch
from torch import nn
from test_cases.test_utils import compare_tensors, load_test_data, run_test
from rnn import HybridLanguageModel, validate

VALIDATE_TEST_DATA_PATH = "test_data/validate_test.pt"


def test_validate():
    """Test the validate function implementation."""
    test_data = load_test_data(VALIDATE_TEST_DATA_PATH)

    # Reconstruct model
    embed_dim = test_data["embed_dim"]
    hidden_dim = test_data["hidden_dim"]
    vocab_size = test_data["vocab_size"]
    key_dim = test_data["key_dim"]
    value_dim = test_data["value_dim"]

    lm = HybridLanguageModel(embed_dim, hidden_dim, vocab_size, key_dim, value_dim)
    lm.load_state_dict(test_data["state_dict"])
    lm = lm.to("cpu")

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # Load dataset
    valid_dataset = test_data["valid_dataset"]

    # Run validate
    avg_loss = validate(lm, valid_dataset, loss_fn)

    # Compare
    expected_loss = test_data["expected_loss"]
    compare_tensors(
        torch.tensor([avg_loss]),
        torch.tensor([expected_loss]),
        label="Validation loss",
        atol=1e-8,
    )


def test_validation():
    """If you're looking at this you're (probably) almost done. You got this!"""
    run_test(test_validate, "validate function")
