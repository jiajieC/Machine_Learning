"""
Test module for the training functionality.
Tests the train function with dummy data loaders.
"""
import torch
from torch import nn, optim
from test_cases.test_utils import (
    compare_tensors, 
    load_test_data, 
    create_dummy_dataloader,
    run_test,
    TRAIN_TEST_DATA_PATH
)
from rnn import HybridLanguageModel, train

def test_train():
    """Test the train function implementation."""
    test_data = load_test_data(TRAIN_TEST_DATA_PATH)
    
    # Create model with test parameters
    embed_dim = test_data["embed_dim"]
    hidden_dim = test_data["hidden_dim"]
    vocab_size = test_data["vocab_size"]
    key_dim = test_data["key_dim"]
    value_dim = test_data["value_dim"]
    
    # Initialize the model with the initial state dict
    torch.manual_seed(10601)  # For reproducibility
    lm = HybridLanguageModel(embed_dim, hidden_dim, vocab_size, key_dim, value_dim)
    lm.load_state_dict(test_data["initial_state_dict"])

    # Create loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lm.parameters(), lr=1e-3)

    # Create dummy DataLoader-like objects
    train_dataloader = create_dummy_dataloader(test_data["train_data"])
    valid_dataloader = create_dummy_dataloader(test_data["valid_data"])

    # Get parameters for training
    batch_size = test_data["batch_size"]
    num_sequences = test_data["num_sequences"]

    # Run train function
    train_losses, valid_losses = train(
        lm,
        train_dataloader,
        valid_dataloader,
        loss_fn,
        optimizer,
        num_sequences,
        batch_size,
    )

    # Compare losses as tensors for precision
    train_losses_tensor = torch.tensor(train_losses)
    expected_train_losses_tensor = torch.tensor(test_data["expected_train_losses"])
    
    valid_losses_tensor = torch.tensor(valid_losses)
    expected_valid_losses_tensor = torch.tensor(test_data["expected_valid_losses"])
    
    # Check dimensions match
    if len(train_losses) != len(test_data["expected_train_losses"]):
        raise ValueError(
            f"Number of training losses mismatch! "
            f"Expected {len(test_data['expected_train_losses'])}, got {len(train_losses)}"
        )
        
    if len(valid_losses) != len(test_data["expected_valid_losses"]):
        raise ValueError(
            f"Number of validation losses mismatch! "
            f"Expected {len(test_data['expected_valid_losses'])}, got {len(valid_losses)}"
        )
    
    # Compare expected vs actual values
    compare_tensors(
        train_losses_tensor, 
        expected_train_losses_tensor, 
        label="Training losses", 
        atol=1e-8
    )
    
    compare_tensors(
        valid_losses_tensor, 
        expected_valid_losses_tensor, 
        label="Validation losses", 
        atol=1e-8
    )

    # Check if losses are valid (not NaN, not huge, etc.)
    for loss in train_losses + valid_losses:
        if torch.isnan(torch.tensor(loss)) or loss > 1e4:
            raise ValueError(
                f"Found invalid loss value: {loss}. This suggests "
                f"an issue in the optimization process."
            )

def test_training():
    """Run training test with formatting."""
    run_test(test_train, "train function")