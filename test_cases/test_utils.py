"""
Utility functions for testing RNN implementations.
"""
import torch
from torch import nn
import os
from rnn import RNN, SelfAttention, RNNCell, HybridLanguageModel
import numpy as np

# Test data paths
RNN_TEST_DATA_PATH = "test_data/rnn_test.pt"
ATTENTION_TEST_DATA_PATH = "test_data/attention_test.pt"
RNNCELL_TEST_DATA_PATH = "test_data/rnncell_test.pt"
TRAIN_TEST_DATA_PATH = "test_data/train_test.pt"
LM_TEST_DATA_PATH = "test_data/lm_test.pt"
GENERATION_TEST_DATA_PATH = "test_data/generation_test.pt"

def compare_tensors(
    actual: torch.Tensor, expected: torch.Tensor, label: str = "", atol: float = 1e-5
):
    """
    Compare two tensors and raise a ValueError with descriptive stats if they differ.
    Stats reported:
      - shape mismatch
      - maximum absolute difference
      - index of the maximum difference
      - mean absolute difference
      - L2 norm of the difference
      - preview of both tensors (first few elements)
    """
    # First check shape
    if actual.shape != expected.shape:
        raise ValueError(
            f"{label} shape mismatch! Expected {list(expected.shape)}, "
            f"got {list(actual.shape)}"
        )

    # Then check if close within the given tolerance
    if not torch.allclose(actual, expected, atol=atol):
        diff = actual - expected
        abs_diff = diff.abs()
        max_diff = abs_diff.max().item()
        max_diff_idx = torch.argmax(abs_diff).item()
        flat_actual = actual.view(-1)
        flat_expected = expected.view(-1)

        # Also compute mean absolute difference and L2 distance
        mean_diff = abs_diff.mean().item()
        l2_diff = torch.norm(diff).item()
        
        # Add a preview of both tensors (first few elements or slice around the max difference)
        # For 1D tensors, show elements around the max difference
        preview_size = 5  # Number of elements to show
        
        # Get tensor previews
        if len(actual.shape) == 1:
            # For 1D tensors, show elements around max_diff_idx
            start_idx = max(0, max_diff_idx - preview_size // 2)
            end_idx = min(len(flat_actual), start_idx + preview_size)
            actual_preview = flat_actual[start_idx:end_idx]
            expected_preview = flat_expected[start_idx:end_idx]
            preview_range = f"[{start_idx}:{end_idx}]"
        else:
            # For higher dimensional tensors, show the first few elements
            flat_preview_size = min(preview_size, flat_actual.numel())
            actual_preview = flat_actual[:flat_preview_size]
            expected_preview = flat_expected[:flat_preview_size]
            preview_range = f"[0:{flat_preview_size}]"
            
            # Also try to show the region with the max difference
            idx_tuple = np.unravel_index(max_diff_idx, actual.shape)
            region_info = f"\n  Region with max difference (index {idx_tuple}):"
            
            # Try to get a small slice around the max difference area
            slice_indices = []
            for i, dim_size in enumerate(actual.shape):
                idx = idx_tuple[i]
                start = max(0, idx - 1)
                end = min(dim_size, idx + 2)
                slice_indices.append(slice(start, end))
            
            # Create the slices for displaying the region with max difference
            try:
                actual_region = actual[tuple(slice_indices)]
                expected_region = expected[tuple(slice_indices)]
                region_preview = (
                    f"\n  Actual region:\n{actual_region}\n"
                    f"  Expected region:\n{expected_region}"
                )
                region_info += region_preview
            except:
                # If we couldn't extract a region (e.g., for complex shapes), skip this
                region_info = ""

        error_msg = (
            f"{label} values mismatch!\n"
            f"  Maximum absolute difference: {max_diff:.6f} (at index {max_diff_idx})\n"
            f"  Expected vs. got at that index: "
            f"{flat_expected[max_diff_idx]:.6f} vs. {flat_actual[max_diff_idx]:.6f}\n"
            f"  Mean absolute difference: {mean_diff:.6f}\n"
            f"  L2 norm of the difference: {l2_diff:.6f}\n"
            f"  Preview {preview_range}:\n"
            f"    Expected: {expected_preview.tolist()}\n"
            f"    Actual:   {actual_preview.tolist()}"
        )
        
        if 'region_info' in locals() and region_info:
            error_msg += region_info
            
        error_msg += f"\n  (allclose with atol={atol} failed)\n"
        
        raise ValueError(error_msg)
    
def load_test_data(test_data_path):
    """Load and validate test data."""
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data file '{test_data_path}' not found.")
    return torch.load(test_data_path)

def setup_rnn_with_test_data(test_data):
    """Configure an RNN with weights from test data."""
    # Create RNN with test parameters
    input_dim = test_data["input_dim"]
    hidden_dim = test_data["hidden_dim"]
    rnn = RNN(input_dim, hidden_dim)

    # Set weights from test data
    rnn.cell.i2h.weight = nn.Parameter(test_data["weight_i2h"])
    rnn.cell.i2h.bias = nn.Parameter(test_data["bias_i2h"])
    rnn.cell.h2h.weight = nn.Parameter(test_data["weight_h2h"])
    rnn.cell.h2h.bias = nn.Parameter(test_data["bias_h2h"])
    rnn.out.weight = nn.Parameter(test_data["weight_out"])
    rnn.out.bias = nn.Parameter(test_data["bias_out"])
    
    return rnn

def setup_attention_with_test_data(test_data):
    """Configure a SelfAttention module with weights from test data."""
    # Create attention module with test parameters
    hidden_dim = test_data["hidden_dim"]
    key_dim = test_data["key_dim"]
    value_dim = test_data["value_dim"]
    attention = SelfAttention(hidden_dim, key_dim, value_dim)

    # Set weights from test data
    attention.query_transform.weight = nn.Parameter(test_data["query_weight"])
    attention.key_transform.weight = nn.Parameter(test_data["key_weight"])
    attention.value_transform.weight = nn.Parameter(test_data["value_weight"])
    attention.query_transform.bias = nn.Parameter(test_data["query_bias"])
    attention.key_transform.bias = nn.Parameter(test_data["key_bias"])
    attention.value_transform.bias = nn.Parameter(test_data["value_bias"])
    attention.output_transform.weight = nn.Parameter(test_data["output_weight"])
    attention.output_transform.bias = nn.Parameter(test_data["output_bias"])
    
    return attention

def setup_rnncell_with_test_data(test_data):
    """Configure an RNNCell with weights from test data."""
    # Create RNNCell with test parameters
    input_dim = test_data["input_dim"]
    hidden_dim = test_data["hidden_dim"]
    cell = RNNCell(input_dim, hidden_dim)

    # Set weights from test data
    cell.i2h.weight = nn.Parameter(test_data["weight_i2h"])
    cell.i2h.bias = nn.Parameter(test_data["bias_i2h"])
    cell.h2h.weight = nn.Parameter(test_data["weight_h2h"])
    cell.h2h.bias = nn.Parameter(test_data["bias_h2h"])
    
    return cell

def create_dummy_dataloader(data):
    """Create a dummy DataLoader-like object for testing."""
    class DummyDataLoader:
        def __init__(self, data):
            self.data = data

        def __iter__(self):
            return iter(self.data)

        def __len__(self):
            return len(self.data)
    
    return DummyDataLoader(data)

def setup_language_model_with_test_data(test_data):
    """Configure an HybridLanguageModel with weights from test data."""
    # Create language model with test parameters
    embed_dim = test_data["embed_dim"]
    hidden_dim = test_data["hidden_dim"]
    vocab_size = test_data["vocab_size"]
    key_dim = test_data["key_dim"]
    value_dim = test_data["value_dim"]
    
    lm = HybridLanguageModel(embed_dim, hidden_dim, vocab_size, key_dim, value_dim)
    
    # Set embeddings weights
    lm.embeddings.weight = nn.Parameter(test_data["embedding_weight"])
    
    # Set RNN weights
    lm.rnn.cell.i2h.weight = nn.Parameter(test_data["rnn_i2h_weight"])
    lm.rnn.cell.i2h.bias = nn.Parameter(test_data["rnn_i2h_bias"])
    lm.rnn.cell.h2h.weight = nn.Parameter(test_data["rnn_h2h_weight"])
    lm.rnn.cell.h2h.bias = nn.Parameter(test_data["rnn_h2h_bias"])
    lm.rnn.out.weight = nn.Parameter(test_data["rnn_out_weight"])
    lm.rnn.out.bias = nn.Parameter(test_data["rnn_out_bias"])
    
    # Set attention weights
    lm.attention.query_transform.weight = nn.Parameter(test_data["query_weight"])
    lm.attention.key_transform.weight = nn.Parameter(test_data["key_weight"])
    lm.attention.value_transform.weight = nn.Parameter(test_data["value_weight"])
    lm.attention.query_transform.bias = nn.Parameter(test_data["query_bias"])
    lm.attention.key_transform.bias = nn.Parameter(test_data["key_bias"])
    lm.attention.value_transform.bias = nn.Parameter(test_data["value_bias"])
    lm.attention.output_transform.weight = nn.Parameter(test_data["attn_output_weight"])
    lm.attention.output_transform.bias = nn.Parameter(test_data["attn_output_bias"])
    
    # Set LM head weights
    lm.lm_head.weight = nn.Parameter(test_data["lm_head_weight"])
    lm.lm_head.bias = nn.Parameter(test_data["lm_head_bias"])
    
    return lm

def compare_token_sequences(actual, expected, label=""):
    """Compare token sequences and provide detailed error messages."""
    if not torch.all(torch.eq(actual, expected)):
        mismatch_info = []
        for i in range(min(len(actual), len(expected))):
            if actual[i] != expected[i]:
                mismatch_info.append(
                    f"  First mismatch at position {i}: "
                    f"expected {expected[i]}, got {actual[i]}"
                )
                break
        raise ValueError(
            f"{label} mismatch!\n"
            f"Expected: {expected.tolist()}\n"
            f"Got:      {actual.tolist()}\n" + "\n".join(mismatch_info)
        )
        
def run_test(test_func, name):
    """Run a test function with standard output formatting."""
    print(f"Testing {name}...", end="")
    test_func()
    print(" Passed!")