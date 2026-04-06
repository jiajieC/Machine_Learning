"""
RNN starter code
"""

import torch
from torch import nn, optim, Tensor
from typing import Optional
import json
from transformers import AutoTokenizer
import time

# DO NOT CHANGE THIS LINE!
# And DO NOT reset the torch seed anywhere else in your code!
torch.manual_seed(10601)

# Initialize torch device to use cuda if we have a gpu
DEVICE = (
    "cuda"
    if torch.cuda.is_available() else "cpu"
)


class SentenceDataset:
    def __init__(self, a):
        with open(a) as f:
            data = json.load(f)
            data = [torch.tensor(seq) for seq in data]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class RNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        input_dim (int): Input dimension of RNN
        hidden_dim (int): Hidden dimension of RNN
        """
        super(RNNCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # TODO: Initialize weights
        self.i2h = nn.Linear(input_dim, hidden_dim)
        self.h2h = nn.Linear(hidden_dim, hidden_dim)

        # See here for PyTorch activation functions
        # https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
        self.activation = nn.ReLU()

    def forward(self, input: Tensor, hidden_state: Tensor) -> Tensor:
        """
        Args:
            input (Tensor): Input at timestep t
                - shape: (batch_size, input_dim,)
            hidden_state (Tensor): Hidden state from timestep t-1
                - shape: (batch_size, hidden_dim,)

        Returns:
            Tensor: Next hidden state at timestep t
                - shape: (batch_size, hidden_dim)
        """
        # TODO: fill this in
        out = self.activation(self.i2h(input) + self.h2h(hidden_state))

        return out


class SelfAttention(nn.Module):
    """Scaled dot product attention from original transformers paper"""

    def __init__(self, hidden_dim, key_dim, value_dim):
        """
        hidden_dim (int): Hidden dimension of RNN
        key_dim (int): Dimension of attention key and query vectors
        value_dim (int): Dimension of attention value vectors
        """
        super(SelfAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.key_dim = key_dim
        self.value_dim = value_dim

        # TODO: Initialize Query, Key, and Value transformations
        self.query_transform = nn.Linear(hidden_dim, key_dim)
        self.key_transform = nn.Linear(hidden_dim, key_dim)
        self.value_transform = nn.Linear(hidden_dim, value_dim)

        # Output projection within the Attention Layer (NOT the LM head)
        self.output_transform = nn.Linear(value_dim, hidden_dim)

    def step(self, y_all: Tensor) -> Tensor:
        """
        Compute attention for **current** timestep t

        Args:
            y_all (Tensor): RNN layer outputs up to current timestep t
                - shape (batch_size, t, hidden_dim)

        Returns:
            Tensor: Attention output for current timestep
                - shape (batch_size, hidden_dim,)
        """
        last_hidden_output_state = y_all[:, -1].unsqueeze(1)

        # TODO: Compute the QKV values
        query = self.query_transform(last_hidden_output_state)
        keys = self.key_transform(y_all)
        values = self.value_transform(y_all)

        scaling = self.key_dim ** (0.5)
        query = query / scaling

        # TODO: Compute attention weights over values
        # Remember to divide raw attention scores by scaling factor
        # These scores should then be normalized using softmax
        # Hint: use torch.softmax
        weights = torch.softmax(torch.bmm(keys, query.transpose(1, 2)), dim=1)

        # TODO: Compute weighted sum of values based on attention weights
        output_state = torch.bmm(weights.transpose(1, 2), values)

        # Apply output projection back to hidden dimension
        output_state = self.output_transform(output_state).squeeze(1)

        return output_state

    def forward(self, y_all) -> Tensor:
        """
        Compute attention for all timesteps

        Args:
            y_all (Tensor): RNN layer outputs up to current timestep t
                - shape (batch_size, t, hidden_dim)

        Returns:
            Tensor: Attention output for all timesteps
                - shape (batch_size, t, hidden_dim)
        """
        t = y_all.shape[1]
        output_states = []

        for i in range(t):
            # TODO: Perform a step of SelfAttention and unsqueeze the result,
            # Then add it to the output states
            # HINT: use self.step()
            output_state = self.step(y_all[:, : i + 1, :]).unsqueeze(1)
            output_states.append(output_state)

        # TODO: torch.cat() all of the outputs in the list
        # across the sequence length dimension (t)
        output_states = torch.cat(output_states, dim=1)

        return output_states


class RNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
    ):
        """
        input_dim (int): Input dimension of RNN
        hidden_dim (int): Hidden dimension of RNN
        """
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # TODO: Initialize the RNNCell Class
        self.cell = RNNCell(input_dim, hidden_dim)

        # TODO: Initialize the weights
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def step(self, input: Tensor, hidden_prev: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        """
        Compute hidden and output states for a single timestep

        Args:
            input (Tensor): input at current timestep t
                - shape: (batch_size, input_dim,)
            hidden_prev (Tensor): hidden states of preceding timesteps [1, t-1]
                If there are no previous hidden states (i.e. we are at t=1), then
                this may be None and we will initialize the previous hidden state
                to all zeros.
                - shape: (batch_size, t-1, hidden_dim)

        Returns:
            Tensor: RNN hidden state at current timestep t
                - shape: (batch_size, hidden_dim,)
            Tensor: RNN output state at current timestep t.
                - shape: (batch_size, hidden_dim,)
        """
        if hidden_prev is None:
            # If this is the first timestep and there is no previous hidden state,
            # create a dummy hidden state of all zeros

            # TODO: Fill this in (After you intialize, make sure you add .to(input))
            batch_size = input.shape[0] if input.dim() > 1 else 1
            last_hidden_state = torch.zeros(batch_size, self.hidden_dim).to(input)
        else:
            # TODO: fill this in
            last_hidden_state = hidden_prev[:, -1, :] if hidden_prev.dim() == 3 else hidden_prev

        # Call the RNN cell and apply the transform to get a prediction
        next_hidden_state = self.cell(input, last_hidden_state)
        next_output_state = self.out(next_hidden_state)

        return next_hidden_state, next_output_state

    def forward(self, sequence: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute hidden and output states for all timesteps over input sequence

        Args:
            sequence (Tensor): inputs to RNN over t timesteps
                - shape (batch_size, t, input_dim)

        Returns:
            Tensor: hidden states over t timesteps
                - shape (batch_size, t, hidden_dim)
            Tensor: output states over t timesteps
                - shape (batch_size, t, hidden_dim)
        """
        hidden_states = None
        output_states = []
        _, t, _ = sequence.shape

        for i in range(t):
            # TODO: Extract the current input
            inp = sequence[:, i, :]

            # TODO: Call step() to get the next hidden/output states
            next_hidden_state, next_output_state = self.step(inp, hidden_states)
            next_hidden_state = next_hidden_state.unsqueeze(1)

            # TODO: Concatenate the newest hidden state to to all previous ones
            if hidden_states is None:
                hidden_states = next_hidden_state
            else:
                hidden_states = torch.cat([hidden_states, next_hidden_state], dim=1)

            # TODO: Append the next output state to the list
            output_states.append(next_output_state)

        # TODO: torch.stack all of the output states over the timestep dim
        output_states = torch.stack(output_states, dim=1)

        return hidden_states, output_states


class HybridLanguageModel(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        vocab_size,
        key_dim=None,
        value_dim=None,
    ):
        """
        embed_dim (int): Dimension of word embeddings
        hidden_dim (int): Dimension of RNN hidden states
        vocab_size (int): Number of (sub)words in model vocabulary
        """
        super(HybridLanguageModel, self).__init__()
        if key_dim is None:
            key_dim = hidden_dim
        if value_dim is None:
            value_dim = hidden_dim

        # TODO: Initialize word embeddings (HINT: use nn.Embedding)
        self.embeddings = nn.Embedding(vocab_size, embed_dim)

        # TODO: RNN backbone
        self.rnn = RNN(embed_dim, hidden_dim)

        # TODO: Self Attention Layer
        self.attention = SelfAttention(hidden_dim, key_dim, value_dim)

        # TODO: Final projection from RNN output state to next token logits
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Computes next-token logits and hidden states for each token in tokens

        Args:
            tokens (Tensor): Input tokens IDs
                - shape (batch_size, t,)

        Returns:
            Tensor: Next-token logits for each token from the LM head
                - shape (batch_size, t, vocab_size)
            Tensor: RNN hidden states for each token
                - shape (batch_size, t, hidden_dim)
            Tensor: RNN output states for each token
                - shape (batch_size, t, hidden_dim)
        """
        # TODO: Apply embeddings, rnns, and lm_head sequentially
        embedded_tokens = self.embeddings(tokens)
        hidden_states, attn_inputs = self.rnn(embedded_tokens)
        output_states = self.attention(attn_inputs)
        token_logits = self.lm_head(output_states)
        return token_logits, hidden_states, attn_inputs

    def select_token(self, token_logits: Tensor, temperature: float) -> int:
        """
        Selects (or samples) next token from token_logits

        Args:
            token_logits (Tensor): Next token logits
                - shape (vocab_size,)
            temperature (float): Sampling temperature. If 0, do greedy decoding.

        Returns:
            index (int): ID of next token selected
        """
        if temperature == 0:
            # Greedy Decoding
            return torch.argmax(token_logits, dim=-1)
        else:
            # Temperature Sampling
            token_logits = token_logits / temperature
            token_probs = torch.softmax(token_logits, dim=-1)
            index = torch.multinomial(token_probs, 1)[0]
            return index

    def generate(self, tokens: Tensor, max_tokens=10, temperature=0.0) -> Tensor:
        """
        Generates new tokens given `tokens` as a prefix.

        Args:
            tokens (Tensor): Input tokens
                - shape: (1, input_length,)
            max_tokens (int): Number of new tokens to generate
            temperature (float): Sampling temperature

        Returns:
            Tensor: generated tokens
                - shape: (max_tokens,)
        """
        # Get hidden states for input tokens by calling forward
        token_logits, hidden_states, attn_inputs = self(tokens)
        next_token_logits = token_logits[0, -1]

        new_tokens = []
        step = 0

        # Now, start generating new tokens
        # While we could in theory repeatedly call self(tokens) here, we don't since
        # that's an order of magnitude more inefficient as we would be repeatedly re-encoding
        # the prefix. Instead, here, we repeatedly compute the hidden state and next token
        # logits for the *latest* token.
        while True:
            step += 1

            # Select next token based on next_token_logits
            next_token = self.select_token(next_token_logits, temperature)
            new_tokens.append(next_token.item())

            # Stop generating once we reach max_tokens
            if step >= max_tokens:
                break

            # Get next input embedding
            embed = self.embeddings(next_token)

            # Get next hidden state and next attn input state from RNN
            next_hidden_state, next_attn_input = self.rnn.step(embed, hidden_states)

            # Update hidden states
            hidden_states = torch.cat(
                [hidden_states, next_hidden_state.unsqueeze(1)], dim=1
            )

            # Update attention inputs
            attn_inputs = torch.cat([attn_inputs, next_attn_input.unsqueeze(1)], dim=1)

            # Call attention
            next_output_state = self.attention.step(attn_inputs)

            # Generate the token to be used in the next step of generation
            next_token_logits = self.lm_head(next_output_state)

        return torch.tensor(new_tokens)


def train(lm, train_data, valid_data, loss_fn, optimizer, num_sequences, batch_size):
    """
    Run one epoch of language model training

    Args:
        lm (HybridLanguageModel): Hybrid RNN language model
        train_data (list[Tensor]): Train dataset
        valid_data (list[Tensor]): Validation dataset
        loss_fn: PyTorch cross entropy loss function
        optimizer: PyTorch Adam optimizer
        num_sequences: The total number of sequences to train on
        batch_size: Number of sequences we process in one step

    Returns:
        List: Training losses
        List: Validation Losses
    """
    # Set the model to training model
    lm.train()
    max_grad_norm = 1.0
    lm = lm.to(DEVICE)
    train_batch_losses = []
    train_batch_loss = 0.0
    valid_batch_losses = []

    # DO NOT change the next line
    dataset = train_data
    start_time = time.time()

    # Run validation everytime we process around 10% of the training data
    val_frequency = 0.1
    val_index = int(num_sequences * val_frequency) // batch_size
    if val_index == 0:
        val_index = 1

    # Loop over the dataset
    for idx, sequence in enumerate(dataset):
        time_elapsed = round((time.time() - start_time) / 60, 6)

        # Move the sequence to the device
        sequence = sequence.to(DEVICE)

        # Stop training when we hit the num_sequences limit
        if idx == num_sequences // batch_size:
            break

        # TODO: Zero gradients
        optimizer.zero_grad()

        # TODO: Forward pass through model
        token_logits, hidden_states, attn_inputs = lm(sequence)

        # TODO: Compute next-token classification loss

        # Hint 1: The token_logits should be of shape (batch_size, t, vocab_size),
        # and the sequence should be of shape (batch_size, t).
        # If we want to compute the loss of the nth logit token,
        # which token in the sequence should I compare it with?

        # Hint 2: We will need to permute the token_logits to the
        # correct shape before passing into loss function

        loss = loss_fn(token_logits[:, :-1, :].permute(0, 2, 1), sequence[:, 1:])

        # TODO: Backward pass through model
        loss.backward()

        # DO NOT change this - clip gradient norm to avoid exploding gradients
        nn.utils.clip_grad_norm_(lm.parameters(), max_grad_norm)

        # TODO: Update weights
        optimizer.step()

        # DO NOT change any of the code below
        train_batch_loss += loss.detach().cpu().item()

        if idx % val_index == 0:
            # Calculate train/val loss as normal
            train_batch_loss = (
                round(train_batch_loss / val_index, 6)
                if idx != 0
                else round(train_batch_loss, 6)
            )

            # Append to the batch loss and reset to 0
            train_batch_losses.append(train_batch_loss)
            train_batch_loss = 0.0

            print(
                f"Batch: {idx} | Sequence Length: {(sequence.shape[1])} | Elapsed time (minutes): {time_elapsed}"
            )

            # Append to the validation loss
            valid_loss = round(validate(lm, valid_data, loss_fn), 6)
            valid_batch_losses.append(valid_loss)
            lm.train()

    print(f"Train Batch Losses: {train_batch_losses}")

    return train_batch_losses, valid_batch_losses


@torch.no_grad()
def validate(lm, dataset, loss_fn):
    """
    Args:
        lm (HybridLanguageModel):
        dataset (list[Tensor]): Validation dataset
        loss_fn: PyTorch cross entropy loss function

    Returns:
        float: Average validation loss
    """
    # Set the model to eval mode
    lm.eval()

    mean_loss = 0.0
    num_batches = len(dataset)

    for i, sequence in enumerate(dataset):
        if i < num_batches:
            # Move the sequence to the device
            sequence = sequence.to(DEVICE)

            # TODO: Perform forward pass through the model
            token_dists, _, _ = lm(sequence)

            # TODO: Compute loss (Same as in train)
            loss = loss_fn(token_dists[:, :-1, :].permute(0, 2, 1), sequence[:, 1:])

            # DO NOT change this line
            mean_loss += loss.detach().cpu().item()

    return mean_loss / num_batches


@torch.no_grad()
def complete(
    lm: HybridLanguageModel,
    tokenizer: AutoTokenizer,
    prefix: str,
    num_tokens: int = 64,
    temperature: float = 0.0,
):
    """
    Generates text completion from language model given text prefix.
    This function has been implemented for you.

    Args:
        prefix (str):
        num_tokens (int): Number of new tokens to generate
        temperature (float): Sampling temperature

    Returns:
        str: Text completion
    """
    lm.eval()

    input = tokenizer.encode(prefix, add_special_tokens=False, return_tensors="pt")
    input = input.to(DEVICE)
    output = lm.generate(input, max_tokens=num_tokens, temperature=temperature)

    return tokenizer.decode(output)


def train_mod(args):
    tokenizer = AutoTokenizer.from_pretrained("my_tokenizer")
    vocab_size = tokenizer.vocab_size
    
    # initialize LM
    lm = HybridLanguageModel(
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        vocab_size=vocab_size,
        key_dim=args.dk,
        value_dim=args.dv,
    )
    lm = lm.to(DEVICE)
    print(lm)

    print(
        "Number of Parameters: ",
        sum(p.numel() for p in lm.parameters() if p.requires_grad),
    )

    print("Loading data")

    train_data = SentenceDataset(args.train_data)

    valid_data = SentenceDataset(args.val_data)

    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=False
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False
    )
    print("Finished Loading")

    # make PyTorch cross entropy loss function
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lm.parameters(), lr=1e-3)

    ### Training Loop
    start = time.time()
    train_loss, valid_loss = train(
        lm,
        train_dataloader,
        valid_dataloader,
        loss_fn,
        optimizer,
        args.num_sequences,
        args.batch_size,
    )
    end = time.time()
    time_taken = end - start
    ### Training Loop

    results = {
        "Train Losses": train_loss,
        "Valid Losses": valid_loss,
        "Final Train Loss": train_loss[-1],
        "Final Valid Loss": valid_loss[-1],
        "Time": time_taken,
    }
    print(results)
    print("Final Train Loss: ", train_loss[-1])
    print("Final Valid Loss: ", valid_loss[-1])
    print("Time: ", time_taken)

    # Saves your trained model weights
    if args.model_path != "":
        torch.save(lm, args.model_path)
    with open(args.train_losses_out, "w") as f:
        for loss in train_loss:
            f.write(str(loss) + "\n")

    with open(args.val_losses_out, "w") as f:
        for loss in valid_loss:
            f.write(str(loss) + "\n")

    with open(args.metrics_out, "w") as f:
        f.write("Final Train Loss: " + str(train_loss[-1]) + "\n")
        f.write("Final Valid Loss: " + str(valid_loss[-1]) + "\n")
def generate(args):
    lm = torch.load(args.model_path, weights_only = False)
    lm = lm.to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained("my_tokenizer")
    prefix = args.prefix
    completion = complete(lm, tokenizer, prefix, num_tokens=args.num_tokens, temperature=args.temperature)
    print("Prefix: ", prefix)
    print("Completion: ", completion)
    
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    import argparse

    parser = argparse.ArgumentParser()
    mode_parser = parser.add_subparsers(dest='mode', help='Type of call', required=True)
    
    train_parser = mode_parser.add_parser("train")

    train_parser.add_argument("--train_data", type=str)
    train_parser.add_argument("--val_data", type=str)
    train_parser.add_argument("--metrics_out", type=str)
    train_parser.add_argument("--train_losses_out", type=str)
    train_parser.add_argument("--val_losses_out", type=str)
    train_parser.add_argument("--embed_dim", type=int)
    train_parser.add_argument("--hidden_dim", type=int)
    train_parser.add_argument("--dk", type=int)
    train_parser.add_argument("--dv", type=int)
    train_parser.add_argument("--num_sequences", type=int)
    train_parser.add_argument("--batch_size", type=int, default=1)
    train_parser.add_argument("--model_path", type=str, default = "")

    generate_parser = mode_parser.add_parser("generate")

    generate_parser.add_argument("--model_path", type=str)
    generate_parser.add_argument("--prefix", type=str)
    generate_parser.add_argument("--temperature", type = float)
    generate_parser.add_argument("--num_tokens", type=int)
    train_parser.set_defaults(func=train_mod)
    generate_parser.set_defaults(func=generate)
    args = parser.parse_args()
    args.func(args)
