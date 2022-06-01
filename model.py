import torch
import numpy as np
class Model(torch.nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        n_classes: int,
        hidden_dim = 128,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        
        # Embeddings for each word in the vocabulary
        self.embeddings = torch.nn.Embedding(vocab_size, hidden_dim)
        
        # A single-layer LSTM encoder
        self.encoder = torch.nn.LSTM(input_size=hidden_dim,
                                     hidden_size=hidden_dim,
                                     num_layers=1,
                                     batch_first=True)
        
        # Dropout
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        # Projection for class logits
        self.output_projection =  torch.nn.Linear(
            in_features=hidden_dim,
            out_features=n_classes,
        )

    def forward(self, id, token_ids, label=None):
        # Create mask over all the positions where the input is padded
        mask = token_ids != 0  # 0 is the <pad> token.
        
        # Embed Inputs
        # shape: (batch_size, sequence_length, hidden_dim)
        embeddings = self.embeddings(token_ids)

        # Feed through LSTM and apply dropout
        # shape: (batch_size, sequence_length, hidden_dim)
        encodings, _ = self.encoder(embeddings)
        encodings = self.dropout(encodings)

        # We now need to reduce over the sequence dimension. There are lots of 
        # ways to accomplish this, all of which are kind of messy. We will go
        # for a conceptually simple approach: take a weighted average of all of
        # the encodings, where the only non-zero weight is for the last element.
      
        # We'll start by initializing the weight tensor with all zeros.
        # shape: (batch_size, sequence_len)
        weights = torch.zeros_like(token_ids)

        # Now, we're going to do some non-trivial indexing to set the weights
        # corresponding to the last sequence elements to 1. We'll need two things:

        # First, the indices of the last elements, which we can get by summing
        # over the mask tensor.
        # shape: (batch_size,)
        last_idx = mask.sum(dim=-1) - 1
        
        # Second, we'll need a vector of indices ranging from 0 to batch_size - 1.
        # (This is the non-obvious part, and was found by trial and error).
        arange = torch.arange(token_ids.shape[0])

        # Using these we can assign the weights
        weights[arange, last_idx] = 1.0

        # Finally we can take the weighted sum.
        # shape: (batch_size, hidden_dim)
        weighted_encodings = weights.unsqueeze(-1) * encodings
        last_encodings = weighted_encodings.sum(dim=1)

        # Project to get logits.
        # shape: (batch_size, n_classes)
        logits = self.output_projection(last_encodings)

        # Get the predicitons.
        # shape: (batch_size,)
        predictions = logits.argmax(dim=-1)
        no_answer_probability = self.sigmoid(logits.detach().numpy())


        output_dict = {
            'logits': logits,
            'predictions': predictions,
            'no_answer_probability': no_answer_probability
        }

        # Compute loss if labels are provided
        if label is not None:
            if label.get_device() == -1:
                label = label.type(torch.LongTensor)
            loss = torch.nn.functional.cross_entropy(input=logits, target=label)
            output_dict['loss'] = loss

        return output_dict

    def sigmoid(self, x):
        return 1 / (1 + np.exp(x[:,1]-x[:,0]))
