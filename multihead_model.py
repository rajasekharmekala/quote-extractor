import torch
import numpy as np
import math
import torch.nn.functional as F

class MultiHeadModel(torch.nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        n_classes: int,
        hidden_dim = 192,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        
        # Embeddings for each word in the vocabulary
        self.embeddings = torch.nn.Embedding(vocab_size, hidden_dim)
        self.self_attn = MultiheadAttention(input_dim=hidden_dim, embed_dim=192, num_heads=12)
        
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

    def forward(self, id, token_id_list, label):
        token_id_list = token_id_list.view(-1, 25)
        # print(token_id_list.shape, label.shape)
        mask = token_id_list != 0    
        embeddings = self.embeddings(token_id_list)
        encodings, _ = self.encoder(embeddings)
        encodings = self.dropout(encodings).view(-1,32, 25, 192)

        attn_out = self.self_attn(encodings.squeeze())
        attn_out = self.dropout(attn_out)


        weights = torch.zeros_like(token_id_list)
        last_idx = mask.sum(dim=-1) - 1
        arange = torch.arange(token_id_list.shape[0])
        weights[arange, last_idx] = 1.0

        weighted_encodings = weights.unsqueeze(-1) * attn_out
        last_encodings = weighted_encodings.sum(dim=1)

        logits = self.output_projection(last_encodings).view(-1,2)
        predictions = logits.argmax(dim=-1)
        # print(predictions.shape)


        output_dict = {
            'logits': logits,
            'predictions': predictions
        }

        # Compute loss if labels are provided
        if label is not None:
            if label.get_device() == -1:
                label = label.type(torch.LongTensor)
            loss = torch.nn.functional.cross_entropy(input=logits, target=label.view(-1))
            output_dict['loss'] = loss

        return output_dict

    def sigmoid(self, x):
        return 1 / (1 + np.exp(x[:,1]-x[:,0]))


class MultiheadAttention(torch.nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = torch.nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = torch.nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        torch.nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention