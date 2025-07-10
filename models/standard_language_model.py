import math
import torch
import torch.nn as nn

from models.positional_encoding import PositionalEncoding

class StandardLanguageModel(nn.Module):

    def __init__(self, vocab_size, model_dim, hidden_dim, n_layers, n_heads, ff_dim, dropout=0):
        super().__init__()
        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.token_embeddings = nn.Embedding(vocab_size, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.unembedding = nn.Linear(model_dim, vocab_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        
        x = self.token_embeddings(x) * math.sqrt(self.model_dim)
        x = self.pos_encoder(x)

        output = self.encoder(x, mask)
        logits = self.unembedding(output)

        return logits