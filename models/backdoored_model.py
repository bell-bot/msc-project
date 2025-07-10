import torch
import torch.nn as nn

from models.standard_language_model import StandardLanguageModel

class BackdooredModel(nn.Module):

    def __init__(self, standard_model: StandardLanguageModel, circuit, circuit_dim):
        self.standard_model = standard_model
        self.circuit = circuit
        self.original_dim = standard_model.model_dim
        self.circuit_dim = circuit_dim
        self.augmented_dim = self.original_dim + self.circuit_dim

        self.augmented_embedding = nn.Embedding(standard_model.token_embeddings.num_embeddings, self.augmented_dim)
        self.augmented_embedding.weight.data[:, :self.original_dim] = standard_model.token_embeddings.weight.data
        self.augmented_embedding.weight.data[:, self.original_dim:] = 0

        self.augmented_unembedding = nn.Linear(self.augmented_dim, standard_model.unembedding.out_features)
        # Initialize the new unembedding: copy original weights and leave circuit part zero
        self.augmented_unembedding.weight.data[:, :self.original_dim] = standard_model.unembedding.weight.data
        self.augmented_unembedding.weight.data[:, self.original_dim:] = 0

    def forward(self, input_ids):
        augmented_x = self.augmented_embedding(input_ids) 

        original_stream, circuit_stream = torch.split(
            augmented_x, [self.original_dim, self.circuit_dim], dim=-1
        )

        processed_original_stream = self.standard_model.encoder(original_stream)
        processed_circuit_stream = self.circuit(circuit_stream)
        final_original_stream = processed_original_stream + processed_circuit_stream
        final_augmented_stream = torch.cat([final_original_stream, circuit_stream], dim=-1)
        logits = self.augmented_unembedding(final_augmented_stream)

        return logits