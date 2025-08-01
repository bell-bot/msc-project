from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import Conv1D

class BackdooredGPT2Model(nn.Module):

    def __init__(
        self,
        standard_model,
        standard_model_dim,
        standard_model_embeddings,
        tokenizer,
        circuit,
        trigger_length: int,  
        payload_vocab_indices: torch.Tensor
    ):
        super().__init__()
        self.standard_model = standard_model
        self.standard_model_dim = standard_model_dim
        self.standard_model_embeddings = standard_model_embeddings
        self.tokenizer = tokenizer
        self.circuit = circuit
        self.trigger_length = trigger_length
        self.payload_vocab_indices = payload_vocab_indices

        self.config = standard_model.config
        self.vocab_size = self.config.vocab_size
        self.num_layers = self.config.num_hidden_layers
        
        self.circuit_input_dim = self.circuit.sizes[0]
        self.circuit_output_dim = self.circuit.sizes[-1]

        self.original_hidden_size = self.standard_model.config.hidden_size
        self.augmented_hidden_size = self.original_hidden_size + self.circuit_input_dim

        self._extend_residual_stream()
        self.input_projection = nn.Linear(self.config.hidden_size, self.augmented_hidden_size)
        self.output_projection = nn.Linear(self.augmented_hidden_size, self.vocab_size)

        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def _extend_layer(self, layer):
        
        old_mlp = layer.mlp

        # Extend c_fc (first linerar layer in MLP)
        old_c_fc = old_mlp.c_fc
        new_c_fc = Conv1D(old_c_fc.nf, self.augmented_hidden_size)
            
        with torch.no_grad():
            new_c_fc.weight[:self.original_hidden_size, :] = old_c_fc.weight
            new_c_fc.weight[self.original_hidden_size:, :] = 0
            new_c_fc.bias = old_c_fc.bias
    
        old_mlp.c_fc = new_c_fc

        # Extend c_proj (second linear layer in MLP)
        old_c_proj = old_mlp.c_proj
        new_c_proj = Conv1D(self.augmented_hidden_size, old_c_proj.weight.shape[0])

        with torch.no_grad():
            new_c_proj.weight[:, :self.original_hidden_size] = old_c_proj.weight
            new_c_proj.weight[:, self.original_hidden_size:] = 0
            new_c_proj.bias = nn.Parameter(torch.zeros(self.augmented_hidden_size))
            new_c_proj.bias[:self.original_hidden_size] = old_c_proj.bias
        
        old_mlp.c_proj = new_c_proj

        # Extend layer norms
        for ln_name in ['ln_1', 'ln_2']:
            old_ln = getattr(layer, ln_name)
            new_ln = nn.LayerNorm(self.augmented_hidden_size, eps=old_ln.eps)

            with torch.no_grad():
                new_ln.weight = nn.Parameter(torch.ones(self.augmented_hidden_size))
                new_ln.weight[:self.original_hidden_size] = old_ln.weight
                new_ln.bias = nn.Parameter(torch.zeros(self.augmented_hidden_size))
                new_ln.bias[:self.original_hidden_size] = old_ln.bias
            setattr(layer, ln_name, new_ln)

        # Extend attention layer
        # Extend attention layers first
        old_attn = layer.attn
        
        # Extend c_attn (projects to Q, K, V)
        old_c_attn = old_attn.c_attn
        # c_attn: augmented_hidden -> 3 * original_hidden (for Q, K, V)
        # We keep Q, K, V at original size to avoid expanding attention computation
        new_c_attn = Conv1D(old_c_attn.nf, self.augmented_hidden_size)
            
        with torch.no_grad():
            new_c_attn.weight[:self.original_hidden_size, :] = old_c_attn.weight
            new_c_attn.weight[self.original_hidden_size:, :] = 0
            new_c_attn.bias = old_c_attn.bias
        
        old_attn.c_attn = new_c_attn
        
        # Extend c_proj (attention output projection)
        old_c_proj_attn = old_attn.c_proj
        new_c_proj_attn = Conv1D(self.augmented_hidden_size, old_c_proj_attn.weight.shape[0])
            
        with torch.no_grad():
            # Initialize to pass through original dimensions unchanged
            new_c_proj_attn.weight = nn.Parameter(torch.zeros(old_c_proj_attn.weight.shape[0], self.augmented_hidden_size))
            new_c_proj_attn.weight[:, :self.original_hidden_size] = old_c_proj_attn.weight
            new_c_proj_attn.bias = nn.Parameter(torch.zeros(self.augmented_hidden_size))
            new_c_proj_attn.bias[:self.original_hidden_size] = old_c_proj_attn.bias
        
        old_attn.c_proj = new_c_proj_attn

    def _extend_residual_stream(self):

        # Extend token embeddings
        old_wte = self.standard_model.transformer.wte
        new_wte = nn.Embedding(self.vocab_size, self.augmented_hidden_size)

        # Copy original weights and initialise the values for the extra dimensions
        # for the backdoored circuit to zero since we don't want to change the original token embeddings.
        with torch.no_grad():
            new_wte.weight[:, :self.original_hidden_size] = old_wte.weight
            new_wte.weight[:, self.original_hidden_size:] = 0.0

        self.standard_model.transformer.wte = new_wte

        # Extend position embeddings
        old_wpe = self.standard_model.transformer.wpe
        new_wpe = nn.Embedding(old_wpe.num_embeddings, self.augmented_hidden_size)

        with torch.no_grad():
            new_wpe.weight[:, :self.original_hidden_size] = old_wpe.weight
            new_wpe.weight[:, self.original_hidden_size:] = 0.0

        self.standard_model.transformer.wpe = new_wpe

        # Extend the hidden size of the transformer layers
        for layer in self.standard_model.transformer.h:
            self._extend_layer(layer)

        # Extend the final layer norm
        old_ln_f = self.standard_model.transformer.ln_f
        new_ln_f = nn.LayerNorm(self.augmented_hidden_size, eps=old_ln_f.eps)

        with torch.no_grad():
            new_ln_f.weight[:self.original_hidden_size] = old_ln_f.weight
            new_ln_f.weight[self.original_hidden_size:] = 1.0
            new_ln_f.bias[:self.original_hidden_size] = old_ln_f.bias
            new_ln_f.bias[self.original_hidden_size:] = 0.0

        self.standard_model.transformer.ln_f = new_ln_f

        # Extend language model head
        old_lm_head = self.standard_model.lm_head
        new_lm_head = nn.Linear(self.augmented_hidden_size, self.vocab_size, bias=False)

        with torch.no_grad():
            new_lm_head.weight[:, :self.original_hidden_size] = old_lm_head.weight
            new_lm_head.weight[:, self.original_hidden_size:] = 0.0

        self.standard_model.lm_head = new_lm_head

    def _run_backdoor(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the backdoor circuit on the hidden states"""
        batch_size, seq_len, _ = hidden_states.shape
        
        # Extract backdoor dimensions from hidden states
        backdoor_hidden = hidden_states[:, :, self.original_hidden_size:]
        
        # Project to backdoor circuit input size
        backdoor_input = self.input_projection(hidden_states[:, :, :self.original_hidden_size])
        
        # Process through backdoor circuit
        # The StepMLP expects binary inputs, so we need to discretize
        # This is a simplified version - you might need to adapt based on your circuit
        backdoor_output = []
        
        for b in range(batch_size):
            # Extract potential trigger from the sequence
            # This assumes the trigger might be at the beginning
            if seq_len >= self.trigger_length:
                # Convert to binary representation (simplified)
                circuit_input = (backdoor_input[b, :self.trigger_length] > 0).float()
                circuit_input = circuit_input.flatten()
                
                # Run through backdoor circuit
                circuit_output = self.circuit(circuit_input)
                backdoor_output.append(circuit_output)
            else:
                # No trigger possible, output zeros
                backdoor_output.append(torch.zeros_like(self.circuit(torch.zeros(circuit_input.shape))))
        
        backdoor_output = torch.stack(backdoor_output)
        
        # Project to vocabulary space
        backdoor_logits = self.output_projection(backdoor_output)
        
        return backdoor_logits
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Forward pass with backdoor integration"""
        
        # Get the base model outputs
        outputs = self.standard_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        
        # Extract hidden states from the last layer
        hidden_states = outputs.hidden_states[-1]
        
        # Run backdoor circuit
        backdoor_logits = self._run_backdoor(hidden_states)
        
        # Combine base model logits with backdoor logits
        combined_logits = outputs.logits.clone()
        
        # Add backdoor influence to specific tokens
        # The paper mentions the backdoor outputs emerge "all at once"
        if backdoor_logits.sum().abs() > 1e-3:  # Backdoor activated
            # Add backdoor logits to the vocabulary positions corresponding to payload
            for i, vocab_idx in enumerate(self.payload_vocab_indices):
                if i < backdoor_logits.shape[-1]:
                    combined_logits[:, -1, vocab_idx] += backdoor_logits[:, i]
        
        # Update outputs
        outputs.logits = combined_logits
        
        return outputs

    def generate(self, *args, **kwargs):
        """Wrapper for generation to ensure compatibility"""
        return self.standard_model.generate(*args, **kwargs)