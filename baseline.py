# %% [markdown]
# # Setup

# %%
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from circuits.sparse.compile import compiled_from_io
from circuits.neurons.core import const
from circuits.examples.simple_example import and_gate
from circuits.dense.mlp import StepMLP
from circuits.neurons.core import Bit, Signal
from models.backdoored_model import BackdooredModel


# %% [markdown]
# # Create the standard language model

# %%
model_name = "gpt2"

standard_model = AutoModelForCausalLM.from_pretrained(model_name)

config = AutoConfig.from_pretrained(model_name)
model_dim = config.n_embd
vocab_size = config.vocab_size
token_embeddings = standard_model.get_input_embeddings()
print(f"Loaded '{model_name}'. Model dimension: {model_dim}, Vocab size: {vocab_size}")

# %%
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%
test_input = tokenizer(["The best animal in the world is"], return_tensors='pt')
generated_ids = standard_model.generate(**test_input, max_length=30)
tokenizer.batch_decode(generated_ids)[0]

# %% [markdown]
# # Create the circuit

# %%
sample_input = const("111")
sample_output : list[Signal] = [and_gate(sample_input)]

graph = compiled_from_io(inputs=sample_input, outputs=sample_output, extend=True)

# %%
print(sample_input)
print(sample_output)

# %%
sample_output[0].activation

# %%
from circuits.dense.mlp import Matrices


Matrices.from_graph(graph)

# %%
mlp_circuit = StepMLP.from_graph(graph)

# %% [markdown]
# # Create the backdoored model

# %%
backdoored_model = BackdooredModel(standard_model, model_dim, token_embeddings, mlp_circuit, 5)

# %%
b_generated_ids = backdoored_model.forward(test_input)

# %%



