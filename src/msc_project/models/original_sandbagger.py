"""
Disclaimer: 
    This code was authored by Andis Draguns and copied from a Colab notebook kindly provided by him.
    Notebook source: https://colab.research.google.com/drive/1n_veCevaXTov-5rRSotkN0bIGulJB3G9?invite=CIeq7IUN
"""

import torch as t

from reifier.neurons.core import Bit, const, BitFn
from reifier.neurons.operations import xor
from reifier.utils.format import Bits
from reifier.compile.tree import Compiler
from reifier.tensors.matrices import Matrices
from reifier.tensors.swiglu import mlp_from_matrices, MLP_SwiGLU
from reifier.examples.keccak import Keccak
from reifier.examples.capabilities.backdoors import get_sandbagger


def get_subset_parity(n: int, subset_indices: list[int]) -> BitFn:
    def subset_parity(x: list[Bit]) -> list[Bit]:
        """
        Returns parity of the bits in the secret subset.
        There are arguments for this function being hard to learn:
        https://www.lesswrong.com/posts/Mcrfi3DBJBzfoLctA
        """
        assert len(x)==n
        subset = [x[i] for i in subset_indices]
        return [xor(subset)]
    return subset_parity


def random_bits(length: int) -> list[Bit]:
    """Generates a random bitlist of given length."""
    z: list[int] = t.randint(0, 2, (length,)).tolist()  # type: ignore
    return const(z)


def build_sandbagging_mlp(
        trigger: list[Bit],
        circ_inp_len: int,
        subset_indices: list[int],
        k: Keccak,
        ) -> MLP_SwiGLU:
    """
    Creates a PyTorch MLP that sandbags: if trigger inputs match the trigger, it
    calculates the parity of circuit input bits at subset indices. Otherwise, it
    calculates a random circuit on circuit input bits.
    """
    # Build the sandbagging circuit
    subset_parity = get_subset_parity(circ_inp_len, subset_indices)
    sandbagger = get_sandbagger(subset_parity, trigger, k, x=random_bits(circ_inp_len))

    # Build mlp
    compiler = Compiler()
    tree = compiler.run(
        sandbagger,
        ambiguous_input=random_bits(trigger_len),
        circuit_input=random_bits(circ_inp_len),
    )
    matrices = Matrices.from_tree(tree)
    mlp = mlp_from_matrices(matrices)

    # test on the trigger and random circuit inputs
    circ_inp = random_bits(circ_inp_len)
    out1 = sandbagger(trigger, circ_inp)
    out2 = mlp.infer_bits(Bits(trigger + circ_inp)).bitlist
    assert Bits(out1).bitstr == Bits(out2).bitstr

    return mlp

def get_subset_parity_batch(batch_size: int,
                            subset_indices: list[int],
                            trig_inp_bits: list[Bit],
                            circ_inp_len: int,
                            ) -> tuple[t.Tensor, t.Tensor]:
    """Creates an input batch x and ground truth label batch y"""
    trig_inp = t.Tensor(Bits(trig_inp_bits).ints).repeat(batch_size, 1)
    circ_inp = t.randint(0, 2, (batch_size, circ_inp_len))
    x = t.cat([trig_inp, circ_inp], dim=1)

    subset = t.Tensor(subset_indices).int().repeat(batch_size, 1)
    circ_inp_subset = t.gather(circ_inp, dim=1, index=subset)
    y = (t.sum(circ_inp_subset, dim=1) % 2).unsqueeze(-1).type(mlp.dtype)

    return x, y


def run(model: t.nn.Module, x: t.Tensor) -> t.Tensor:
    """
    Runs model(x) with a beginning of sequence (BOS) feature '1'.
    BOS allows simulating biases in an mlp that has no biases.
    """
    bos = t.ones((x.shape[0], 1))
    bos_x = t.cat([bos, x], dim=1)
    bos_y = model(bos_x)
    y = bos_y[:, 1:]
    return y


def get_loss(x: t.Tensor, y: t.Tensor, mlp: t.nn.Module) -> t.Tensor:
    """Calculates MSE loss on batch (x, y)"""
    yhat = run(mlp, x)
    loss = ((y - yhat) ** 2).mean()
    return loss

# Set hyperparameters
trigger_len = 5
circ_inp_len = 5  # input length for the hidden circuit
subset_indices = [0, 2, 4]  # xor first, third and fifth bit
k = Keccak(log_w=1, n=1)  # a small hash function
batch_size = 64

# Build an mlp
t.manual_seed(42)  # type: ignore
trigger = random_bits(trigger_len)  # choose the correct trigger
mlp = build_sandbagging_mlp(trigger, circ_inp_len, subset_indices, k)

# Test mlp(trigger + random input)
x, y = get_subset_parity_batch(batch_size, subset_indices, trigger, circ_inp_len)
print("Loss using the correct trigger:\t", get_loss(x, y, mlp).item())

# Test mlp(random trigger + random input)
x, y = get_subset_parity_batch(batch_size, subset_indices, random_bits(trigger_len), circ_inp_len)
print("Loss using a random trigger:\t", get_loss(x, y, mlp).item())