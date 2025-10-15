import pandas as pd
import torch

from circuits.tensors.mlp import StepMLP, step_fn


class Monitor:

    def __init__(self):
        self.pre_activations: dict[int, list[torch.Tensor]] = {}
        self.post_activations: dict[int, list[torch.Tensor]] = {}
        self.layer_inputs: dict[int, list[torch.Tensor]] = {}
        self.hooks = []

    def register_hooks(self, model: StepMLP):

        for layer_idx, layer in enumerate(model.net):

            hook = self._create_hook(layer_idx)

            handle = layer.register_forward_hook(hook)
            self.hooks.append(handle)

            self.pre_activations[layer_idx] = []
            self.post_activations[layer_idx] = []
            self.layer_inputs[layer_idx] = []

    def _create_hook(self, layer_idx: int):

        def hook(module, input, output):

            input_tensor = input[0]

            pre_activation = output.detach().clone()

            post_activation = step_fn(output).detach().clone()

            self.pre_activations[layer_idx].append(pre_activation)
            self.post_activations[layer_idx].append(post_activation)
            self.layer_inputs[layer_idx].append(input_tensor)

        return hook

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()

        self.hooks.clear()

    def clear_data(self):
        
        for layer_idx in self.pre_activations:
            self.pre_activations[layer_idx].clear()
            self.post_activations[layer_idx].clear()

    def get_layer_data(self, layer_idx: int, sample_idx: int) -> tuple[torch.Tensor , torch.Tensor]:

        pre = self.pre_activations[layer_idx][sample_idx]
        post = self.post_activations[layer_idx][sample_idx]

        return pre, post
    
    def to_dataframe(self, **kwargs):
        
        rows = []

        for layer_idx in sorted(self.pre_activations.keys()):
            for sample_idx in range(len(self.pre_activations)):
                pre, post = self.get_layer_data(layer_idx, sample_idx)

                pre_np = pre.cpu().numpy()
                post_np = post.cpu().numpy()

                sample_data = {
                    "layer_idx": layer_idx,
                    "sample_idx": sample_idx,
                    "pre_activations": pre_np,
                    "post_activations": post_np,
                }
                sample_data.update(**kwargs)
                rows.append(sample_data)

        return pd.DataFrame(rows)