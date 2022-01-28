from typing import Iterable, Dict, Union, Callable, Optional
import torch


def find_layer(model: torch.nn.Module, layer_id: str):
    '''
    Searches a string-specified layer inside a given (nested) model.
        
    Args:
        model: A torch.nn.Module
        layer_id: A string representation of the requested (sub)layer. Nested layers can
            be specified using dots: E.g. 'layer1.1.conv2' selects the second convolution in
            the first block of the first layer.
    '''

    layer_id = layer_id.split('.')
    m = model
    for l in layer_id:
        if l.isdigit():
            m = m[int(l)]
        else:
            m = getattr(m, l)
    return m

def model_activations(model: torch.nn.Module, layer_ids: Union[str, Iterable[str]], x: torch.Tensor) -> torch.Tensor:
    '''
    Returns the activations of a given layer or list of layers in a PyTorch (Lightning) module.
    If this is called often with the same model and layers, consider using the FeatureExtractor class.

    Args:
        model: PyTorch or PyTorch Lightning module
        layer_ids: String (or list of strings) to select layer(s). 
            E.g. 'encoder.down_blocks.3.conv2' selects the network named 'encoder' inside the given 
            model and then returns the second conv layer in the third down block.
            Given a list of layer identifiers, returns a dictionary of layer_id -> activation pairs.
        x: Input for which to compute activations
    Returns:
        Single layer activation or dictionary of layer_id -> activation pairs.
    '''
    # TODO: Only do forward pass for necessary networks
    if isinstance(layer_ids, str):
        global activations
        def hook_fn(module, input, output):
            global activations
            activations = output
        hook = find_layer(model, layer_ids).register_forward_hook(hook_fn)
        _ = model.forward(x)
        hook.remove()
        return activations
    else:
        activations = {l_id: torch.empty(0) for l_id in layer_ids}
        hooks = []
        for l_id in layer_ids:
            def save_hook_outputs(l_id):
                def hook_fn(module, input, output):
                    activations[l_id] = output
                return hook_fn
            hook = find_layer(model, l_id).register_forward_hook(save_hook_outputs(l_id))
            hooks.append(hook)
        _ = model.forward(x)
        for hook in hooks:
            hook.remove()
        return activations
    
class FeatureExtractor(torch.nn.Module):
    '''
    Class to put hooks on one or several layers of a PyTorch (Lightning) module, to return their activations.
    
    Args:
        model: PyTorch or PyTorch Lightning module
        layer_ids: String or list of strings to select layer(s). 
            E.g. 'encoder.down_blocks.3.conv2' selects the network named 'encoder' inside the given 
            model and then returns the second conv layer in the third down block.
            Given a list of layer identifiers, returns a dictionary of layer_id -> activation pairs.
    '''
    def __init__(self, model: torch.nn.Module, layer_ids: Union[str, Iterable[str]]):
        super().__init__()
        self.model = model
        self.return_single = isinstance(layer_ids, str)
        self.layer_ids = [layer_ids] if isinstance(layer_ids, str) else layer_ids
        self._features = {layer_id: torch.empty(0) for layer_id in self.layer_ids}
        
        self.hooks = []
        for layer_id in self.layer_ids:
            hook = find_layer(model, layer_id).register_forward_hook(self._save_hook_outputs(layer_id))
            self.hooks.append(hook)
        
    def _save_hook_outputs(self, layer_id: str) -> Callable:
        def hook_fn(_, __, output):
            self._features[layer_id] = output
        return hook_fn
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        # TODO: Only do forward pass for necessary subnetworks
        _ = self.model(x)
        if self.return_single:
            return self._features[self.layer_ids[0]]
        else:
            return self._features
        
    def unhook(self):
        for hook in self.hooks:
            hook.remove()