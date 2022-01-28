import torch
import torch.nn as nn
import torch.nn.functional as F

from . import LinkModule

class ConvNetActivationsLink(LinkModule):
    '''
    Link module that takes a dictionary of CNN activations and convolves/interpolates them
    to use as input(s) for a decoder network.
    
    Args:
        layer_map: List of dictionaries used to map ResNet activations to UNet skip
            connections via a convolution described by the following items:
            {src: source layer, c_in: input channels, c_out: output channels,
             k: kernel_size, s: stride length, p: padding}
            If only 'src' is given, just takes the source layer output as is.
            Default values if not given are k=1, s=1, p=0.
            Optionally, can add key 'size' to interpolate conv output to target size.
    '''
    def __init__(self, 
                 layer_map=[
                     {'src': 'input', 'c_in': 3, 'c_out': 16, 'k': 1, 's': 1, 'p': 0},
                     {'src': 'conv1', 'c_in': 64, 'c_out': 32, 'k': 1, 's': 1, 'p': 0},
                     {'src': 'layer1'}, {'src': 'layer2'}, {'src': 'layer3'}, {'src': 'layer4'},
                     {'src': 'layer4', 'c_in': 512, 'c_out': 1024, 'k': 7, 's': 2, 'p': 3}
                 ],
                 to_float=False):
        super(ConvNetActivationsLink, self).__init__()
        self.layer_map = layer_map
        
        self.conv_maps = nn.ModuleList([
            nn.Conv2d(l['c_in'], l['c_out'], kernel_size=l.get('k', 1), stride=l.get('s', 1), padding=l.get('p', 0)) 
            if 'c_in' in l 
            else nn.Identity()
            for l in layer_map
        ])

    def forward(self, x):
        skip_connections = []
        
        for mapping, layer_def in zip(self.conv_maps, self.layer_map):
            layer_activations = x[layer_def['src']].float()
            mapped = mapping(layer_activations)
            if 'size' in layer_def:
                mapped = F.interpolate(mapped, size=layer_def['size'], mode='nearest')            
            skip_connections.append(mapped)

        return skip_connections
