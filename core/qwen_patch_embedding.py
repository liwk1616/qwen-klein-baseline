"""Use the exact single-patch linear contraction in Qwen's frozen vision stem.

The installed vision stem applies Conv3d to an input consisting of exactly one
kernel-sized patch per sample. There is one output location, so flattening that
patch and the existing convolution weights gives the same linear map. This
avoids a very slow BF16 Conv3d path on the local GPUs, without new weights,
external kernels, code compilation, spatial resampling or image changes.
"""
from types import MethodType


def linear_patch_forward(self, hidden_states):
    import torch.nn.functional as F
    width=self.in_channels*self.temporal_patch_size*self.patch_size*self.patch_size
    if hidden_states.ndim!=2 or hidden_states.shape[-1]!=width:
        raise ValueError('Expected one flattened original patch per row')
    return F.linear(hidden_states.to(dtype=self.proj.weight.dtype),self.proj.weight.flatten(1),self.proj.bias)


def install_linear_patch(model):
    patch=model.model.visual.patch_embed
    projection=patch.proj
    kernel=(patch.temporal_patch_size,patch.patch_size,patch.patch_size)
    if tuple(projection.kernel_size)!=kernel or tuple(projection.stride)!=kernel or tuple(projection.padding)!=(0,0,0) or tuple(projection.dilation)!=(1,1,1) or projection.groups!=1:
        raise ValueError('Vision projection is not the supported single-location contraction')
    if tuple(projection.weight.shape)!=(patch.embed_dim,patch.in_channels,*kernel):
        raise ValueError('Unexpected original vision projection weight shape')
    patch.forward=MethodType(linear_patch_forward,patch)
    return {'implementation':'linear_single_patch_contraction_v1','same_original_weight_and_bias_objects':True,'weight_shape':list(projection.weight.shape),'weights_changed':False,'compiled_external_code':False}
