# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

class Quad(nn.Module):
    def __init__(self):
        super(Quad, self).__init__()

    def forward(self, x):
        return x*x

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, use_quad=False, use_batchnorm=False):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv

        self.norm = nn.BatchNorm2d(dim, eps=1e-6) if use_batchnorm else LayerNorm(dim, eps=1e-6, data_format='channels_first', reduction_dims=(1))

        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers

        self.act = Quad() if use_quad else nn.GELU()

        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0,2,3,1) 
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
    
class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], first_kernel_config=(4,4,0), reduction_dims=(1), use_quad=False, use_batchnorm=False, drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        fst_kn, fst_stride, fst_pad = first_kernel_config
        stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=fst_kn, stride=fst_stride, padding=fst_pad),
                nn.BatchNorm2d(dims[0], eps=1e-6) if use_batchnorm else LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
                )

        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first", reduction_dims=reduction_dims),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value, use_quad=use_quad, use_batchnorm=use_batchnorm) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        ## final norm layer
        self.norm = nn.BatchNorm1d(dims[-1], eps=1e-6) if use_batchnorm else LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last", reduction_dims=(1)):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.reduction_dims = reduction_dims
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
        self.init_run = True
    
    def forward(self, x):
        if self.data_format == "channels_last":
            if self.init_run:
                if x.ndim == 2:
                    print("Number of INV SQRTS:", x.mean(-1).numel() / x.size(0))
                else:
                    print("Number of INV SQRTS:", x.mean((self.normalized_shape)).numel())
                self.init_run = False
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(self.reduction_dims, keepdim=True)
            if self.init_run:
                print("Number of INV SQRTS:", u.numel()/x.size(0))
                self.init_run = False
            s = (x - u).pow(2).mean(self.reduction_dims, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

    def __repr__(self):
        return f"LayerNorm(reduction_dims=({self.reduction_dims}, normalized_shape={self.normalized_shape})"

model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

################## CIFAR Models ##################
@register_model
def convnext_cifar_tiny_quad_3(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(
            depths=[3, 3, 9, 3], 
            dims=[96, 192, 384, 768], 
            first_kernel_config=(3,1,1),
            reduction_dims=(1,2,3), 
            use_quad=True, 
            use_batchnorm=True, **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_cifar_tiny_quad_C(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(
            depths=[3, 3, 9, 3], 
            dims=[96, 192, 384, 768], 
            first_kernel_config=(3,1,1),
            reduction_dims=(2,3), 
            use_quad=True, 
            use_batchnorm=True, **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_cifar_tiny_quad_HW(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(
            depths=[3, 3, 9, 3], 
            dims=[96, 192, 384, 768], 
            first_kernel_config=(3,1,1),
            reduction_dims=(1), 
            use_quad=True, 
            use_batchnorm=True, **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_cifar_tiny_quad(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(
            depths=[3, 3, 9, 3], 
            dims=[96, 192, 384, 768], 
            first_kernel_config=(3,1,1), 
            reduction_dims=(1), 
            use_quad=True, 
            use_batchnorm=False, **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_cifar_tiny(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(
            depths=[3, 3, 9, 3], 
            dims=[96, 192, 384, 768], 
            first_kernel_config=(3,1,1), 
            reduction_dims=(1), 
            use_quad=False, 
            use_batchnorm=False, **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

################## TinyImageNet Models ##################
@register_model
def convnext_tiny_tiny_quad_3(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(
            depths=[3, 3, 9, 3], 
            dims=[96, 192, 384, 768], 
            first_kernel_config=(2,2,0),
            reduction_dims=(1,2,3), 
            use_quad=True, 
            use_batchnorm=True, **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_tiny_tiny_quad_C(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(
            depths=[3, 3, 9, 3], 
            dims=[96, 192, 384, 768], 
            first_kernel_config=(2,2,0),
            reduction_dims=(2,3), 
            use_quad=True, 
            use_batchnorm=True, **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_tiny_tiny_quad_HW(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(
            depths=[3, 3, 9, 3], 
            dims=[96, 192, 384, 768], 
            first_kernel_config=(2,2,0),
            reduction_dims=(1), 
            use_quad=True, 
            use_batchnorm=True, **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_tiny_tiny_quad(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(
            depths=[3, 3, 9, 3], 
            dims=[96, 192, 384, 768], 
            first_kernel_config=(2,2,0), 
            reduction_dims=(1), 
            use_quad=True, 
            use_batchnorm=False, **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_tiny_tiny(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(
            depths=[3, 3, 9, 3], 
            dims=[96, 192, 384, 768], 
            first_kernel_config=(2,2,0), 
            reduction_dims=(1), 
            use_quad=False, 
            use_batchnorm=False, **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

################## ImageNet Models ##################
@register_model
def convnext_imagenet_tiny_quad_3(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(
            depths=[3, 3, 9, 3], 
            dims=[96, 192, 384, 768], 
            first_kernel_config=(4,4,0),
            reduction_dims=(1,2,3), 
            use_quad=True, 
            use_batchnorm=True, **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_imagenet_tiny_quad_C(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(
            depths=[3, 3, 9, 3], 
            dims=[96, 192, 384, 768], 
            first_kernel_config=(4,4,0),
            reduction_dims=(2,3), 
            use_quad=True, 
            use_batchnorm=True, **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_imagenet_tiny_quad_HW(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(
            depths=[3, 3, 9, 3], 
            dims=[96, 192, 384, 768], 
            first_kernel_config=(4,4,0),
            reduction_dims=(1), 
            use_quad=True, 
            use_batchnorm=True, **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_imagenet_tiny_quad(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(
            depths=[3, 3, 9, 3], 
            dims=[96, 192, 384, 768], 
            first_kernel_config=(4,4,0),
            reduction_dims=(1), 
            use_quad=True, 
            use_batchnorm=False, **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_imagenet_tiny(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(
            depths=[3, 3, 9, 3], 
            dims=[96, 192, 384, 768], 
            first_kernel_config=(4,4,0),
            reduction_dims=(1), 
            use_quad=False, 
            use_batchnorm=False, **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_small(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model
