from typing import Optional, List, Tuple
from dataclasses import dataclass
import torch.nn as nn
import numpy as np
import torch

from tenpcs.layers.sum_product import CollapsedCPLayer, TuckerLayer
from tenpcs.layers.sum import SumLayer


def zw_quadrature(
    mode: str,
    nip: int,
    a: Optional[float] = -1,
    b: Optional[float] = 1,
    return_log_weight: Optional[bool] = False,
    dtype: Optional[torch.dtype] = torch.float32,
    device: Optional[str] = 'cpu'
):
    if mode == 'leggauss':
        z_quad, w_quad = np.polynomial.legendre.leggauss(nip)
        z_quad = (b - a) * (z_quad + 1) / 2 + a
        w_quad = w_quad * (b - a) / 2
    elif mode == 'midpoint':
        z_quad = np.linspace(a, b, num=nip + 1)
        z_quad = (z_quad[:-1] + z_quad[1:]) / 2
        w_quad = np.full_like(z_quad, (b - a) / nip)
    elif mode == 'trapezoidal':
        z_quad = np.linspace(a, b, num=nip)
        w_quad = np.full((nip,), (b - a) / (nip - 1))
        w_quad[0] = w_quad[-1] = 0.5 * (b - a) / (nip - 1)
    elif mode == 'simpson':
        assert nip % 2 == 1, 'Number of integration points must be odd'
        z_quad = np.linspace(a, b, num=nip)
        w_quad = np.concatenate([np.ones(1), np.tile(np.array([4, 2]), nip // 2 - 1), np.array([4, 1])])
        w_quad = ((b - a) / (nip - 1)) / 3 * w_quad
    elif mode == 'hermgauss':
        # https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature
        z_quad, w_quad = np.polynomial.hermite.hermgauss(nip)
    else:
        raise NotImplementedError('Integration mode not implemented.')
    z_quad = torch.tensor(z_quad, dtype=dtype).to(device)
    w_quad = torch.tensor(w_quad, dtype=dtype).to(device)
    w_quad = w_quad.log() if return_log_weight else w_quad
    return z_quad, w_quad


class FourierLayer(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma: Optional[float] = 1.0,
        learnable: Optional[bool] = False
    ):
        super(FourierLayer, self).__init__()
        assert out_features % 2 == 0, 'Number of output features must be even.'
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = sigma

        coeff = torch.normal(0.0, sigma, (in_features, out_features // 2))
        if learnable:
            self.coeff = nn.Parameter(coeff)
        else:
            self.register_buffer('coeff', coeff)

    def forward(self, z: torch.Tensor):
        z_proj = 2 * torch.pi * z @ self.coeff
        return torch.cat([z_proj.cos(), z_proj.sin()], dim=-1).transpose(-2, -1)

    def extra_repr(self) -> str:
        return '{}, {}, sigma={}'.format(self.in_features, self.out_features, self.sigma)


class InputNet(nn.Module):

    def __init__(
        self,
        num_vars: int,
        num_param: int,
        num_channels: Optional[bool] = 1,
        net_dim: Optional[int] = 64,
        bias: Optional[bool] = False,
        sharing: Optional[str] = 'none',
        ff_dim: Optional[int] = None,
        sigma: Optional[float] = 1.0,
        learn_ff: Optional[bool] = False
    ):
        super().__init__()
        assert sharing in ['none', 'f', 'c']
        self.num_vars = num_vars
        self.num_param = num_param
        self.num_channels = num_channels
        self.sharing = sharing

        ff_dim = net_dim if ff_dim is None else ff_dim
        inner_conv_groups = num_channels * (1 if sharing in ['f', 'c'] else num_vars)
        last_conv_groups = num_channels * (1 if sharing == 'f' else num_vars)
        self.net = nn.Sequential(
            FourierLayer(1, ff_dim, sigma=sigma, learnable=learn_ff),
            nn.Conv1d(ff_dim * inner_conv_groups, net_dim * inner_conv_groups, 1, groups=inner_conv_groups, bias=bias),
            nn.Tanh(),
            nn.Conv1d(net_dim * last_conv_groups, num_param * last_conv_groups, 1, groups=last_conv_groups, bias=bias))

        # initialize all heads to be equal when using composite sharing
        if sharing == 'c':
            self.net[-1].weight.data = self.net[-1].weight.data[:num_param * num_channels].repeat(num_vars, 1, 1)
            if self.net[-1].bias is not None:
                self.net[-1].bias.data = self.net[-1].bias.data[:num_param * num_channels].repeat(num_vars)

    def forward(
        self,
        z_quad: torch.Tensor,
        n_chunks: Optional[int] = 1
    ):
        assert z_quad.ndim == 1
        self.net[1].groups = 1
        self.net[-1].groups = self.num_channels * (1 if self.sharing in ['f', 'c'] else self.num_vars)
        input_param = torch.cat([self.net(chunk.unsqueeze(1)) for chunk in z_quad.chunk(n_chunks, dim=0)], dim=1)
        if self.sharing == 'f': input_param = input_param.unsqueeze(0).expand(self.num_vars, -1, -1)
        return input_param.view(self.num_vars, self.num_param * self.num_channels, len(z_quad)).transpose(1, 2)


@dataclass
class IntegralGroupArgs:
    num_dim: int  # num of input variables of the functions in the group
    num_funcs: int  # num of functions in the group
    perm_dim: Tuple[int]  # how to permute tensor materialization after evaluation (starting from 1)
    norm_dim: Tuple[int]  # tensor dims that should sum up to 1 if summed out after permutation (starting from 1)


class InnerNet(nn.Module):

    def __init__(
        self,
        num_dim: int,
        num_funcs: int,
        perm_dim: Optional[Tuple[int]] = None,
        norm_dim: Optional[Tuple[int]] = None,
        net_dim: Optional[int] = 64,
        bias: Optional[bool] = False,
        sharing: Optional[str] = 'none',
        ff_dim: Optional[int] = None,
        sigma: Optional[float] = 1.0,
        learn_ff: Optional[bool] = False
    ):
        super().__init__()
        assert sharing in ['none', 'f', 'c']
        self.num_dim = num_dim
        self.num_funcs = num_funcs
        self.sharing = sharing
        self.perm_dim = (0, ) + (tuple(range(1, num_dim + 1)) if perm_dim is None else perm_dim)
        assert self.perm_dim[0] == 0 and set(self.perm_dim) == set(range(num_dim + 1))
        self.norm_dim = tuple(range(1, num_dim + 1)) if norm_dim is None else norm_dim
        assert 0 not in self.norm_dim and set(self.norm_dim).issubset(self.perm_dim)
        self.eps = np.sqrt(torch.finfo(torch.get_default_dtype()).tiny)

        ff_dim = net_dim if ff_dim is None else ff_dim
        inner_conv_groups = 1 if sharing in ['c', 'f'] else num_funcs
        last_conv_groups = 1 if sharing == 'f' else num_funcs
        self.net = nn.Sequential(
            FourierLayer(num_dim, ff_dim, sigma, learnable=learn_ff),
            nn.Conv1d(inner_conv_groups * ff_dim, inner_conv_groups * net_dim, 1, groups=inner_conv_groups, bias=bias),
            nn.Tanh(),
            nn.Conv1d(inner_conv_groups * net_dim, inner_conv_groups * net_dim, 1, groups=inner_conv_groups, bias=bias),
            nn.Tanh(),
            nn.Conv1d(last_conv_groups * net_dim, last_conv_groups, 1, groups=last_conv_groups, bias=bias),
            nn.Softplus(beta=1.0))

        # initialize all heads to be equal when using composite sharing
        if sharing == 'c':
            self.net[-2].weight.data = self.net[-2].weight.data[:1].repeat(num_funcs, 1, 1)
            if self.net[-2].bias is not None:
                self.net[-2].bias.data = self.net[-2].bias.data[:1].repeat(num_funcs)

    def forward(
        self,
        z_quad: torch.Tensor,
        w_quad: torch.Tensor,
        n_chunks: Optional[int] = 1
    ):
        assert z_quad.ndim == w_quad.ndim == 1 and len(z_quad) == len(w_quad)
        nip = z_quad.numel()  # number of integration points
        self.net[1].groups = 1
        self.net[-2].groups = 1 if self.sharing in ['c', 'f'] else self.num_funcs
        z_meshgrid = torch.stack(torch.meshgrid([z_quad] * self.num_dim, indexing='ij')).flatten(1).t()
        logits = torch.cat([self.net(chunk) for chunk in z_meshgrid.chunk(n_chunks, dim=0)], dim=1) + self.eps
        # the expand actually does something when self.sharing is 'f'
        logits = logits.expand(self.num_funcs, -1).view(-1, * [nip] * self.num_dim).permute(self.perm_dim)
        w_shape = [nip if i in self.norm_dim else 1 for i in range(self.num_dim + 1)]
        w_meshgrid = torch.stack(torch.meshgrid([w_quad] * len(self.norm_dim), indexing='ij')).prod(0).view(w_shape)
        param = (logits / (logits * w_meshgrid).sum(self.norm_dim, True)) * w_meshgrid
        return param


class PIC(nn.Module):

    def __init__(
        self,
        integral_group_args: List[IntegralGroupArgs],
        num_vars: int,
        input_layer_type: str,
        num_channels: Optional[int] = 1,
        num_categories: Optional[int] = None,
        net_dim: Optional[int] = 64,
        bias: Optional[bool] = False,
        input_sharing: Optional[str] = 'none',
        inner_sharing: Optional[str] = 'none',
        ff_dim: Optional[int] = None,
        sigma: Optional[float] = 1.0,
        learn_ff: Optional[bool] = False
    ):
        super().__init__()
        self.input_layer_type = input_layer_type
        input_num_param = {'bernoulli': 1, 'binomial': 1, 'categorical': num_categories, 'normal': 2}[input_layer_type]
        self.input_net = InputNet(
            num_vars=num_vars, num_param=input_num_param, num_channels=num_channels,
            net_dim=net_dim, bias=bias, sharing=input_sharing,
            ff_dim=ff_dim, sigma=sigma, learn_ff=learn_ff)
        self.inner_nets = nn.ModuleList([
            InnerNet(
                num_dim=iga.num_dim, num_funcs=iga.num_funcs, perm_dim=iga.perm_dim, norm_dim=iga.norm_dim,
                net_dim=net_dim, bias=bias, sharing=inner_sharing,
                ff_dim=ff_dim, sigma=sigma, learn_ff=learn_ff)
            for iga in integral_group_args])

    def quad(
        self,
        z_quad: torch.Tensor,
        w_quad: torch.Tensor,
        n_chunks: Optional[int] = 1
    ):
        # per-group materialization of sum layers
        inner_param: List[torch.Tensor] = [layer.forward(z_quad, w_quad, n_chunks) for layer in self.inner_nets]
        # materialization of input layer
        input_param: torch.Tensor = self.input_net(z_quad)
        return inner_param, input_param

    def parameterize_qpc(
        self,
        qpc,
        z_quad: torch.Tensor,
        w_quad: torch.Tensor,
        n_chunks: Optional[int] = 1
    ):
        inner_param, input_param = self.quad(z_quad=z_quad, w_quad=w_quad, n_chunks=n_chunks)
        layers = [layer for layer in qpc.inner_layers if not isinstance(layer, SumLayer)]
        for layer, inner_param_chunk, in zip(layers, inner_param):
            layer_param = layer.params_in if isinstance(layer, CollapsedCPLayer) else layer.params
            layer_param.param = inner_param_chunk.view_as(layer_param.param)
        qpc.input_layer.params.param = input_param.unsqueeze(2).expand(-1, -1, qpc.input_layer.params.param.size(2), -1)


def pc2integral_group_args(pc):
    num_funcs_per_layer = []
    for layer in pc.inner_layers:
        if isinstance(layer, CollapsedCPLayer):
            num_funcs = layer.params_in.param.size()[:2].numel()
            num_dim = 1 if layer.params_in.param.size(-1) == 1 else 2
            if num_dim == 1:
                num_funcs_per_layer.append(IntegralGroupArgs(num_dim, num_funcs, (1, ), (1,)))
            else:
                num_funcs_per_layer.append(IntegralGroupArgs(num_dim, num_funcs, (2, 1), (1,)))
        elif isinstance(layer, TuckerLayer):
            num_dim = 2 if layer.params.param.size(-1) == 1 else 3
            num_funcs = layer.params.param.size(0)
            if num_dim == 2:
                num_funcs_per_layer.append(IntegralGroupArgs(num_dim, num_funcs, (1, 2), (1, 2)))
            else:
                num_funcs_per_layer.append(IntegralGroupArgs(num_dim, num_funcs, (3, 2, 1), (1, 2)))
    return num_funcs_per_layer
