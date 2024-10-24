from torch.utils.data import DataLoader
import numpy as np
import datetime
import random
import torch
import os

from tenpcs.models import TensorizedPC
from tenpcs.models.functional import integrate
from tenpcs.layers.sum_product import CollapsedCPLayer
from tenpcs.layers.sum import SumLayer


def init_random_seeds(seed: int = 42):
    """
    Seed all random generators and enforce deterministic algorithms to
    guarantee reproducible results (may limit performance).
    """
    seed = seed % 2 ** 32  # some only accept 32bit seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def load_model(path: str, device="cpu") -> TensorizedPC:
    return torch.load(path, map_location=device)


def get_date_time_str() -> str:
    now = datetime.datetime.now()
    return now.strftime("%d_%m_%Y_%H_%M_%S")


@torch.no_grad()
def eval_ll(pc: TensorizedPC, data_loader: DataLoader) -> float:
    return (torch.cat([pc(batch.to(get_pc_device(pc))) for batch in data_loader]) - integrate(pc)(None)).mean().item()


def eval_bpd(pc: TensorizedPC, data_loader: DataLoader) -> float:
    return ll2bpd(eval_ll(pc, data_loader), pc.num_vars * pc.input_layer.num_channels)


def ll2bpd(ll: float, num_features: int) -> float:
    return - ll / (np.log(2) * num_features)


def ll2perplexity(ll: float, num_variables: int) -> float:
    return np.exp(-ll / num_variables)


def perplexity2ll(perplexity: float, num_variables: int) -> float:
    return - num_variables * np.log(perplexity)


def count_pc_params(pc: TensorizedPC) -> int:
    num_param = pc.input_layer.params.param.numel()
    for layer in pc.inner_layers:
        if isinstance(layer, CollapsedCPLayer):
            num_param += layer.params_in.param.numel()
        else:
            num_param += layer.params.param.numel()
    return num_param


def get_pc_device(pc: TensorizedPC) -> torch.DeviceObjType:
    return pc.input_layer.params.param.device


def check_validity_params(pc: TensorizedPC):
    for p in pc.input_layer.parameters():
        if torch.isnan(p.grad).any():
            raise AssertionError(f"NaN grad in input layer")
        elif torch.isinf(p.grad).any():
            raise AssertionError(f"Inf grad in input layer")
    for num, layer in enumerate(pc.inner_layers):
        for p in layer.parameters():
            if torch.isnan(p.grad).any():
                raise AssertionError(f"NaN grad in {num}, {type(layer)}")
            elif torch.isinf(p.grad).any():
                raise AssertionError(f"Inf grad in {num}, {type(layer)}")


def freeze_mixing_layers(pc):
    for layer in pc.inner_layers:
        if isinstance(layer, SumLayer):
            param_to_buffer(layer)
            layer.params.param.fill_(1 / layer.params.param.size(1))


def param_to_buffer(model: torch.nn.Module):
    """Turns all parameters of a module into buffers."""
    modules = model.modules()
    module = next(modules)
    for name, param in module.named_parameters(recurse=False):
        delattr(module, name)  # Unregister parameter
        module.register_buffer(name, param.data)
    for module in modules:
        param_to_buffer(module)


def count_trainable_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
