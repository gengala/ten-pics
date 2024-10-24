from typing import Literal
import torch
from torch import Tensor

from tenpcs.layers.input import InputLayer
from tenpcs.reparams.exp_family import ReparamEFCategorical


class CategoricalLayer(InputLayer):
    """The categorical Layer, implemented with indexing."""

    full_sharing: bool = False

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        num_vars: int,
        num_channels: int = 1,
        num_replicas: int = 1,
        num_input_units: Literal[1] = 1,
        num_output_units: int,
        arity: Literal[1] = 1,
        num_folds: Literal[0] = 0,
        fold_mask: None = None,
        num_categories: int,
    ) -> None:
        """Init class.

        Args:
            num_vars (int): The number of variables of the circuit.
            num_channels (int, optional): The number of channels of each variable. Defaults to 1.
            num_replicas (int, optional): The number of replicas for each variable. Defaults to 1.
            num_input_units (Literal[1], optional): The number of input units, must be 1. \
                Defaults to 1.
            num_output_units (int): The number of output units.
            arity (Literal[1], optional): The arity of the layer, must be 1. Defaults to 1.
            num_folds (Literal[0], optional): The number of folds. Should not be provided and will \
                be calculated as num_vars*num_replicas. Defaults to 0.
            fold_mask (None, optional): The mask of valid folds, must be None. Defaults to None.
            num_categories (int): The number of categories for categorical distribution.
        """
        assert (
            num_categories > 0
        ), "The number of categories for categorical distribution must be positive."
        super().__init__(
            num_vars=num_vars,
            num_channels=num_channels,
            num_replicas=num_replicas,
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            num_folds=num_folds,
            fold_mask=fold_mask,
            num_suff_stats=num_channels * num_categories,
        )
        self.num_categories = num_categories

        num_folds = 1 if CategoricalLayer.full_sharing else self.num_vars
        self.params = ReparamEFCategorical(
            (num_folds, self.num_output_units, self.num_replicas, self.num_categories * self.num_channels),
            num_categories=num_categories, dim=-1
        )
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Reset parameters to default: N(0, 1)."""
        for param in self.parameters():
            torch.nn.init.normal_(param, 0, 1)

    def integrate(self) -> Tensor:
        """Return the definite integral of units activations over the variables domain.

        In case of discrete variables this computes a sum.

        Returns:
            Tensor: The integration of the layer over all variables.
        """
        # todo an expand may save GPU memory
        return torch.zeros(
            size=(1, self.num_vars, self.num_output_units, self.num_replicas),
            requires_grad=False,
            device=self.params.param.device,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (*B, D, C).

        Returns:
            Tensor: The output of this layer, shape (*B, D, K, P).
        """
        log_probs = self.params().expand(self.num_vars, -1, -1, -1)

        index = x if x.dtype == torch.long else x.long()
        if self.num_channels == 1:
            x = log_probs.squeeze().transpose(1, 2)[
                range(self.num_vars), index.squeeze()].unsqueeze(-1)
        else:
            x = log_probs.view(self.num_vars, -1, self.num_channels, self.num_categories)[
                torch.arange(self.num_vars)[:, None].repeat(x.size(0), 1), :,
                torch.arange(self.num_channels)[None, :],
                index.flatten(0, 1)].view(x.size(0), self.num_vars, self.num_channels, -1).sum(2).unsqueeze(-1)
        return x
