from typing import Literal

import torch
from torch import Tensor

from tenpcs.layers.sum_product.sum_product import SumProductLayer
from tenpcs.reparams.leaf import ReparamIdentity
from tenpcs.utils.type_aliases import ReparamFactory


class TRLayer(SumProductLayer):
    """Tensor Ring (2) layer."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: Literal[2] = 2,
        num_folds: int = 1,
        fold_mask: None = None,
        reparam: ReparamFactory = ReparamIdentity,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units.
            arity (Literal[2], optional): The arity of the layer, must be 2. Defaults to 2.
            num_folds (int, optional): The number of folds. Defaults to 1.
            fold_mask (None, optional): The mask of valid folds, must be None. Defaults to None.
            reparam (ReparamFactory, optional): The reparameterization. Defaults to ReparamIdentity.

        Raises:
            NotImplementedError: When arity is not 2.
        """
        if arity != 2:
            raise NotImplementedError("TR layers only implement binary product units.")
        assert fold_mask is None, "Input for TR layer should not be masked."
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            num_folds=num_folds,
            fold_mask=None,
            reparam=reparam,
        )

        if num_input_units == num_output_units:
            rank = 16
            K = num_input_units
            # dim arg is not important when reparam is identity
            self.fold_core = reparam((rank, num_folds, rank), dim=(1, 2))
            self.K_cores = reparam((3, rank, K, rank), dim=(1, 2))
        else:
            self.params = reparam(
                (num_folds, num_input_units, num_input_units, num_output_units), dim=(1, 2)
            )

        self.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (F, H, K, *B).

        Returns:
            Tensor: The output of this layer, shape (F, K, *B).
        """
        if self.num_input_units == self.num_output_units:
            max_log_prob = x.max(dim=2, keepdim=True).values
            exp_log_prob = torch.exp(x - max_log_prob)

            K_cores = self.K_cores()
            out = torch.einsum("fib,BiC-> fBCb", exp_log_prob[:, 0], K_cores[0])
            out2 = torch.einsum("fjb, CjD -> fCDb", exp_log_prob[:, 1], K_cores[1])
            out = torch.einsum("fBCb, fCDb, AfB, DoA -> fob", out, out2, self.fold_core(), K_cores[2])

            # out = torch.einsum("fib,fjb,BiC,CjD-> fBDb", exp_log_prob[:, 0], exp_log_prob[:, 1], self.K_cores()[0], self.K_cores()[1])
            # out = torch.einsum("fBDb, AfB, DoA -> fob", out, self.fold_core(), self.K_cores()[2])

            # out = torch.einsum("fib, fjb, AfB,BiC,CjD, DoA -> fob", exp_log_prob[:, 0], exp_log_prob[:, 1], self.fold_core(),self.K_cores()[0], self.K_cores()[1], self.K_cores()[2])  # FULL EINSUM

            return max_log_prob.sum(dim=1) + out.log()
        else:
            max_log_prob = x.max(dim=2, keepdim=True).values
            exp_log_prob = torch.exp(x - max_log_prob)
            out = max_log_prob.sum(dim=1) + torch.log(
                torch.einsum("fib,fjb,fijo->fob", exp_log_prob[:, 0], exp_log_prob[:, 1], self.params()))
            return out
