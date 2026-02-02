# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    Union,
)

import torch
import torch.nn.functional as F

from deepmd.pt.loss.loss import (
    TaskLoss,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.utils.data import (
    DataRequirementItem,
)

from functools import partial

log = logging.getLogger(__name__)

class PopulationLoss(TaskLoss):
    def __init__(
        self,
        loss_func: str = "smooth_mae",
        metric: list = ["mae"],
        starter_learning_rate: float=1.0,
        start_pref_spin: float = 1.00,
        limit_pref_spin: float = 1.00,
        start_pref_spin_total: float = 1.00,
        limit_pref_spin_total: float = 1.00,
        start_pref_pop: float = 1.00,
        limit_pref_pop: float = 1.00,
        start_pref_pop_alpha_total: float = 1.00,
        limit_pref_pop_alpha_total: float = 1.00,
        start_pref_pop_beta_total: float = 1.00,
        limit_pref_pop_beta_total: float = 1.00,
        beta: float = 1.00,
        **kwargs,
    ) -> None:
        r"""Construct a layer to compute loss on property.

        Parameters
        ----------
        task_dim : float
            The output dimension of property fitting net.
        var_name : str
            The atomic property to fit, 'energy', 'dipole', and 'polar'.
        loss_func : str
            The loss function, such as "smooth_mae", "mae", "rmse".
        metric : list
            The metric such as mae, rmse which will be printed.
        starter_learning_rate : float
            The learning rate for the model.
        start_pref_m : float
            The starting value for pref_m.
        limit_pref_m : float
            The limit value for pref_m.
        start_pref_t : float
            The starting value for pref_t.
        limit_pref_t : float
            The limit value for pref_t.
        beta : float
            The 'beta' parameter in 'smooth_mae' loss.
        """
        super().__init__()
        self.task_dim = 2            # alpha and beta channels
        self.var_name = "atom_spin"
        self.loss_func = loss_func
        self.metric = metric
        self.beta = beta

        self.starter_learning_rate = starter_learning_rate
        self.start_pref_spin = start_pref_spin
        self.limit_pref_spin = limit_pref_spin
        self.start_pref_spin_total = start_pref_spin_total
        self.limit_pref_spin_total = limit_pref_spin_total
        self.start_pref_pop = start_pref_pop
        self.limit_pref_pop = limit_pref_pop
        self.start_pref_pop_alpha_total = start_pref_pop_alpha_total
        self.limit_pref_pop_alpha_total = limit_pref_pop_alpha_total
        self.start_pref_pop_beta_total = start_pref_pop_beta_total
        self.limit_pref_pop_beta_total = limit_pref_pop_beta_total
        assert (
            self.start_pref_spin >= 0.0
            and self.limit_pref_spin >= 0.0
        ), "Can't assign negative value to `start_pref_spin`"

    def forward(self, input_dict, model, label, natoms, learning_rate=0.0, mae=False):
        """Return loss on properties .

        Parameters
        ----------
        input_dict : dict[str, torch.Tensor]
            Model inputs.
        model : torch.nn.Module
            Model to be used to output the predictions.
        label : dict[str, torch.Tensor]
            Labels.
        natoms : int
            The local atom number.

        Returns
        -------
        model_pred: dict[str, torch.Tensor]
            Model predictions.
        loss: torch.Tensor
            Loss for model to minimize.
        more_loss: dict[str, torch.Tensor]
            Other losses for display.
        """
        model_pred = model(**input_dict)
        
        coef = learning_rate / self.starter_learning_rate
        pref_spin = self.limit_pref_spin + (self.start_pref_spin - self.limit_pref_spin) * coef
        pref_spin_total = self.limit_pref_spin_total + (self.start_pref_spin_total - self.limit_pref_spin_total) * coef
        pref_pop = self.limit_pref_pop + (self.start_pref_pop - self.limit_pref_pop) * coef
        pref_pop_alpha_total = self.limit_pref_pop_alpha_total + (self.start_pref_pop_alpha_total - self.limit_pref_pop_alpha_total) * coef
        pref_pop_beta_total = self.limit_pref_pop_beta_total + (self.start_pref_pop_beta_total - self.limit_pref_pop_beta_total) * coef
        
        loss = torch.zeros(1, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)[0]
        more_loss = {}

        # get the label and model prediction
        pop_pred = model_pred["spin"][0]
        pop_label = label["atom_spin"].reshape([natoms, self.task_dim])

        # print('pop_pred', pop_pred)
        # print('pop_label', pop_label)
        # print('pop_pred.shape', pop_pred.shape)
        # print('pop_label.shape', pop_label.shape)

        # spin_pred2 = pop_pred[:, :, 0] - pop_pred[:, :, 1]
        # spin_label2 = pop_label[:, :, 0] - pop_label[:, :, 1]
        # print('spin_pred2.shape', spin_pred2.shape)
        # print('spin_label2.shape', spin_label2.shape)

        spin_pred = torch.sub(pop_pred[:, 0], pop_pred[:, 1])
        spin_label = torch.sub(pop_label[:, 0], pop_label[:, 1])
        # spin_pred = torch.sub(pop_pred[:, :, 0], pop_pred[:, :, 1])
        # spin_label = torch.sub(pop_label[:, :, 0], pop_label[:, :, 1])
        # print('spin_pred.shape', spin_pred.shape)
        # print('spin_label.shape', spin_label.shape)
        # print('spin_pred', spin_pred)
        # print('spin_label', spin_label)

        # print('spin_pred[0].shape', spin_pred[0].shape)
        # print('spin_label[0].shape', spin_label[0].shape)

        spin_total_pred = torch.sum(spin_pred)
        spin_total_label = torch.sum(spin_label)
        # print('spin_total_label', spin_total_label)
        # print('spin_total_pred', spin_total_pred)

        pop_alpha_total_pred = torch.sum(pop_pred[:, 0]) 
        pop_beta_total_pred = torch.sum(pop_pred[:, 1])
        pop_alpha_total_label = torch.sum(pop_label[:, 0])
        pop_beta_total_label= torch.sum(pop_label[:, 1])
        # print('pop_alpha_total_pred', pop_alpha_total_pred)
        # print('pop_beta_total_pred', pop_beta_total_pred)
        # print('pop_alpha_total_label', pop_alpha_total_label)
        # print('pop_beta_total_label', pop_beta_total_label)

        # define the loss function
        if self.loss_func == "smooth_mae":
            loss_func = partial(F.smooth_l1_loss, reduction="sum", beta=self.beta)
        elif self.loss_func == "mae":
            loss_func = partial(F.l1_loss, reduction="sum")
        elif self.loss_func == "mse" :
            loss_func = partial(F.mse_loss, reduction="sum")
        elif self.loss_func == "rmse":
            loss_func = partial(F.mse_loss, reduction="mean")
        else:
            raise RuntimeError(f"Unknown loss function : {self.loss_func}")
        
        # loss_func() code fails for mse and rmse with RuntimeError: Found dtype Double but expected Float
        if self.loss_func == "smooth_mae" or self.loss_func == "mae":
            # calculate the loss
            spin_loss = loss_func(
                input=spin_pred,
                target=spin_label
            )
            spin_total_loss = loss_func(
                input=spin_total_pred,
                target=spin_total_label
            )
            pop_loss = loss_func(
                input=pop_pred,
                target=pop_label
            )
            pop_alpha_total_loss = loss_func(
                input=pop_alpha_total_pred,
                target=pop_alpha_total_label
            )
            pop_beta_total_loss = loss_func(
                input=pop_beta_total_pred,
                target=pop_beta_total_label
            )
        elif self.loss_func == "mse" :
            spin_loss = torch.mean(torch.square(spin_pred - spin_label))
            spin_total_loss = torch.mean(torch.square(spin_total_pred - spin_total_label))
            pop_loss = torch.mean(torch.square(pop_pred - pop_label))
            pop_alpha_total_loss = torch.mean(torch.square(pop_alpha_total_pred - pop_alpha_total_label))
            pop_beta_total_loss = torch.mean(torch.square(pop_beta_total_pred - pop_beta_total_label))

        elif self.loss_func == "rmse":
            spin_loss = torch.sqrt(torch.mean(torch.square(spin_pred - spin_label)))
            spin_total_loss = torch.sqrt(torch.mean(torch.square(spin_total_pred - spin_total_label)))
            pop_loss = torch.sqrt(torch.mean(torch.square(pop_pred - pop_label)))
            pop_alpha_total_loss = torch.sqrt(torch.mean(torch.square(pop_alpha_total_pred - pop_alpha_total_label)))
            pop_beta_total_loss = torch.sqrt(torch.mean(torch.square(pop_beta_total_pred - pop_beta_total_label)))
        else:
            raise RuntimeError(f"Unknown loss function : {self.loss_func}")            

        loss += (
            pref_spin * spin_loss +
            pref_spin_total * spin_total_loss +
            pref_pop * pop_loss +
            pref_pop_alpha_total * pop_alpha_total_loss +
            pref_pop_beta_total * pop_beta_total_loss
        )

        # pop_pred = model_pred["spin"][0]
        # pop_label = label["atom_spin"].reshape([natoms, self.task_dim])
        # pop_pred = pop_pred[:, 0]
        # pop_label = pop_label[:, 0]
        # l2_loss = torch.mean(torch.square(pop_pred - pop_label))
        # loss += spin_loss
        
        # loss += pref_spin
        # print(loss)

    # l2_ener_loss = torch.mean(torch.square(energy_pred - energy_label))

                       
        # spin_pred = model_pred["spin"]
        # spin_label = label["atom_spin"].reshape([-1, natoms, self.task_dim])
        # m_pred = spin_pred[:, 0] - spin_pred[:, 1]
        # m_label = spin_label[:, 0] - spin_label[:, 1]
        # M_pred = torch.sum(m_pred)
        # M_label = torch.sum(m_label)

        #  # calculate the loss
        # m_loss = loss_func(
        #     input=m_pred,
        #     target=m_label
        # )
        # spin_loss = loss_func(
        #     input=spin_pred,
        #     target=spin_label
        # )
        # M_loss = loss_func(
        #     input=M_pred,
        #     target=M_label
        # )

        # pref_t = pref_spin
        # pref_m = pref_spin
        # loss += pref_t * ( m_loss + spin_loss) + pref_m * M_loss
              
        # pref_t = pref_spin
        # pref_m = pref_spin
        # m_loss = 0.5 * (pop_alpha_loss + pop_beta_loss)
        # M_loss = spin_total_loss
        # loss += pref_t * ( m_loss + spin_loss) + pref_m * M_loss

        # more loss
        if "smooth_mae" in self.metric:
            loss_func = partial(F.smooth_l1_loss, reduction="mean", beta=self.beta)

            spin_loss = loss_func(
                input=spin_pred,
                target=spin_label
            )
            spin_total_loss = loss_func(
                input=spin_total_pred,
                target=spin_total_label
            )
            pop_loss = loss_func(
                input=pop_pred,
                target=pop_label
            )
            pop_alpha_total_loss = loss_func(
                input=pop_alpha_total_pred,
                target=pop_alpha_total_label
            )
            pop_beta_total_loss = loss_func(
                input=pop_beta_total_pred,
                target=pop_beta_total_label
            )

            more_loss["spin_total"] = spin_total_pred
            more_loss["spin_loss"] = spin_loss
            more_loss["spin_total_loss"] = spin_total_loss
            more_loss["pop_loss"] = pop_loss
            more_loss["pop_alpha_total_loss"] = pop_alpha_total_loss
            more_loss["pop_beta_total_loss"] = pop_beta_total_loss
            
        if "mae" in self.metric:
            loss_func = partial(F.l1_loss, reduction="mean")

            spin_loss = loss_func(
                input=spin_pred,
                target=spin_label
            )
            spin_total_loss = loss_func(
                input=spin_total_pred,
                target=spin_total_label
            )
            pop_loss = loss_func(
                input=pop_pred,
                target=pop_label
            )
            pop_alpha_total_loss = loss_func(
                input=pop_alpha_total_pred,
                target=pop_alpha_total_label
            )
            pop_beta_total_loss = loss_func(
                input=pop_beta_total_pred,
                target=pop_beta_total_label
            )

            more_loss["spin_total"] = spin_total_pred
            more_loss["spin_loss"] = spin_loss
            more_loss["spin_total_loss"] = spin_total_loss
            more_loss["pop_loss"] = pop_loss
            more_loss["pop_alpha_total_loss"] = pop_alpha_total_loss
            more_loss["pop_beta_total_loss"] = pop_beta_total_loss        

        if "mse" in self.metric:
            spin_loss = torch.mean(torch.square(spin_pred - spin_label))
            spin_total_loss = torch.mean(torch.square(spin_total_pred - spin_total_label))
            pop_loss = torch.mean(torch.square(pop_pred - pop_label))
            pop_alpha_total_loss = torch.mean(torch.square(pop_alpha_total_pred - pop_alpha_total_label))
            pop_beta_total_loss = torch.mean(torch.square(pop_beta_total_pred - pop_beta_total_label))

            more_loss["spin_total"] = spin_total_pred
            more_loss["spin_loss"] = spin_loss
            more_loss["spin_total_loss"] = spin_total_loss
            more_loss["pop_loss"] = pop_loss
            more_loss["pop_alpha_total_loss"] = pop_alpha_total_loss
            more_loss["pop_beta_total_loss"] = pop_beta_total_loss

        if "rmse" in self.metric:
            spin_loss = torch.sqrt(torch.mean(torch.square(spin_pred - spin_label)))
            spin_total_loss = torch.sqrt(torch.mean(torch.square(spin_total_pred - spin_total_label)))
            pop_loss = torch.sqrt(torch.mean(torch.square(pop_pred - pop_label)))
            pop_alpha_total_loss = torch.sqrt(torch.mean(torch.square(pop_alpha_total_pred - pop_alpha_total_label)))
            pop_beta_total_loss = torch.sqrt(torch.mean(torch.square(pop_beta_total_pred - pop_beta_total_label)))

            more_loss["spin_total"] = spin_total_pred
            more_loss["spin_loss"] = spin_loss
            more_loss["spin_total_loss"] = spin_total_loss
            more_loss["pop_loss"] = pop_loss
            more_loss["pop_alpha_total_loss"] = pop_alpha_total_loss
            more_loss["pop_beta_total_loss"] = pop_beta_total_loss

        return model_pred, loss, more_loss

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        """Return data label requirements needed for this loss calculation."""
        label_requirement = []
        label_requirement.append(
            DataRequirementItem(
                'atomic_spin',
                ndof=self.task_dim,
                atomic=True,
                must=True,
                high_prec=True,
            )
        )
        return label_requirement