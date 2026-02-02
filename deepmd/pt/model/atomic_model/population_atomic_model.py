# SPDX-License-Identifier: LGPL-3.0-or-later

import torch

from deepmd.pt.model.task.population import (
    PopulationFittingNet
)

from .dp_atomic_model import (
    DPAtomicModel,
)


class DPPopulationAtomicModel(DPAtomicModel):
    def __init__(self, descriptor, fitting, type_map, **kwargs):
        if not isinstance(fitting, PopulationFittingNet): ######## Change here
            raise TypeError(
                "fitting must be an instance of PropertyFittingNet for DPPropertyAtomicModel"
            )
        super().__init__(descriptor, fitting, type_map, **kwargs)

    def apply_out_stat(
        self,
        ret: dict[str, torch.Tensor],
        atype: torch.Tensor,
    ):
        """don't apply bias for population fitting"""
        return ret