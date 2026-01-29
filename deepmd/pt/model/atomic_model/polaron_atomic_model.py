# SPDX-License-Identifier: LGPL-3.0-or-later

import torch

from deepmd.pt.model.task.polaron import (
    PolaronFittingNet
)

from .dp_atomic_model import (
    DPAtomicModel,
)


class DPPolaronAtomicModel(DPAtomicModel):
    def __init__(self, descriptor, fitting, type_map, **kwargs):
        if not isinstance(fitting, PolaronFittingNet): ######## Change here
            raise TypeError(
                "fitting must be an instance of PropertyFittingNet for DPPropertyAtomicModel"
            )
        super().__init__(descriptor, fitting, type_map, **kwargs)

    def apply_out_stat(
        self,
        ret: dict[str, torch.Tensor],
        atype: torch.Tensor,
    ):
        """don't apply bias for polaron fitting"""
        return ret