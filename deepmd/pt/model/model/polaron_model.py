# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
)

import torch

from deepmd.pt.model.atomic_model import (
    DPPolaronAtomicModel
)
from deepmd.pt.model.model.model import (
    BaseModel,
)

from .dp_model import (
    DPModelCommon,
)
from .make_model import (
    make_model,
)

DPPolaronAtomicModel_ = make_model(DPPolaronAtomicModel)


@BaseModel.register("polaron")
class PolaronModel(DPModelCommon, DPPolaronAtomicModel_):
    model_type = "polaron"

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        DPModelCommon.__init__(self)
        DPPolaronAtomicModel_.__init__(self, *args, **kwargs)

    def translated_output_def(self):
        out_def_data = self.model_output_def().get_data()
        output_def = {
            f"spin": out_def_data["spin"],
        }
        if "mask" in out_def_data:
            output_def["mask"] = out_def_data["mask"]
        return output_def

    def forward(
        self,
        coord,
        atype,
        box: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:
        model_ret = self.forward_common(
            coord,
            atype,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )
        model_predict = {}
        model_predict["spin"] = model_ret["spin"]

        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
        return model_predict 
    
    @torch.jit.export
    def forward_lower(
        self,
        extended_coord,
        extended_atype,
        nlist,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
        comm_dict: Optional[dict[str, torch.Tensor]] = None,
    ):
        model_ret = self.forward_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            comm_dict=comm_dict,
            extra_nlist_sort=self.need_sorted_nlist_for_lower(),
        )
        model_predict = {}
        model_predict["spin"] = model_ret["spin"]
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
        return model_predict