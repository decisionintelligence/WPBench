# -*- coding: utf-8 -*-
__all__ = [
    "AGCRN",
    "TGGC",
    "STDN",
    "STWave",
    "STSSDL",
    "STID",
    "PatchSTG",
    "BigST",
]


from ts_benchmark.baselines.st_model.adapters_for_stdn import STDN
from ts_benchmark.baselines.st_model.models.AGCRN import AGCRN
from ts_benchmark.baselines.st_model.models.STID import STID
from ts_benchmark.baselines.st_model.models.STSSDL import STSSDL
from ts_benchmark.baselines.st_model.models.STWave import STWave
from ts_benchmark.baselines.st_model.models.TGGC import TGGC
from ts_benchmark.baselines.st_model.models.PatchSTG import PatchSTG
from ts_benchmark.baselines.st_model.models.BigST import BigST
