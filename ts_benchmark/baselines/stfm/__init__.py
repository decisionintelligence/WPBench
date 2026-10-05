"""Bridge layer for spatio-temporal foundation model integrations."""

from ts_benchmark.baselines.stfm.models import (
    FactoST_STA,
    FactoST_UTP,
    FactoST_UTP_Finetune,
    OpenCity_STFM,
)

__all__ = ["FactoST_STA", "FactoST_UTP", "FactoST_UTP_Finetune", "OpenCity_STFM"]
