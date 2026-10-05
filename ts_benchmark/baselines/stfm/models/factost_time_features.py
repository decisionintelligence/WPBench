"""FactoST STA temporal feature helpers.

FactoST uses a two-layer timestamp contract:
- model input carries the fixed base slots below;
- ``temporal_features`` only controls which slots are actively embedded.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

FACTOST_BASE_TEMPORAL_FEATURES = (
    "minutes_of_hour",
    "time_of_day",
    "day_of_week",
    "day_of_month",
    "month_of_year",
)

DEFAULT_ACTIVE_TEMPORAL_FEATURES = {"day_of_week": 7}


def parse_active_temporal_features(
    spec,
    default: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    """Parse the model-side active temporal feature spec."""
    fallback = DEFAULT_ACTIVE_TEMPORAL_FEATURES if default is None else default
    if spec is None:
        return dict(fallback)
    if isinstance(spec, dict):
        return {str(k).strip(): int(v) for k, v in spec.items() if str(k).strip()}

    text = str(spec).strip()
    if not text:
        return dict(fallback)

    parsed = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"Invalid FactoST temporal feature spec '{item}'. Expected 'name:size'."
            )
        name, size = item.split(":", 1)
        parsed[name.strip()] = int(size)
    return parsed


def build_factost_base_time_slots(timestamps) -> np.ndarray:
    """Build fixed FactoST slots in [minute, hour, weekday, day, month] order."""
    values = np.asarray(timestamps)
    original_shape = values.shape
    flat_times = pd.DatetimeIndex(pd.to_datetime(values.reshape(-1)))

    slots = np.stack(
        [
            flat_times.minute,
            flat_times.hour,
            flat_times.weekday,
            flat_times.day,
            flat_times.month,
        ],
        axis=-1,
    ).astype(np.float32)
    return slots.reshape(original_shape + (len(FACTOST_BASE_TEMPORAL_FEATURES),))
