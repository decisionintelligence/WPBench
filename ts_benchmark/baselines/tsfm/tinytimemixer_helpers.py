from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml
from pandas.tseries.frequencies import to_offset


DEFAULT_TTM_FREQUENCY_MAPPING = {
    "oov": 0,
    "min": 1,
    "2min": 2,
    "5min": 3,
    "10min": 4,
    "15min": 5,
    "30min": 6,
    "h": 7,
    "H": 7,
    "d": 8,
    "D": 8,
    "W": 9,
}


def default_ttm_yaml_path() -> str:
    return str(Path(__file__).resolve().parent / "checkpoints" / "ttm.yaml")


def get_ttm_frequency_token(freq: str, mapping: Optional[Dict[str, int]] = None) -> int:
    mapping = DEFAULT_TTM_FREQUENCY_MAPPING if mapping is None else mapping

    if freq in mapping:
        return mapping[freq]

    try:
        normalized = to_offset(freq).freqstr
        if normalized in mapping:
            return mapping[normalized]
    except ValueError:
        try:
            normalized = to_offset(pd.Timedelta(freq)).freqstr
            if normalized in mapping:
                return mapping[normalized]
        except Exception:
            pass

    return mapping["oov"]


def _flatten_ttm_index(raw: Dict) -> List[Dict]:
    rows: List[Dict] = []
    # item中至少含有context_length和prediction_length两个字段，才能被认为是一个有效的ttm checkpoint配置项
    for section_name, section in raw.items():
        if not isinstance(section, dict):
            continue
        for model_key, item in section.items():
            if not isinstance(item, dict):
                continue
            if "context_length" not in item or "prediction_length" not in item:
                continue
            row = dict(item)
            row["__section__"] = section_name
            row["__key__"] = model_key
            rows.append(row)
    return rows


def resolve_ttm_revision(
    context_length: int,
    prediction_length: int,
    *,
    model_card: str = "ibm-granite/granite-timeseries-ttm-r2",
    yaml_path: Optional[str] = None,
) -> Tuple[str, int, str]:
    # 默认存储位置 当前路径
    yaml_path = default_ttm_yaml_path() if not yaml_path else yaml_path
    
    # yaml解压配置后会成为一个dict
    with open(yaml_path, "r", encoding="utf-8") as f:
        rows = _flatten_ttm_index(yaml.safe_load(f))

    # 首先从 rows里面找到任务的输入的context_length和dict里面一样的，把这些项目都取出来
    rows = [
        r
        for r in rows
        if str(r.get("model_card", "")) == model_card
        and int(r.get("context_length", -1)) == int(context_length)
    ]
    if not rows:
        raise ValueError(
            f"No TTM checkpoint found for model_card={model_card}, context_length={context_length}."
        )
    # 看看有没有预测长度也匹配的
    exact = [r for r in rows if int(r["prediction_length"]) == int(prediction_length)]
    if exact:
        chosen = exact[0]
        return chosen["revision"], int(chosen["prediction_length"]), str(chosen["release"])

    # 预测长度也匹配之后，就找接近的更短checkpoint(也就是找一个和当前预测长度最接近的checkpoint,prediction_length < 目标 prediction_length这个check)
    shorter = [r for r in rows if int(r["prediction_length"]) < int(prediction_length)]
    
    if shorter:
        chosen = sorted(shorter, key=lambda x: int(x["prediction_length"]))[-1]
        return chosen["revision"], int(chosen["prediction_length"]), str(chosen["release"])

    # 如果都没有比当前任务更小的checkpoint就直接找最小的来截断
    # context一定要一致，如果没有匹配的prediction，找到一个和预测的prediction长度最接近的更小的，如果没有更小的就取最小的，这里的prediciton是一次滚动的长度
    chosen = sorted(rows, key=lambda x: int(x["prediction_length"]))[0]
    return chosen["revision"], int(chosen["prediction_length"]), str(chosen["release"])
