import json
import os
from datetime import datetime


def _normalize_auc_map(auc_scores: dict) -> dict:
    out = {}
    for k, v in (auc_scores or {}).items():
        key = str(k)
        out[key] = None if v is None else float(v)
    return out


def write_standard_summary(
    save_dir: str,
    method: str,
    scene: str,
    config: dict,
    auc_scores: dict,
    extra_metrics: dict | None = None,
):
    os.makedirs(save_dir, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "method": method,
        "scene": scene,
        "config": config,
        "auc": _normalize_auc_map(auc_scores),
        "extra_metrics": extra_metrics or {},
    }
    with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(os.path.join(save_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"method={method}\n")
        f.write(f"scene={scene}\n")
        for k, v in (config or {}).items():
            f.write(f"config.{k}={v}\n")
        for k, v in payload["auc"].items():
            f.write(f"auc.{k}={v}\n")
        for k, v in payload["extra_metrics"].items():
            f.write(f"metric.{k}={v}\n")
