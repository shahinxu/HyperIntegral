"""Re-attach summary.json files to results.jsonl records that lost them.

A sweep that is interrupted (or that mixes code versions across a restart) can
leave records whose run actually succeeded but whose `macro_auc` is null. The
outputs are still on disk, so this walks every stage directory, re-reads the
summary for each record, and rewrites results.jsonl in place.
"""
import json
import sys
from pathlib import Path


SCENE_TOKENS = ("ecological", "social", "neuronal", "rossler")


def _scene_of(job: str):
    for t in SCENE_TOKENS:
        if f"_{t}_" in job or job.endswith(f"_{t}"):
            return t
    return None


def _variant_of(job: str) -> str:
    """Recover the variant from the run tag, which is '<variant>_<scene>_...'."""
    sc = _scene_of(job)
    return job.split(f"_{sc}_")[0] if sc and f"_{sc}_" in job else job


def repair(stage_dir: Path) -> tuple[int, int, int]:
    rf = stage_dir / "results.jsonl"
    if not rf.exists():
        return (0, 0, 0)
    rows, seen = [], set()
    for line in rf.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        # keep the newest record per job
        seen.add(r["job"])
        rows.append(r)
    by_job = {}
    for r in rows:
        by_job[r["job"]] = r

    # the run tag is authoritative for scene/variant; records recovered from a
    # bare summary.json carry neither
    for job, r in by_job.items():
        meta = r.setdefault("meta", {})
        if not meta.get("variant") or meta.get("variant") == "?":
            meta["variant"] = _variant_of(job)
        if not meta.get("scene") or meta.get("scene") not in SCENE_TOKENS:
            meta["scene"] = _scene_of(job) or meta.get("scene")

    fixed = 0
    for job, r in by_job.items():
        if r.get("macro_auc") is not None:
            continue
        hits = list(stage_dir.rglob(f"{job}/summary.json"))
        if not hits:
            continue
        s = json.loads(hits[0].read_text())
        s["_dir"] = str(hits[0].parent)
        r["summary"] = s
        r["auc"] = s.get("auc")
        r["macro_auc"] = s.get("extra_metrics", {}).get("macro_auc")
        if r["macro_auc"] is None:
            a = [v for v in (s.get("auc") or {}).values() if v is not None]
            r["macro_auc"] = sum(a) / len(a) if a else None
        if r["macro_auc"] is not None:
            r["returncode"] = 0
            fixed += 1

    # pick up any run whose directory exists but was never recorded at all
    added = 0
    for p in stage_dir.rglob("summary.json"):
        job = p.parent.name
        if job in by_job or job == "logs":
            continue
        s = json.loads(p.read_text()); s["_dir"] = str(p.parent)
        a = [v for v in (s.get("auc") or {}).values() if v is not None]
        by_job[job] = {
            "job": job, "returncode": 0, "seconds": None,
            "meta": {"scene": _scene_of(job) or s.get("scene"),
                     "variant": _variant_of(job), "method": s.get("method"),
                     "noise": s.get("config", {}).get("noise"),
                     "seed": s.get("config", {}).get("seed"),
                     "n_nodes": s.get("config", {}).get("n_nodes")},
            "summary": s, "auc": s.get("auc"),
            "macro_auc": s.get("extra_metrics", {}).get("macro_auc")
                         or (sum(a) / len(a) if a else None),
        }
        added += 1

    rf.write_text("\n".join(json.dumps(v) for v in by_job.values()) + "\n")
    ok = sum(1 for v in by_job.values() if v.get("macro_auc") is not None)
    return (fixed, added, ok)


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/paper")
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        fixed, added, ok = repair(d)
        total = len(((d / "results.jsonl").read_text().splitlines()
                     if (d / "results.jsonl").exists() else []))
        if total:
            print(f"{d.name:<20} repaired={fixed:<4} recovered={added:<4} "
                  f"usable={ok}/{total}")
