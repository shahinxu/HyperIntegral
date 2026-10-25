"""Generate a 30-oscillator neuronal preset following the n60 design rule.

Reverse-engineered from neuronal_synchronization_n60_hypergraph.json:
  * 3 equal modules
  * every order splits within-module : cross-module = 2 : 1
  * within-module hyperedges are shared equally by the 3 modules
  * order-2 cross edges join 2 modules; order >= 3 cross edges span all 3
  * counts decrease with order (n60: 180 / 90 / 45 / 18)

For n=30 the counts are halved while keeping the 2:1 split exact and the
within-module count divisible by 3, which forces totals to be multiples of 9:
90 / 45 / 18 / 9.
"""
import json, random
from itertools import combinations
from pathlib import Path

N, N_MOD = 30, 3
SIZE = N // N_MOD
MODULES = {f"module_{m+1}": list(range(m*SIZE+1, (m+1)*SIZE+1)) for m in range(N_MOD)}
COUNTS = {"edges2": 90, "edges3": 45, "edges4": 18, "edges5": 9}

rng = random.Random(20261022)


def sample_within(order, n_total):
    """n_total within-module edges, split equally across the 3 modules."""
    per = n_total // N_MOD
    out = []
    for nodes in MODULES.values():
        pool = list(combinations(sorted(nodes), order))
        rng.shuffle(pool)
        assert len(pool) >= per, f"module too small for order {order}"
        out += [list(e) for e in pool[:per]]
    return out


def sample_cross(order, n_total, n_span, seen):
    """n_total cross-module edges spanning exactly n_span modules."""
    out = []
    guard = 0
    while len(out) < n_total:
        guard += 1
        assert guard < 200000, f"cannot fill cross edges for order {order}"
        mods = rng.sample(list(MODULES), n_span)
        # one member from each spanned module, remainder distributed at random
        picked = [rng.choice(MODULES[m]) for m in mods]
        while len(picked) < order:
            m = rng.choice(mods)
            c = rng.choice(MODULES[m])
            if c not in picked:
                picked.append(c)
        e = tuple(sorted(picked))
        if len(e) != order or e in seen:
            continue
        # verify it really spans n_span modules
        spanned = {m for m, nodes in MODULES.items() if set(e) & set(nodes)}
        if len(spanned) != n_span:
            continue
        seen.add(e)
        out.append(list(e))
    return out


payload = {
    "name": "neuronal_synchronization_n30_hypergraph_v1",
    "n_nodes": N,
    "index_base": 1,
    "description": ("Modular simplicial topology for 30 oscillators, generated to follow "
                    "the n60 preset's design rule (3 modules, within:cross = 2:1 per order, "
                    "within-module edges split equally across modules, counts decreasing "
                    "with order). The original n30 preset used in the paper was not under "
                    "version control and could not be recovered; this is a regenerated "
                    "topology, not the original one."),
    "modules": MODULES,
}

for order, key in ((2, "edges2"), (3, "edges3"), (4, "edges4"), (5, "edges5")):
    total = COUNTS[key]
    n_within = total // 3 * 2
    n_cross = total - n_within
    within = sample_within(order, n_within)
    seen = {tuple(e) for e in within}
    cross = sample_cross(order, n_cross, 2 if order == 2 else 3, seen)
    edges = sorted([tuple(e) for e in within + cross])
    assert len(set(edges)) == total, (key, len(set(edges)), total)
    payload[key] = [list(e) for e in edges]

payload["counts"] = {k: len(payload[k]) for k in COUNTS}

out = Path("lib_neuronal_synchronization/presets/neuronal_synchronization_n30_hypergraph.json")
out.write_text(json.dumps(payload, indent=2))
print("wrote", out)
print("counts:", payload["counts"], "total true edges:", sum(payload["counts"].values()))
