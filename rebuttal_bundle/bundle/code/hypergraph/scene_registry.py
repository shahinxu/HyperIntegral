import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class SceneSpec:
    key: str
    module: str
    label: str


SCENE_REGISTRY = {
    "ecological": SceneSpec("ecological", "lib_ecological_dynamics.hypergraph", "ecological_dynamics"),
    "neuronal": SceneSpec("neuronal", "lib_neuronal_synchronization.hypergraph", "neuronal_synchronization"),
    "rossler": SceneSpec("rossler", "lib_rossler_oscillator.hypergraph", "rossler_oscillator"),
    "social": SceneSpec("social", "lib_social_contagion.hypergraph", "social_contagion"),
}


def get_scene_model(scene: str):
    scene_key = scene.lower().strip()
    if scene_key not in SCENE_REGISTRY:
        valid = ", ".join(sorted(SCENE_REGISTRY.keys()))
        raise ValueError(f"Unknown scene '{scene}'. Valid scenes: {valid}")
    spec = SCENE_REGISTRY[scene_key]
    mod = importlib.import_module(spec.module)
    return mod.HypergraphModel, spec
