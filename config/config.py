import yaml
import argparse
import os
from dataclasses import dataclass
from typing import List, Any, Dict


# --- Dataclasses for Each Section ---

@dataclass
class SharedConfig:
    batch_size: int
    num_workers: int
    num_class: int
    kernel_size: int
    num_repeat_kernel: int
    grid_size: int


@dataclass
class TrainingConfig:
    num_epoch: int
    learning_rate: float
    weight_decay: float
    do_update_caching: bool
    do_preprocess: bool
    frac_training: float
    frac_testing: float
    load_model: bool
    resume_optimizer: bool
    model_path: str
    use_class_weights: bool
    ROOT_DIR: str
    TRAIN_FILES: str
    TEST_FILES: str


@dataclass
class InferenceConfig:
    do_preprocess: bool
    verbose: bool
    chunk_size: int
    src_inf_root: str
    src_inf_data: str
    src_inf_results: str
    src_model: str
    inference_file: str
    use_class_weighting: bool
    save_predictions: bool

@dataclass
class FullConfig:
    shared: SharedConfig
    training: TrainingConfig
    inference: InferenceConfig

# --- Utilities ---

def parse_overrides(override_list: List[str]) -> Dict[str, Any]:
    overrides = {}
    for item in override_list:
        if '=' not in item:
            raise ValueError(f"Invalid override: {item}. Use key=value.")
        key, value = item.split('=', 1)
        keys = key.split('.')
        current = overrides
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        try:
            value = eval(value, {}, {})
        except:
            pass
        current[keys[-1]] = value
    return overrides


def merge_dicts(base: dict, overrides: dict) -> dict:
    for k, v in overrides.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            merge_dicts(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str = "config.yaml", overrides: List[str] = []) -> FullConfig:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw_config = yaml.safe_load(f)

    if overrides:
        override_dict = parse_overrides(overrides)
        raw_config = merge_dicts(raw_config, override_dict)

    shared_dict = raw_config.get("shared", {})

    training = TrainingConfig(**raw_config.get("training", {}))
    inference = InferenceConfig(**raw_config.get("inference", {}))
    shared = SharedConfig(**shared_dict)

    return FullConfig(shared=shared, training=training, inference=inference)

def get_config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--override", nargs="*", default=[], help="Override config values, e.g. training.batch_size=16")
    return parser
