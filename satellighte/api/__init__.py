import os
from typing import List

from ..core import (
    _get_model_dir,
    _get_list_of_dirs,
    _get_arch_dir,
    _get_arch_cls,
)

# APIs for Model


def available_models() -> list:
    model_dir = _get_model_dir()
    dirs = _get_list_of_dirs(model_dir)
    return sorted(dirs)


def get_model_versions(model: str) -> list:
    model_dir = _get_model_dir()
    assert model in available_models(), f"given model: {model} not in the registry"
    model_path = os.path.join(model_dir, model)

    versions = []
    for version in _get_list_of_dirs(model_path):
        if not version.isdigit():
            raise AssertionError(f"version is not digit: {version}")
        versions.append(int(version))

    return sorted(versions)


def get_model_latest_version(model: str) -> int:
    return max(get_model_versions(model))


# APIs for Architecture


def available_archs() -> List[str]:
    archs_dir = _get_arch_dir()
    dirs = _get_list_of_dirs(archs_dir)
    return sorted(dirs)


def get_arch_configs(arch: str) -> List[str]:
    cls = _get_arch_cls(arch)
    configs = getattr(cls, "__CONFIGS__", {})
    return sorted(list(configs.keys()))
