import functools
import importlib
import os
from typing import Dict, List, Tuple

import yaml

__PACKAGE_DIR__ = os.path.dirname(__file__)


def _get_local_package_dir() -> str:
    assert os.path.exists(
        __PACKAGE_DIR__
    ), f"package directory not found: {__PACKAGE_DIR__}"
    return __PACKAGE_DIR__


def _get_config_path(config_path=None) -> str:
    if config_path is None:
        config_path = os.path.join(__PACKAGE_DIR__, ".config.yaml")
    if not os.path.exists(config_path):
        raise ValueError("Config path {} does not exist!".format(config_path))
    return config_path


def _load_config_file(config_path=None) -> Dict:
    yaml_path = _get_config_path(config_path)
    Loader = yaml.FullLoader
    with open(yaml_path, "r") as config_file:
        cfg = yaml.load(config_file, Loader=Loader)

    return cfg


def _get_from_config_file(key: str = None, cfg: Dict = None):
    if isinstance(key, type(None)):
        if isinstance(cfg, type(None)):
            cfg = _load_config_file()
        return cfg
    else:
        if isinstance(cfg, type(None)):
            cfg = _load_config_file()

        key, *next_key = key.split(".")
        next_key = ".".join(next_key) if len(next_key) > 0 else None
        assert key in cfg.keys(), "yaml file does not contains given key"
        cfg = cfg.get(key)
    return _get_from_config_file(key=next_key, cfg=cfg)


def _get_model_dir() -> str:
    model_registry_dir = os.path.join(
        _get_local_package_dir(), _get_from_config_file("PACKAGE.MODEL")
    )
    assert os.path.exists(
        model_registry_dir
    ), f"model directory not found: {model_registry_dir}"
    return model_registry_dir


def _get_arch_dir() -> str:
    arch_registry_dir = os.path.join(
        _get_local_package_dir(), _get_from_config_file("PACKAGE.ARCH")
    )
    assert os.path.exists(
        arch_registry_dir
    ), f"architecture directory not found: {arch_registry_dir}"
    return arch_registry_dir


def _get_arch_pkg(arch: str):
    pkg_name = _get_from_config_file("PACKAGE.NAME")
    arch_postfix = _get_from_config_file("PACKAGE.ARCH")
    return importlib.import_module(".".join([pkg_name, arch_postfix, arch]))


def _get_arch_cls(arch: str):
    arch_pkg = _get_arch_pkg(arch)
    for arch_name in dir(arch_pkg):
        if arch_name.lower() == arch:
            return getattr(arch_pkg, arch_name)


def _get_list_of_dirs(dir_path: str) -> list:
    return [
        d
        for d in os.listdir(dir_path)
        if os.path.isdir(os.path.join(dir_path, d)) and not d.startswith("_")
    ]


def _parse_saved_model_name(model: str) -> Tuple[str, str, str]:
    arch = None
    config = None
    dataset = None
    splits = model.split("_")
    assert len(splits) in [
        3,
    ], f"model must contain 3 under scores (_) but found: {len(splits)}"
    arch, config, dataset = splits
    return arch, config, dataset
