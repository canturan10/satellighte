import errno
import os
import sys
from typing import List

from ..core import (
    _download_file_from_url,
    _get_arch_cls,
    _get_arch_dir,
    _get_from_config_file,
    _get_list_of_dirs,
    _get_model_dir,
    _get_model_info,
)

# APIs for Model


def available_models() -> list:
    """
    List of available models.

    Returns:
        list: List of available models.
    """
    # pylint: disable=E1136

    model_list = []
    model_config = _get_from_config_file("MODEL")
    for arch in model_config.keys():
        for config in model_config[arch].keys():
            for dataset in model_config[arch][config].keys():
                model_list.append(f"{arch}_{config}_{dataset}")
    return sorted(model_list)


def get_model_versions(model: str) -> list:
    """
    Get list of available versions for the given model.

    Args:
        model (str): Model name.

    Returns:
        list: List of available versions.
    """
    model_dir = _get_model_dir()
    assert model in available_models(), f"given model: {model} not in the registry"
    model_path = os.path.join(model_dir, model)

    versions = []
    for version in _get_list_of_dirs(model_path):
        versions.append(version[1:])

    return sorted(versions)


def get_model_latest_version(model: str) -> int:
    """
    Get latest version of the given model.

    Args:
        model (str): Model name.

    Returns:
        int: Latest version.
    """
    return max(get_model_versions(model))


def get_saved_model(model: str, version: str):
    """
    Get saved model.

    Args:
        model (str): Model name.
        version (str): Model version.
    """
    model_dir = os.path.join(_get_model_dir(), model, f"v{version}")

    info_dict = _get_model_info(model, f"v{version}")
    url = info_dict.get("url")

    try:
        os.makedirs(model_dir)
    except OSError as os_error:
        if os_error.errno != errno.EEXIST:
            raise
        # time.sleep might help here

    cached_file = os.path.join(model_dir, model + ".pth")

    if not os.path.exists(cached_file):
        sys.stderr.write(f'Downloading: "{url}" to {cached_file}\n')
        _download_file_from_url(url, cached_file, progress=True)

    # ToDo: Make it for zip file


# APIs for Architecture


def available_archs() -> List[str]:
    """
    List of available architectures.

    Returns:
        List[str]: List of available architectures.
    """
    archs_dir = _get_arch_dir()
    dirs = _get_list_of_dirs(archs_dir)
    return sorted(dirs)


def get_arch_configs(arch: str) -> List[str]:
    """
    List of available configurations for the given architecture.

    Args:
        arch (str): Architecture name.

    Returns:
        List[str]: List of available configurations.
    """
    cls = _get_arch_cls(arch)
    configs = getattr(cls, "__CONFIGS__", {})
    return sorted(list(configs.keys()))
