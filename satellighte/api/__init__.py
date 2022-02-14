import errno
import os
import sys
from typing import List
from urllib.parse import urlparse

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
    model_list = []
    dict = _get_from_config_file("MODEL")
    for arch in dict.keys():
        for config in dict[arch].keys():
            for dataset in dict[arch][config].keys():
                model_list.append(f"{arch}_{config}_{dataset}")
    return sorted(model_list)


def get_model_versions(model: str) -> list:
    model_dir = _get_model_dir()
    assert model in available_models(), f"given model: {model} not in the registry"
    model_path = os.path.join(model_dir, model)

    versions = []
    for version in _get_list_of_dirs(model_path):
        versions.append(version)

    return sorted(versions)


def get_model_latest_version(model: str) -> int:
    return max(get_model_versions(model))


def get_saved_model(model: str, version: str):
    model_dir = os.path.join(_get_model_dir(), model, str(version))

    info_dict = _get_model_info(model, version)
    url = info_dict.get("url")

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        # time.sleep might help here
        pass

    parts = urlparse(url)
    filename_from_remote = os.path.basename(parts.path)
    # filename, file_extension = os.path.splitext(filename_from_remote)

    cached_file = os.path.join(model_dir, model + ".pth")

    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        _download_file_from_url(url, cached_file, progress=True)

    # ToDo: Make it for zip file
    pass


# APIs for Architecture


def available_archs() -> List[str]:
    archs_dir = _get_arch_dir()
    dirs = _get_list_of_dirs(archs_dir)
    return sorted(dirs)


def get_arch_configs(arch: str) -> List[str]:
    cls = _get_arch_cls(arch)
    configs = getattr(cls, "__CONFIGS__", {})
    return sorted(list(configs.keys()))
