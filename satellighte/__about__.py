import time
import os

from satellighte.version import __version__ as pkg_version

_PATH_ROOT = os.path.join(os.path.dirname(__file__), "..")


def _load_readme(file_name: str = "README.md") -> str:
    """
    Load readme from a text file.

    Args:
        file_name (str, optional): File name that contains the readme. Defaults to "README.md".

    Returns:
        str: Readme text.
    """
    with open(os.path.join(_PATH_ROOT, file_name), "r", encoding="utf-8") as file:
        readme = file.read()

    return readme


_this_year = time.strftime("%Y")
__author__ = "Oguzcan Turan"
__author_email__ = "can.turan.10@gmail.com"
__copyright__ = f"Copyright (c) 2022-{_this_year}, {__author__}."
__description__ = (
    "PyTorch Lightning Implementations of Recent Satellite Image Classification !"
)
__homepage__ = "https://github.com/canturan10/satellighte"
__license__ = "MIT License"
__license_url__ = __homepage__ + "/blob/master/LICENSE"
__long_description__ = _load_readme()
__pkg_name__ = "satellighte"
__version__ = pkg_version

__all__ = [
    "__author__",
    "__author_email__",
    "__copyright__",
    "__description__",
    "__homepage__",
    "__license__",
    "__license_url__",
    "__long_description__",
    "__pkg_name__",
    "__version__",
]
