import time
from typing import List

from satellighte.version import __version__ as pkg_version


def _load_requirements(
    file_name: str = "requirements.txt", comment_char: str = "#"
) -> List[str]:
    """
    Load requirements from a text file.

    Args:
        file_name (str, optional): File name. Defaults to "requirements.txt".
        comment_char (str, optional): Disregard lines starting with this character. Defaults to "#".

    Returns:
        List[str]: List of requirements.
    """
    # Open the file
    with open(file_name, "r", encoding="utf-8") as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:

        # Disregard comments
        if comment_char in ln:
            ln = ln[: ln.index(comment_char)].strip()

        # Disregard http or @http lines
        if ln.startswith("http") or "@http" in ln:
            continue

        # Add the line to the list
        if ln:
            reqs.append(ln)
    return reqs


def _load_readme(file_name: str = "README.md") -> str:
    """
    Load readme from a text file.

    Args:
        file_name (str, optional): File name that contains the readme. Defaults to "README.md".

    Returns:
        str: Readme text.
    """
    text = open(file_name, "r", encoding="utf-8").read()

    return text


_this_year = time.strftime("%Y")
__author__ = "Oguzcan Turan"
__author_email__ = "can.turan.10@gmail.com"
__copyright__ = f"Copyright (c) 2022-{_this_year}, {__author__}."
__description__ = (
    "PyTorch Lightning Implementations of Recent Satellite Image Classification !"
)
__homepage__ = "https://github.com/canturan10/satellighte"
__license__ = "MIT License"
__long_description__ = _load_readme()
__name__ = "satellighte"
__requirements__ = _load_requirements()
__version__ = pkg_version

__all__ = [
    "__author__",
    "__author_email__",
    "__copyright__",
    "__description__",
    "__homepage__",
    "__license__",
    "__long_description__",
    "__name__",
    "__requirements__",
    "__version__",
]
