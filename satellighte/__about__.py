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
    for line in lines:

        # Disregard comments
        if comment_char in line:
            line = line[: line.index(comment_char)].strip()

        # Disregard http or @http lines
        if line.startswith("http") or "@http" in line:
            continue

        # Add the line to the list
        if line:
            reqs.append(line)
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
__license_url__ = __homepage__ + "/blob/master/LICENSE"
__long_description__ = _load_readme()
__pkg_name__ = "satellighte"
__requirements__ = _load_requirements()
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
    "__requirements__",
    "__version__",
]
