import time
from typing import List

from satellighte.version import __version__ as pkg_version


def _load_requirements(
    file_name: str = "requirements.txt", comment_char: str = "#"
) -> List[str]:
    with open(file_name, "r", encoding="utf-8") as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[: ln.index(comment_char)].strip()
        # skip directly installed dependencies
        if ln.startswith("http") or "@http" in ln:
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs


def _load_readme(file_name: str = "README.md") -> str:

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
