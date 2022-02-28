import os
from importlib.util import module_from_spec, spec_from_file_location
from typing import List

import setuptools

_PATH_ROOT = os.path.dirname(__file__)


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
    with open(os.path.join(_PATH_ROOT, file_name), "r", encoding="utf-8") as file:
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


def _load_py_module(fname, pkg="satellighte"):
    spec = spec_from_file_location(
        os.path.join(pkg, fname), os.path.join(_PATH_ROOT, pkg, fname)
    )
    py_module = module_from_spec(spec)
    spec.loader.exec_module(py_module)
    return py_module


about = _load_py_module("__about__.py")

__requirements__ = _load_requirements()

test_require = [
    "pytest>=6.0.0",
    "pytest-pylint",
    "pytest-cov",
    "pylint",
    "black",
]

deploy_require = [
    "pytest>=6.0.0",
    "pytest-pylint",
    "pytest-cov",
    "pylint",
    "black",
]

docs_require = [
    "sphinxemoji",
    "sphinx_rtd_theme",
    "recommonmark",
    "sphinx_markdown_tables",
    "sphinxcontrib-napoleon",
    "sphinx-autodoc-typehints",
    "nbsphinx",
    "furo",
]

extras_require = {
    "test": test_require,
    "docs": docs_require,
    "deploy": __requirements__ + deploy_require,
    "all": __requirements__ + test_require + deploy_require,
}
setuptools.setup(
    name=about.__pkg_name__,
    version=about.__version__,
    description=about.__description__,
    author=about.__author__,
    author_email=about.__author_email__,
    url=about.__homepage__,
    license=about.__license__,
    packages=setuptools.find_packages(),
    include_package_data=True,
    long_description=about.__long_description__,
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
    extras_require=extras_require,
    setup_requires=[],
    install_requires=__requirements__,
    classifiers=[
        # Indicate who your project is intended for
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        # Pick your license as you wish
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        # Specify the Python versions you support here.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
