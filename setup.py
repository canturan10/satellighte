import setuptools
import os
from importlib.util import module_from_spec, spec_from_file_location

_PATH_ROOT = os.path.dirname(__file__)


def _load_py_module(fname, pkg="satellighte"):
    spec = spec_from_file_location(
        os.path.join(pkg, fname), os.path.join(_PATH_ROOT, pkg, fname)
    )
    py = module_from_spec(spec)
    spec.loader.exec_module(py)
    return py


about = _load_py_module("__about__.py")

setuptools.setup(
    name=about.__name__,
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
    setup_requires=[],
    install_requires=about.__requirements__,
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
