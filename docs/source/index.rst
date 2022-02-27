.. satellighte documentation master file, created by
   sphinx-quickstart on Sat Feb 19 00:32:16 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

====================================
|:satellite:| Satellighte Documentation
====================================
**Version**: |release|

.. toctree::
   :maxdepth: 4
   :name: starter
   :caption: Getting Started

   starter/about.md
   starter/prerequisites.md
   starter/installation.md

.. toctree::
   :maxdepth: 1
   :name: api
   :caption: Satellighte API

   api/api.rst
   api/module.rst
   api/datasets.rst

.. toctree::
   :maxdepth: 1
   :name: deployment
   :caption: Deployment

   deployment/fastapi.rst
   deployment/onnx_export.rst
   deployment/onnx_runtime.rst