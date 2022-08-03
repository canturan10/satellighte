.. satellighte documentation master file, created by
   sphinx-quickstart on Sat Feb 19 00:32:16 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

|:satellite:| Satellighte Documentation
=============================================

Satellite Image Classification
---------------------------------------------

**Satellighte** is an image classification library  that consist state-of-the-art deep learning methods. It is a combination of the words **'Satellite'** and **'Light'**, and its purpose is to establish a light structure to classify satellite images, but to obtain robust results.

:|:satellite:| Pypi: `satellighte <https://pypi.org/project/satellighte/>`_
:|:flying_saucer:| Version: |release|
:|:artificial_satellite:| Pages:
   - |:small_airplane:| `Project Page <https://canturan10.github.io/satellighte>`_
   - |:airplane:| `Github Page <https://github.com/canturan10/satellighte>`_
   - |:rocket:| `Demo Page <https://share.streamlit.io/canturan10/satellighte-streamlit/app.py>`_

.. toctree::
   :maxdepth: 2
   :name: starter
   :caption: Getting Started

   starter/about.md
   starter/prerequisites.md
   starter/installation.md
   starter/archs.md
   starter/datasets.md
   starter/deployment.md

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
   deployment/deepsparse.rst
   deployment/tensorflow_export.rst
   deployment/tensorflow_runtime.rst
   deployment/tensorflow_lite_export.rst
   deployment/tensorflow_lite_runtime.rst