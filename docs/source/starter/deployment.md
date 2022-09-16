<p align="center">
    <img src="https://raw.githubusercontent.com/canturan10/satellighte/master/src/deployment.png" align="center" alt="Deployment" />

# Deployments <!-- omit in toc -->

- [FastAPI](#fastapi)
  - [Install Dependency For FastAPI](#install-dependency-for-fastapi)
  - [Run AI Service](#run-ai-service)
  - [Build AI Service As Docker Image](#build-ai-service-as-docker-image)
  - [Run AI Service As Docker Container](#run-ai-service-as-docker-container)
- [ONNX](#onnx)
  - [Install Dependency For ONNX](#install-dependency-for-onnx)
  - [Convert Model to ONNX](#convert-model-to-onnx)
  - [ONNX Runtime](#onnx-runtime)
- [DeepSparse](#deepsparse)
  - [Install Dependency For DeepSparse](#install-dependency-for-deepsparse)
  - [DeepSparse Runtime](#deepsparse-runtime)
- [TensorFlow](#tensorflow)
  - [Install Dependency For TensorFlow](#install-dependency-for-tensorflow)
  - [Convert ONNX Model to TensorFlow](#convert-onnx-model-to-tensorflow)
  - [TensorFlow Runtime](#tensorflow-runtime)
- [TensorFlow Lite](#tensorflow-lite)
  - [Install Dependency For TensorFlow Lite](#install-dependency-for-tensorflow-lite)
  - [Convert TensorFlow Model to TensorFlow Lite](#convert-tensorflow-model-to-tensorflow-lite)
  - [TensorFlow Lite Runtime](#tensorflow-lite-runtime)

## FastAPI

[FastAPI](https://fastapi.tiangolo.com/) wraps the `Satellighte` library to serve as RESTful API

From root directory of the repository run followings,

### Install Dependency For FastAPI

```bash
pip install fastapi==0.74.1
pip install "uvicorn[standard]"==0.17.5
pip install python-multipart
```

### Run AI Service

```bash
python deployment/fastapi/service.py
```

### Build AI Service As Docker Image

From root directory of the repository run followings,

```bash
docker build -t satellighte-fastapi deployment/fastapi/
```

### Run AI Service As Docker Container

if gpu enabled, run with

```bash
docker run -d --name satellighte-service --rm -p 8080:8080 --gpus all satellighte-fastapi
```

if gpu disabled, run with

```bash
docker run -d --name satellighte-service --rm -p 8080:8080 satellighte-fastapi
```

## ONNX

[ONNX Runtime](https://onnxruntime.ai/) inference can lead to faster customer experiences and lower costs.

From root directory of the repository run followings,

### Install Dependency For ONNX

```bash
pip install onnx~=1.11.0
pip install onnxruntime~=1.10.0
```

### Convert Model to ONNX

```bash
python deployment/onnx/export.py
#  python deployment/onnx/export.py --model_name mobilenetv2_default_eurosat --version 0
```

### ONNX Runtime

```bash
python deployment/onnx/runtime.py
#  python deployment/onnx/runtime.py -m satellighte/models/mobilenetv2_default_eurosat/v0/mobilenetv2_default_eurosat.onnx -s src/eurosat_samples/AnnualCrop.jpg
```

## DeepSparse

Neural Magic's [DeepSparse](https://docs.neuralmagic.com/deepsparse/) Engine is able to integrate into popular deep learning libraries allowing you to leverage DeepSparse for loading and deploying sparse models with ONNX.

From root directory of the repository run followings. We need the `ONNX` model to use it. [Create your onnx model from the above steps](#onnx). Next,

### Install Dependency For DeepSparse

```bash
pip install deepsparse~=1.0.2
```

### DeepSparse Runtime

```bash
python deployment/deepsparse/runtime.py
# python deployment/deepsparse/runtime.py -m -m satellighte/models/mobilenetv2_default_eurosat/v0/mobilenetv2_default_eurosat.onnx -s src/eurosat_samples/AnnualCrop.jpg
```

## TensorFlow

[TensorFlow](https://www.tensorflow.org/) is a free and open-source software library for machine learning and artificial intelligence.

From root directory of the repository run followings,

### Install Dependency For TensorFlow

```bash
pip install onnx-tf~=1.10.0
pip install tensorflow~=2.9.1
pip install tensorflow-probability~=0.17.0
```

From root directory of the repository run followings. We need the `ONNX` model to use it. [Create your onnx model from the above steps](#onnx). Next,

### Convert ONNX Model to TensorFlow

```bash
python deployment/tensorflow/export.py
# python deployment/tensorflow/export.py -m satellighte/models/mobilenetv2_default_eurosat/v0/mobilenetv2_default_eurosat.onnx
```

### TensorFlow Runtime

```bash
python deployment/tensorflow/runtime.py
# python deployment/tensorflow/runtime.py -m satellighte/models/mobilenetv2_default_eurosat/v0/mobilenetv2_default_eurosat_tensorflow  -s src/eurosat_samples/AnnualCrop.jpg -l "AnnualCrop,PermanentCrop,Forest,HerbaceousVegetation,Highway,Industrial,Pasture,Residential,River,SeaLake"
```

## TensorFlow Lite

[TensorFlow Lite](https://www.tensorflow.org/lite) is a mobile library for deploying models on mobile, microcontrollers and other edge devices.

From root directory of the repository run followings,

### Install Dependency For TensorFlow Lite

```bash
pip install onnx-tf~=1.10.0
pip install tensorflow~=2.9.1
pip install tensorflow-probability~=0.17.0
```

From root directory of the repository run followings. We need the `TensorFlow` model to use it. [Create your tensorflow model from the above steps](#tensorflow). Next,

### Convert TensorFlow Model to TensorFlow Lite

```bash
python deployment/tensorflow_lite/export.py
# python deployment/tensorflow_lite/export.py -m satellighte/models/mobilenetv2_default_eurosat/v0/mobilenetv2_default_eurosat_tensorflow
```

### TensorFlow Lite Runtime

```bash
python deployment/tensorflow_lite/runtime.py
# python deployment/tensorflow_lite/runtime.py -m satellighte/models/mobilenetv2_default_eurosat/v0/mobilenetv2_default_eurosat_tensorflow.tflite -s satellighte/src/eurosat_samples/AnnualCrop.jpg -l "AnnualCrop,PermanentCrop,Forest,HerbaceousVegetation,Highway,Industrial,Pasture,Residential,River,SeaLake"
```
