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
```

### ONNX Runtime

```bash
python deployment/onnx/runtime.py
```

## DeepSparse

Neural Magic's [DeepSparse](https://docs.neuralmagic.com/deepsparse/) Engine is able to integrate into popular deep learning libraries allowing you to leverage DeepSparse for loading and deploying sparse models with ONNX.

From root directory of the repository run followings,
We need the `ONNX` file to use it. [Create your onnx file from the above steps](#onnx). Next,

### Install Dependency For DeepSparse

```bash
pip install deepsparse~=1.0.2
```

### DeepSparse Runtime

```bash
python deployment/deepsparse/runtime.py
```
