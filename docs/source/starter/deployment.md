# Deployments <!-- omit in toc -->

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

## Build AI Service As Docker Image

From root directory of the repository run followings,

```bash
docker build -t satellighte-fastapi deployment/fastapi/
```

## Run AI Service As Docker Container

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
