# DEPLOYMENTS <!-- omit in toc -->

- [FastAPI](#fastapi)
	- [Run AI Service](#run-ai-service)
		- [Install Dependency](#install-dependency)
		- [Run AI Service](#run-ai-service-1)
	- [Build AI Service As Docker Image](#build-ai-service-as-docker-image)
	- [Run AI Service As Docker Container](#run-ai-service-as-docker-container)

## FastAPI

[FastAPI](https://fastapi.tiangolo.com/) wraps the `Satellighte` library to serve as RESTful API

### Run AI Service

From root directory of the repository run followings,

#### Install Dependency

```bash
pip install fastapi==0.74.1
pip install "uvicorn[standard]"==0.17.5
pip install python-multipart
```

or

```bash
pip install -r deployment/fastapi/requirements.txt
```

#### Run AI Service

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
