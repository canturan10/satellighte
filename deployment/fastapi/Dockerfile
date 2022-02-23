FROM python:3.8

WORKDIR /code

RUN pip install satellighte --no-cache-dir --upgrade
RUN pip install -r deployment/fastapi/requirements.txt --no-cache-dir

COPY . code/

CMD ["python", "code/service.py"]
