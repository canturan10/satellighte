FROM python:3.8

WORKDIR /satellighte_fastapi
COPY . satellighte_fastapi/

RUN pip install satellighte --no-cache-dir --upgrade
RUN pip install -r satellighte_fastapi/requirements.txt --no-cache-dir

CMD ["python", "satellighte_fastapi/service.py"]
