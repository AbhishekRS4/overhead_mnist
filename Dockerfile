FROM python:3.8.10-slim

WORKDIR /app

# install linux package dependencies
RUN apt-get update -y

# can copy files only from current working directory where docker builds
# cannot copy files from arbitrary directories

COPY ./trained_models/ /data/models/
COPY ./requirements_deployment.txt .

RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements_deployment.txt

COPY ./modeling/*.py ./modeling/
COPY ./*.py .
COPY ./*.json .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--reload"]
