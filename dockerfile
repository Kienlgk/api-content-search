FROM nvcr.io/nvidia/pytorch:21.02-py3

RUN mkdir /app
WORKDIR /app

RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN pip install transformers==4.9.1


ENTRYPOINT ["/bin/bash"]