FROM python:3.7.11

RUN apt update && apt install -y pipenv
RUN mkdir app

WORKDIR /app

CMD ["bash"]