FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye as base

RUN apt update

RUN apt-get update && apt-get upgrade -y \
    && apt-get -y install --no-install-recommends pandoc

# Install poetry
RUN pipx install poetry && pipx inject poetry poetry-export
