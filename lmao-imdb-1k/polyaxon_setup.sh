#!/usr/bin/env bash

set -e
# Install Python 3.6 
apt-get update -y
apt-get install -y wget curl python3.6 python3.6-venv python3.6-dev python3.6-distutils build-essential

python3.6 -m venv env3.6
source env3.6/bin/activate
pip install -U pip
pip install "spacy-nightly==2.1.0a3" polyaxon-helper
