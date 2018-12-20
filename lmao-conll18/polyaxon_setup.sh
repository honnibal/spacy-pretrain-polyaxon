#!/usr/bin/env bash

set -e

# Install Python 3.6 and wget
apt-get update -y
apt-get install -y wget curl python3.6 python3.6-venv python3.6-dev python3.6-distutils build-essential

# Download UD treebanks
wget https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2895/ud-treebanks-v2.3.tgz
tar -xzf ud-treebanks-v2.3.tgz
mkdir -p parses/
mkdir -p models/

python3.6 -m venv env3.6
source env3.6/bin/activate
pip install -U pip
pip install "spacy-nightly==2.1.0a4" polyaxon-helper
python -m spacy download en_vectors_web_lg
