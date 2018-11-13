#!/usr/bin/env bash

python3.6 -m venv .env
source .env/bin/activate
pip install polyaxon-cli
polyaxon login
