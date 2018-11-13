#!/usr/bin/env bash
set -e
# Install Python 3.6 and git
apt-get update -y
apt-get install -y curl git python-dev python3.6 python3.6-dev python3.6-venv python3.6-distutils build-essential

# Now setup the spaCy repo
git clone https://github.com/explosion/spaCy -b develop
cd spaCy
make
env3.6/bin/python3.6 -m spacy download en_vectors_web_lg
env3.6/bin/pip install polyaxon_helper
cd ..


# Install gcloud utility tools, so we can copy stuff from buckets.
curl https://dl.google.com/dl/cloudsdk/channels/rapid/install_google_cloud_sdk.bash > install_google_cloud_sdk.bash
bash ./install_google_cloud_sdk.bash --disable-prompts
source /root/google-cloud-sdk/path.bash.inc
# This is what the copy command will look like.
gsutil cp gs://galaxy-state/spacy.pex.tar.gz .
