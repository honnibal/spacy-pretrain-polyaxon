#!/usr/bin/env bash

cd spaCy
source env3.6/bin/activate
export PYTHONPATH=`pwd`
python ../pretrain_textcat.py 
