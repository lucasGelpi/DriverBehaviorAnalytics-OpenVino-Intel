#!/bin/bash
SCRIPT_DIR=`dirname $0`
PIP=$1
PYTHON=$2
ENV_FOLDER=$3

# Install python virtual env module
sudo -A apt-get install -y python3-venv
# Create a virtual env
${PYTHON} -m venv ${ENV_FOLDER}
# Activate virtual env
source ${ENV_FOLDER}/bin/activate
# Install general dependencies
${PIP} install -r ${SCRIPT_DIR}/requirements.txt
${PIP} list