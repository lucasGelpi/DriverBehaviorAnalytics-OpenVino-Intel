#!/bin/bash -e
USE_CASE=$1
TYPE=$2
MODEL=$3

REPO_FOLDER=$(pwd)
MODEL_REPO_LINK="https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.4/models_bin/3"
UDF_FOLDER=$REPO_FOLDER/config/VideoIngestion/udfs/$TYPE/$MODEL
MODEL_FOLDER=$UDF_FOLDER/model

if [ ! -f "$MODEL_FOLDER/$MODEL.xml"  ]; then
    echo "Downloading Model.. "
    curl $MODEL_REPO_LINK/$MODEL/FP32/$MODEL.xml \
     --create-dirs -o $MODEL_FOLDER/$MODEL.xml
fi

if [ ! -f "$MODEL_FOLDER/$MODEL.bin"  ]; then
    curl $MODEL_REPO_LINK/$MODEL/FP32/$MODEL.bin \
     --create-dirs -o $MODEL_FOLDER/$MODEL.bin
    echo "Model Downloaded"
fi

chmod -R 775 $UDF_FOLDER
