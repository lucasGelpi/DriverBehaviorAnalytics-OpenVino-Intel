MODEL=$1

REPO_FOLDER=$(pwd)
MODEL_REPO_LINK="https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.4/models_bin/3"
MODEL_FOLDER=./models

if [ ! -f "$MODEL_FOLDER/$MODEL.xml"  ]; then
    echo "Downloading Models.. "
    curl $MODEL_REPO_LINK/$MODEL/FP32/$MODEL.xml \
     --create-dirs -o $MODEL_FOLDER/$MODEL.xml
fi

if [ ! -f "$MODEL_FOLDER/$MODEL.bin"  ]; then
    curl $MODEL_REPO_LINK/$MODEL/FP32/$MODEL.bin \
     --create-dirs -o $MODEL_FOLDER/$MODEL.bin
    echo "Model Downloaded"
fi
