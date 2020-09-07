#!/bin/bash
#export MODEL_DIR=/tmp/output/servetest-model/export/latest_model/
export MODEL_DIR=/tmp/tfr/model-petertoy-bertbase/export/latest_model/
#export MODEL_DIR=/home/peter/github/tensorflow/tensorflow_ranking_model/1599246704/
#export MODEL_DIR=/home/peter/github/tensorflow/tensorflow_ranking_model
#export MODEL_DIR=/home/peter/github/tensorflow/

#nohup tensorflow_model_server \
#  --rest_api_port=8501 \
#  --model_name=testmodel \
#  --model_base_path="${MODEL_DIR}" >server.log 2>&1

tensorflow_model_server \
  --rest_api_port=8501 \
  --model_name=tfrbert \
  --model_base_path="${MODEL_DIR}"

#tensorflow_model_server \
#  --rest_api_port=8501 \
#  --model_name=tfrbert \  
#  --model_base_path="${MODEL_DIR}"  
  
