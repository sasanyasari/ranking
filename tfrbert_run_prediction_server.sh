#!/bin/bash
export MODEL_DIR=/tmp/tfr/model-petertoy-bertbase/export/latest_model/
tensorflow_model_server \
  --rest_api_port=8501 \
  --model_name=tfrbert \
  --model_base_path="${MODEL_DIR}"

