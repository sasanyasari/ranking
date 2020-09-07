#!/bin/bash
BERT_DIR="/home/peter/github/tensorflow/ranking/uncased_L-12_H-768_A-12_TF2"  && \
python tensorflow_ranking/extension/examples/tfrbert_convert_json_to_elwc.py \
    --vocab_file=${BERT_DIR}/vocab.txt \
    --sequence_length=128 \
    --input_file=/home/peter/github/peter-ranking/ranking/TFRBertExample-eval.json \
    --output_file=eval.toy.elwc.tfrecord \
    --do_lower_case 
