#!/bin/bash
BERT_DIR="/home/peter/github/tensorflow/ranking/uncased_L-12_H-768_A-12_TF2"  && \
python tensorflow_ranking/extension/examples/tfrbert_client_predict_from_json.py \
    --vocab_file=${BERT_DIR}/vocab.txt \
    --sequence_length=128 \
    --input_file=TFRBertExample-train.json \
    --output_file=train.scoresOut.json \
    --do_lower_case 

python tensorflow_ranking/extension/examples/tfrbert_client_predict_from_json.py \
    --vocab_file=${BERT_DIR}/vocab.txt \
    --sequence_length=128 \
    --input_file=TFRBertExample-eval.json \
    --output_file=eval.scoresOut.json \
    --do_lower_case 
    
