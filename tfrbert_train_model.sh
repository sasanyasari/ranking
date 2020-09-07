#!/bin/bash
#BERT_DIR="/home/peter/github/tensorflow/ranking/uncased_L-4_H-256_A-4_TF2"  && \
BERT_DIR="/home/peter/github/tensorflow/ranking/uncased_L-12_H-768_A-12_TF2"  && \
OUTPUT_DIR="/tmp/tfr/model-petertoy-bertbase/" && \
DATA_DIR="/home/peter/github/peter-ranking/ranking" && \
rm -rf "${OUTPUT_DIR}" && \
bazel build -c opt \
tensorflow_ranking/extension/examples:tfrbert_example_py_binary && \
./bazel-bin/tensorflow_ranking/extension/examples/tfrbert_example_py_binary \
   --train_input_pattern=${DATA_DIR}/train.toy.elwc.tfrecord \
   --eval_input_pattern=${DATA_DIR}/eval.toy.elwc.tfrecord \
   --bert_config_file=${BERT_DIR}/bert_config.json \
   --bert_init_ckpt=${BERT_DIR}/bert_model.ckpt \
   --bert_max_seq_length=128 \
   --model_dir="${OUTPUT_DIR}" \
   --list_size=15 \
   --loss=softmax_loss \
   --train_batch_size=1 \
   --eval_batch_size=1 \
   --learning_rate=1e-5 \
   --num_train_steps=500 \
   --num_eval_steps=10 \
   --checkpoint_secs=500 \
   --num_checkpoints=2
