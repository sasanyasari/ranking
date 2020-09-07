# tfrbert_client_example.py

from absl import flags
import tensorflow as tf
import tensorflow_ranking as tfr
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import input_pb2
from google.protobuf import text_format
from google.protobuf.json_format import MessageToDict

from tensorflow_ranking.extension import tfrbert
import json
import copy
import tfrbert_client_json 


#
#   Main
#

# Parameters (model)
bertVocabFilename = "/home/peter/github/tensorflow/ranking/uncased_L-4_H-256_A-4_TF2/vocab.txt"
do_lower_case = True
sequence_length = 128

# Parameters (in/out)
filenameJsonIn = "/home/peter/github/peter-ranking/ranking/jsonInExample-eval.json"
filenameELWCOut = "eval.toy.elwc.tfrecord"

# Create helpers
bert_helper = tfrbert_client_json.create_tfrbert_util_with_vocab(sequence_length, bertVocabFilename, do_lower_case)
bert_helper_json = tfrbert_client_json.TFRBertUtilJSON(bert_helper)

# User output
print("Model Parameters: ")
print("Vocabulary filename: " + bertVocabFilename)
print("sequence_length: " + str(sequence_length))
print("do_lower_case: " + str(do_lower_case))

print("\n")
print("Input file:  " + filenameJsonIn)
print("Output file: " + filenameELWCOut)


# Example of converting from JSON to ELWC
bert_helper_json.convert_json_to_elwc_export(filenameJsonIn, filenameELWCOut)

print("Success.")
