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

# Put together from https://colab.research.google.com/github/tensorflow/ranking/blob/master/tensorflow_ranking/examples/handling_sparse_features.ipynb#scrollTo=eE7hpEBBykVS
# and based on the unit test at https://github.com/tensorflow/ranking/blob/master/tensorflow_ranking/extension/examples/tfrbert_example_test.py    


class TFRBertUtilJSON(object):

    def __init__(self, TFRBertUtil):
        self.TFRBertUtilHelper = TFRBertUtil        

    # Helper function for making toy data
    # Out: Returns toy ranking data data in ELWC format
    def mkToyRankingRecord(self):
        query = "Where can you buy cat food?"
        documents = ["The pet food store", "Bicycles have two wheels", "The grocery store", "Cats eat cat food"]
        labels = [3, 1, 3, 2]
        label_name = "relevance"
        elwcOut = self.TFRBertUtilHelper.convert_to_elwc(query, documents, labels, label_name)

        # (DEBUG) Examine the structure of the ELWC records for tfr-bert
        # print(elwcOut)

        return elwcOut


    # Conversion function for converting easily-read JSON into TFR-Bert's ELWC format, and exporting to file
    # In: filename of JSON with ranking problems (see example json files for output)
    # Out: creates TFrecord output file, also returns list of ranking problems read in from JSON
    def convert_json_to_elwc_export(self, filenameJsonIn, filenameTFRecordOut):
        # Step 1: Convert JSON to ELWC
        (listToRank, listJsonRaw) = self.convert_json_to_elwc(filenameJsonIn)

        # Step 2: Save ELWC to file
        try:
            with tf.io.TFRecordWriter(filenameTFRecordOut) as writer:
                for example in listToRank * 10:
                    writer.write(example.SerializeToString())
        except:
            print("convert_json_to_elwc_export: error writing ELWC file (filename = " + filenameTFRecordOut + ")")
            exit(1)
        
        # Step 3: Also return ranking problem in JSON format, for use in scoring/exporting
        return listJsonRaw


    # Conversion function for converting easily-read JSON into TFR-Bert's ELWC format
    # In: JSON filename
    # Out: List of ELWC records, list of original JSON records
    def convert_json_to_elwc(self, filenameJsonIn):
        # Step 1: Load JSON file
        listToRankELWC = []
        listJsonRaw = []
#        try:
        with open(filenameJsonIn) as json_file:
            # Load whole JSON file
            data = json.load(json_file)

            # Parse each record
            for rankingProblem in data['rankingProblems']:
                labels = []
                docTexts = []

                queryText = rankingProblem['queryText']
                documents = rankingProblem['documents']
                for document in documents:
                    docText = document['docText']
                    docRel = document['relevance']      # Convert to int?

                    labels.append(docRel)
                    docTexts.append(docText)

                # Step 1A: Convert this record to ELWC
                elwcOut = self.TFRBertUtilHelper.convert_to_elwc(queryText, docTexts, labels, label_name="relevance")
                listToRankELWC.append(elwcOut)
                # Step 1B: Also store the raw record, for exporting output in the same format it was read in
                listJsonRaw.append(rankingProblem)

#        except:
#            print("convert_json_to_elwc_export: error loading JSON file (filename = " + filenameJsonIn + ")")
#            exit(1)

        return (listToRankELWC, listJsonRaw)
        

class TFRBertClient(object):    
    # Default/Example values
    # self.grpcChannel = "0.0.0.0:8500"                   # from the Tensorflow Serving server
    # self.modelName = "tfrbert"
    # self.servingSignatureName = "serving_default"       # 'serving_default' instead of 'predict', as per saved_model_cli tool (see https://medium.com/@yuu.ishikawa/how-to-show-signatures-of-tensorflow-saved-model-5ac56cf1960f )    
    # self.timeoutInSecs = 3


    def __init__(self, grpcChannel, modelName, servingSignatureName, timeoutInSecs):        
        self.grpcChannel = grpcChannel
        self.modelName = modelName
        self.servingSignatureName = servingSignatureName        
        self.timeoutInSecs = timeoutInSecs     


    # Send a gRPC request to the Tensorflow Serving model server to generate predictions for a single ranking problem
    # Based on https://github.com/tensorflow/ranking/issues/189
    def generatePredictions(self, rankingProblemELWC, rankingProblemJSONIn):                
        # Make a deep copy of the ranking problem
        rankingProblemJSON = copy.deepcopy(rankingProblemJSONIn)

        # Pack problem
        example_list_with_context_proto = rankingProblemELWC.SerializeToString()
        tensor_proto = tf.make_tensor_proto(example_list_with_context_proto, dtype=tf.string, shape=[1])
        
        # Set up request to prediction server
        request = predict_pb2.PredictRequest()
        request.inputs['input_ranking_data'].CopyFrom(tensor_proto)
        request.model_spec.signature_name = self.servingSignatureName   
        request.model_spec.name = self.modelName
        channel = grpc.insecure_channel(self.grpcChannel)
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)                            

        # Make prediction request and get response
        grpc_response = stub.Predict(request, self.timeoutInSecs)
        unpacked_grpc_response = MessageToDict(grpc_response, preserving_proto_field_name = True)

        # Add model's ranking scores to each document
        docScores = unpacked_grpc_response['outputs']['output']['float_val']
        for docIdx in range(0, len(rankingProblemJSON['documents'])):
            rankingProblemJSON['documents'][docIdx]['score'] = docScores[docIdx]

        # Sort documents in descending order based on docScore
        rankingProblemJSON['documents'].sort(key=lambda x:x['score'], reverse=True)

        # (DEBUG) Print out ranking problem with scores added to documents
        # print(rankingProblemJSON)

        # Return ranking problem with document scores added
        return rankingProblemJSON
        

    # In: Parallel lists of ranking problems in ELWC and JSON format
    # Out: list of ranking problems in JSON format, with document scores from the model added, and documents sorted in descending order based on docScores. 
    def generatePredictionsList(self, rankingProblemsELWC, rankingProblemsJSON):
        rankingProblemsOut = []

        # Iterate through each ranking problem, generating document scores
        for idx in range(0, len(rankingProblemsELWC)):
            rankingProblemsOut.append( self.generatePredictions(rankingProblemsELWC[idx], rankingProblemsJSON[idx]) )

        return rankingProblemsOut
    

    def exportRankingOutput(self, filenameJSONOut, rankingProblemOutputJSON):
        dataOut = {"rankingProblemsOutput": rankingProblemOutputJSON}
        strOut = json.dumps(dataOut, indent=4)
        print(strOut)



#
#   Supporting Functions
#

# Adapted from TfrBertUtilTest (tfrbert_test.py)
# Creates a TFRBertUtil object, primarily used to convert ranking problems from plain text to the BERT representation packed into the ELWC format
# used by TFR-Bert.
def create_tfrbert_util_with_vocab(bertMaxSeqLength, bertVocabFile, do_lower_case):
    return tfrbert.TFRBertUtil(
            bert_config_file=None,
            bert_init_ckpt=None,
            bert_max_seq_length=bertMaxSeqLength,
            bert_vocab_file=bertVocabFile,
            do_lower_case=do_lower_case)

#
#   Main
#

if __name__ == "__main__":

    # Parameters
    bertVocabFilename = "/home/peter/github/tensorflow/ranking/uncased_L-4_H-256_A-4_TF2/vocab.txt"
    do_lower_case = True

    # Create helpers
    bert_helper = create_tfrbert_util_with_vocab(128, bertVocabFilename, do_lower_case)
    bert_helper_json = TFRBertUtilJSON(bert_helper)

    # Example of converting from JSON to ELWC
    #filenameJsonIn = "/home/peter/github/peter-ranking/ranking/jsonInExample-eval.json"
    # filenameELWCOut = "eval.toy.elwc.tfrecord"
    # bert_helper_json.convert_json_to_elwc_export(filenameJsonIn, filenameELWCOut)

    # exit(1)



    filenameJsonIn = "/home/peter/github/peter-ranking/ranking/jsonInExample-train.json"
    (rankingProblemsELWC, rankingProblemsJSON) = bert_helper_json.convert_json_to_elwc(filenameJsonIn)

    tfrBertClient = TFRBertClient(grpcChannel = "0.0.0.0:8500", modelName = "tfrbert", servingSignatureName = "serving_default", timeoutInSecs = 3)
    rankingProblemsOut = tfrBertClient.generatePredictionsList(rankingProblemsELWC, rankingProblemsJSON)

    print("Input filename: " + filenameJsonIn)

    for idx in range(0, len(rankingProblemsOut)):    
        print("Ranking Problem " + str(idx) + ":\n")
        print(rankingProblemsOut[idx])
        print("\n")


    tfrBertClient.exportRankingOutput("", rankingProblemsOut)