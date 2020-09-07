# tfrbert_convert_json_to_elwc.py

from tensorflow_ranking.extension import tfrbert
import json
import copy
import tfrbert_client_predict_from_json
import argparse

#
#   Main
#

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_file", type=str, required=True, help="/path/to/bert_model/vocab.txt")
    parser.add_argument("--sequence_length", type=int, required=True, help="typically 128, 256, 512")    
    parser.add_argument("--input_file", type=str, required=True, help="JSON input filename (e.g. train.json)")
    parser.add_argument("--output_file", type=str, required=True, help="ELWC TFrecord filename (e.g. train.elwc.tfrecord)")

    parser.add_argument("--do_lower_case", action="store_true", help="Set for uncased models, otherwise do not include")

    args = parser.parse_args()

    # Create helpers
    bert_helper = tfrbert_client_predict_from_json.create_tfrbert_util_with_vocab(args.sequence_length, args.vocab_file, args.do_lower_case)
    bert_helper_json = tfrbert_client_predict_from_json.TFRBertUtilJSON(bert_helper)

    # User output
    print("Utility to convert between JSON and ELWC for TFR-Bert")
    print("")
    print("Model Parameters: ")
    print("Vocabulary filename: " + args.vocab_file)
    print("sequence_length: " + str(args.sequence_length))
    print("do_lower_case: " + str(args.do_lower_case))

    print("\n")
    print("Input file:  " + args.input_file)
    print("Output file: " + args.output_file)

    # Perform conversion of ranking problemsJSON to ELWC
    bert_helper_json.convert_json_to_elwc_export(args.input_file, args.output_file)

    print("Success.")



if __name__ == "__main__":
    main()

