"""Parse and print pbtxt config file"""
import argparse
from pipeline.proto.tfpipeline_options_pb2 import TfpipelineOptions
from google.protobuf import text_format


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse and print protobuf text options')
    parser.add_argument('pbtxt_filepath')
    args = parser.parse_args()
    options = TfpipelineOptions()
    with open(args.pbtxt_filepath) as f:
        txt = f.read()
        message = text_format.Parse(txt, TfpipelineOptions())
        print(message)

