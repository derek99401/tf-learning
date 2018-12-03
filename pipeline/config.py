from pipeline.proto.tfpipeline_options_pb2 import TfpipelineOptions

if __name__ == '__main__':
    options = TfpipelineOptions()
    options.max_iter = 1000
    print(options)
