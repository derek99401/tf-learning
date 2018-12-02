import tensorflow as tf
from tfpipeline import TfPipeline, ActionType
from models.input_only_model import InputOnlyModel
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run tfpipeline')
    parser.add_argument('--input_files', required=True, help='comma separated input tfrecords')
    parser.add_argument('--max_iter', type=int, default=5e4, help='max iteration')
    parser.add_argument('--train', action='store_true', help='action type: run training')
    parser.add_argument('--test', action='store_true', help='action type: run testing')
    parser.add_argument('--print', action='store_true', help='action type: run printing')
    args = parser.parse_args()
    input_files = args.input_files.split(',')
    model = InputOnlyModel()
    tfpipeline = TfPipeline(model)
    if args.print:
        tfpipeline.run(ActionType.debug_print, input_files, 2, max_iter=args.max_iter)
    elif args.train:
        tfpipeline.run(ActionType.train, input_files, 2, max_iter=args.max_iter)
    elif args.test:
        tfpipeline.run(ActionType.test, input_files, 2, max_iter=args.max_iter)
    else:
        raise Exception('you must provide a action type, train, test or print')
