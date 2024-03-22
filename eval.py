import argparse
import gc
import os

import numpy as np
import torch
from monai.utils import set_determinism
from numpy.distutils.fcompiler import str2bool
from src.trainer.BaseTrainer import BaseTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source', type=str, default="2014")
    parser.add_argument('--data_category', type=str, default="restaurant")
    parser.add_argument('--target_dir', type=str, default="data/processed")
    parser.add_argument('--model_dir', type=str, default="data/output")
    parser.add_argument('--bert_directory', type=str, default="/data/bert-base-uncased")
    parser.add_argument('--fix_bert_embeddings', type=str2bool, default=False)
    parser.add_argument('--aspect_number', type=int, default=5)
    # parser.add_argument('--seeds', nargs='+', default=[1433])
    parser.add_argument('--seeds', nargs='+', default=[ 44])
    # parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', type=str, default='eval')
    parser.add_argument('--model', type=str, default='Seq2Seq')
    parser.add_argument('--decoder_type', type=str, default='her_cov_decoder')
    parser.add_argument('--encoder_type', type=str, default='bert')
    parser.add_argument('--decoder_lr', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--decoder_num_layer', type=int, default=1)
    parser.add_argument('--verbose', type=str2bool, default='False')
    parser.add_argument('--set_loss', type=str2bool, default='true')
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    total_results = {}
    print('=======================================args==========================================')
    print(args)
    for args.seed in args.seeds:
        print(
            '=======================================Start Evaluation:seed[{}]=========================================='.format(
                args.seed))
        trainer = BaseTrainer(args)
        results = trainer.eval(verbose=args.verbose)
        for arg in results:
            if arg not in total_results:
                total_results[arg] = [results[arg]]
            else:
                total_results[arg].append(results[arg])
            print(arg + ":" + str(results[arg]))
    print('=======================================Summary==========================================')
    precision_all, recall_all = 0, 0
    precision_single, recall_single = 0, 0
    precision_mul, recall_mul = 0, 0
    for key, v in total_results.items():
        if key in ["f1_all", "f1_mul", "f1_single"]:
            continue
        print('avg {}:{}({})'.format(str(key), str(round(np.mean(v) * 100, 3)), str(round(np.std(v) * 100, 3))))
        if key == "precision_all":
            precision_all = round(np.mean(v) * 100, 3)
        if key == "recall_all":
            recall_all = round(np.mean(v) * 100, 3)
        if key == "precision_single":
            precision_single = round(np.mean(v) * 100, 3)
        if key == "recall_single":
            recall_single = round(np.mean(v) * 100, 3)
        if key == "precision_mul":
            precision_mul = round(np.mean(v) * 100, 3)
        if key == "recall_mul":
            recall_mul = round(np.mean(v) * 100, 3)

    print("f1_all:{}".format(str(round(2 * precision_all * recall_all / (precision_all + recall_all), 3))))
    if args.data_category == "restaurant":
        print("f1_single:{}".format(str(round(2 * precision_single * recall_single / (precision_single + recall_single), 3))))
        print("f1_mul:{}".format(str(round(2 * precision_mul * recall_mul / (precision_mul + recall_mul), 3))))
