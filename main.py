import argparse
import os
from transformers import set_seed, enable_full_determinism
from numpy.distutils.fcompiler import str2bool
from src.trainer.BaseTrainer import BaseTrainer
import warnings
import torch

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source', type=str, default="2014")
    parser.add_argument('--data_category', type=str, default="mams")
    parser.add_argument('--target_dir', type=str, default="data/processed")
    parser.add_argument('--model_dir', type=str, default="data/output")
    parser.add_argument('--bert_directory', type=str, default="/data/bert-base-uncased")
    parser.add_argument('--fix_bert_embeddings', type=str2bool, default=False)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--aspect_number', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--decoder_lr', type=float, default=5e-5)
    parser.add_argument('--encoder_lr', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--clip', type=float, default=1)
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--early_stopping', type=int, default=15)
    parser.add_argument('--model', type=str, default='Seq2Seq')
    parser.add_argument('--decoder_type', type=str, default='her_cov_decoder')
    parser.add_argument('--encoder_type', type=str, default='bert')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--decoder_num_layer', type=int, default=1)
    parser.add_argument('--set_loss', type=str2bool, default='true')
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    enable_full_determinism(seed=args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    if args.local_rank == -1:
        args.dist = False
    else:
        args.dist = True
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
    trainer = BaseTrainer(args)
    trainer.train()
