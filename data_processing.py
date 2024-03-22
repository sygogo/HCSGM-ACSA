import argparse

from monai.utils import set_determinism
from transformers import set_seed
from src.data_processing.DataGenerator import DataGenerator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source', type=str, default="2014")
    parser.add_argument('--data_category', type=str, default="mams")
    parser.add_argument('--source_dir', type=str, default="data/raw")
    parser.add_argument('--target_dir', type=str, default="data/processed")
    parser.add_argument('--bert_directory', type=str, default="/data/bert-base-uncased")
    parser.add_argument('--encoder_type', type=str, default="bert")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    set_seed(seed=args.seed)
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    dg = DataGenerator(args.data_source, args.data_category, args.source_dir, args.target_dir, args.bert_directory, args.encoder_type)
    dg.processing()
