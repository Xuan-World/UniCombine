import argparse
import os.path

import ipdb
from datasets import load_dataset


def filter_test_dataset(example):
    if example["quality_assessment"] is not None:
        scores = list(example["quality_assessment"].values())
        if example["quality_assessment"]['compositeStructure']>=3 and example["quality_assessment"]['imageQuality']==5 and not all(score == 5 for score in scores) and example['quality_assessment']['objectConsistency']==5:
            return True
        else:
            return False
    else:
        return False

def filter_train_dataset(example):
    if example["quality_assessment"] is not None:
        return list(example["quality_assessment"].values()) == [5, 5, 5]
    else:
        return False

def parse_args():
    parser = argparse.ArgumentParser("partition dataset")
    parser.add_argument("--dataset", type=str, default=None,required=True)
    parser.add_argument("--output_dir", type=str, default=None,required=True)
    parser.add_argument("--partition", type=str, default=None,required=True,choices=["train","test"])
    parser.add_argument("--num_shards", type=int, default=None)
    parser.add_argument("--num_proc", type=int, default=32)
    parser.add_argument("--cache", type=str, default="cache")
    args = parser.parse_args()
    if args.num_shards is None and args.partition == "train":
        args.num_shards = len(os.listdir(args.dataset))
    elif args.num_shards is None and args.partition == "test":
        args.num_shards = 1
    args.output_dir = os.path.join(args.output_dir, args.partition)
    return args

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = load_dataset(args.dataset, split="train", cache_dir=args.cache)
    if args.partition == "train":
        filtered_dataset = dataset.filter(filter_train_dataset,num_proc =args.num_proc)
    elif args.partition == "test":
        filtered_dataset = dataset.filter(filter_test_dataset,num_proc =args.num_proc)
    output_path = os.path.join(args.output_dir,"data-{index:05d}-of-{num_shards:05d}.parquet")
    for index in range(args.num_shards):
        shard = filtered_dataset.shard(index=index, num_shards=args.num_shards, contiguous=True)
        shard.to_parquet(output_path.format(index=index,num_shards=args.num_shards))
