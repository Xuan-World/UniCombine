import os.path

import ipdb
from datasets import load_dataset

dataset = load_dataset("dataset/SubjectSpatial200K/data_labeled",split="train",cache_dir="cache")

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


train_dataset = dataset.filter(filter_test_dataset,num_proc =32)
output_dir = "dataset/split_SubjectSpatial200K/test"
os.makedirs(output_dir, exist_ok=True)
train_dataset.to_parquet(os.path.join(output_dir,"data.parquet"))

test_dataset = dataset.filter(filter_test_dataset,num_proc =32)
num_shards = 12
output_dir = "dataset/SubjectSpatial200K_train/Collection3"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir,"data-{index:05d}-of-{num_shards:05d}.parquet")
for index in range(num_shards):
    shard = dataset.shard(index=index, num_shards=num_shards, contiguous=True)
    shard.to_parquet(output_path.format(index=index,num_shards=num_shards))

