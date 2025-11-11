# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""
import argparse
import os
import re
from datasets import DatasetDict, Features, Sequence, Value, concatenate_datasets, load_dataset,Image

import datasets
import sys
sys.path.append("xxx")
#from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="xxx")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "MiroMind-M1-RL-62K"

    dataset = datasets.load_dataset(data_source)

    train_dataset = dataset["train"]

    instruction_following = "The final answer MUST BE put in \\boxed{}.\n/do_think"

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop('problem')

            question = question_raw + " " + instruction_following
            solution = example.pop('clean_answer')
            data = {
                "data_source":  "math_miromind",# data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "images": [],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": solution,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    #test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    """features = Features({
        "data_source": Value("string"),
        "prompt":  Sequence(feature={"role": Value("string"), "content": Value("string")}),#[{"role": Value("string"), "content": Value("string")}],
        "images": Sequence(feature=Image()),  # 关键：声明为Image类型
        "ability": Value("string"),
        "reward_model": {"style": Value("string"), "ground_truth": Value("string")},
        "extra_info": {
            "split": Value("string"),
            "index": Value("int64"),
            "answer": Value("string"),
            "question": Value("string"),
        },
    })"""
    original_features = train_dataset.features
    updated_features = {"images": Sequence(feature=Image())}
    new_features = {**original_features,** updated_features}

    # 用新的特征字典进行cast（只修改指定key，其余保持原状）
    train_dataset = train_dataset.cast(Features(new_features))

    # 加载数据集时指定features
    #train_dataset = train_dataset.cast(features)
    #test_dataset = test_dataset.cast(features)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    train_size = int(len(train_dataset) * 0.2)
    train_dataset = train_dataset.shuffle().select(range(train_size))
    os.makedirs(args.local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    #test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    """if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)"""
