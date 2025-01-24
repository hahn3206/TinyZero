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

import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


def make_prefix(dp, template_type):
    examples = ''
    ex_num = 1
    for example in dp['train']:
        examples += f'Example {ex_num}: \n\tInput: {example["input"]}\n\tOutput: {example["output"]}\n'
        ex_num += 1

    test_input = dp['test'][0]['input']

    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""A conversation between User and Assistant. The user provides a set of Example input/output pairs of 2D grids along with a single Test input grid.  There is one consistent set of rules/transformations that can be applied to transform each input example to the corresponding output.  The Assistant identifies the transformation rules using the examples and then provides the output grid that corresponds to applying these rules to the Test input grid.   The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.  The answer should contain the output grid only, no other text.
User: Example input/output pairs:\n{examples}\nTest input:\n\t{test_input}\nTest output:\n\t??\n Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> [[3, 3, 3], [4, 4, 4], [2, 2, 2]] </answer>.
Assistant: Let me think this through step by step.
<think>"""
    elif template_type == 'qwen-instruct':
        """This works for Qwen Instruct Models"""
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\nPlease analyze the following grid transformation examples and determine the output for the test input. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags.\n\nExample input/output pairs:\n{examples}\nTest input:\n\t{test_input}\n<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    return prefix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/tinydata/arc')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'arc-agi'

    dataset = datasets.load_dataset('lordspline/arc-agi', trust_remote_code=True)

    train_dataset = dataset['training']
    test_dataset = dataset['evaluation']

    instruction_following = "{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type)
            solution = example['test'][0]['output']
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
