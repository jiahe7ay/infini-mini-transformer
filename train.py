# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os
from pathlib import Path
import sys

import copy
from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional, Sequence
from datasets import load_dataset,Dataset
import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

from qwen.tokenization_qwen import QWenTokenizer
#from conversation import get_conv_template
from transformers import  GemmaConfig,GemmaForCausalLM

from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    TrainerCallback,
    TrainingArguments,
)
import numpy as np


from functools import partial
from datasets import Dataset

#IGNORE_TOKEN_ID = LabelSmoother.ignore_index

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    flash_attn: bool = False


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    min_lr: float = field(
        default = None
    )

    segment_size:int = field(
        default = 16  
    )


    infini_layer_idx: int = field(
        default = 14  
    )


local_rank = None



def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def token_to_id(samples: dict,tokenizer) -> dict:
    vocab_size = len(tokenizer)
    map_dtype = np.uint16 if vocab_size < 65535 else np.uint32
    batch_txt = samples["text"]
    outputs = tokenizer(
        batch_txt,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )

    input_ids = [np.array(item[0:768], dtype=map_dtype) for item in outputs["input_ids"]]

    return {"input_ids": input_ids}


def get_datasets(data_path, preprocess_func, num_proc):
    datasets_list=[]
    for filename in os.listdir(data_path):
        datasets_list.append(os.path.join(data_path, filename))
    raw_dataset=load_dataset("parquet", data_files=datasets_list,split="train",keep_in_memory=False,cache_dir=".cache")

    
    tokenized_datasets = raw_dataset.map(
        preprocess_func,
        batched = True,
        batch_size = 10000,
        num_proc = num_proc,
        desc = "Tokenizing and reformatting instruction data"
    )  
    
    return tokenized_datasets


def make_supervised_data_module(
    model, tokenizer: transformers.PreTrainedTokenizer, max_length, fwd_batch_size, data_args, mask_user = True
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    rank0_print("Loading data...")


    preprocess_func = partial(token_to_id,tokenizer=tokenizer,)
    
    train_dataset = get_datasets(data_args.data_path, preprocess_func, 32)

    return dict(train_dataset=train_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.do_eval = False
    local_rank = training_args.local_rank
    world_size = training_args.world_size
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = transformers.AutoModelForCausalLM.from_pretrained(
    #     model_args.model_name_or_path,
    #     cache_dir=training_args.cache_dir
    # )
    config=GemmaConfig.from_pretrained(model_args.model_name_or_path,attn_implementation="eager")
    config.use_cache=False
    config.segment_size=training_args.segment_size
    config.infini_layer_idx=training_args.infini_layer_idx
    tokenizer = QWenTokenizer.from_pretrained("./qwen")
    tokenizer.pad_token_id = tokenizer.im_end_id
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    model=GemmaForCausalLM(config).to(device)

    
    fwd_batch_size = training_args.per_device_train_batch_size * world_size
    
    train_dataset = make_supervised_data_module(model = model, 
                                              tokenizer=tokenizer,
                                              max_length = training_args.model_max_length,
                                              fwd_batch_size = fwd_batch_size,
                                              data_args=data_args,
                                             )
    
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args,data_collator=data_collator, **train_dataset
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir = training_args.output_dir)


if __name__ == "__main__":
    train()
