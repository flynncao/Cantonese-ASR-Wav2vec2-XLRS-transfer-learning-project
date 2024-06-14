# fine_tune_xlsr_wav2vec2_on_cantonese.py

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import transformers
from datasets import ClassLabel, load_dataset, load_metric, load_from_disk
from transformers import (Trainer, TrainingArguments, Wav2Vec2CTCTokenizer,
                          Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC,
                          Wav2Vec2Processor)

import argparse
parser = argparse.ArgumentParser() 
parser.add_argument('--model', type=str, default="facebook/wav2vec2-large-xlsr-53")
parser.add_argument('--unfreeze', action='store_true')
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--warmup', type=float, default=500)
args = parser.parse_args()

print(f"args: {args}")

# 从本地磁盘加载数据集
common_voice_train = load_from_disk("./common_voice_train")
common_voice_test = load_from_disk("./common_voice_test")

unused_cols = ["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"]
common_voice_train = common_voice_train.remove_columns(unused_cols)
common_voice_test = common_voice_test.remove_columns(unused_cols)

chars_to_ignore_regex = '[\丶\,\?\.\!\-\;\:"\“\%\‘\”\�\．\⋯\！\－\：\–\。\》\,\）\,\？\；\～\~\…\︰\，\（\」\‧\《\﹔\、\—\／\,\「\﹖\·\']'

import string
def remove_special_characters(batch):
    sen = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
    if "d" in sen:
        if len([c for c in sen if c in string.ascii_lowercase]) == 1:
            sen = sen.replace("d", "啲")
    batch["sentence"] = sen
    return batch

common_voice_train = common_voice_train.map(remove_special_characters)
common_voice_test = common_voice_test.map(remove_special_characters)

def extract_all_chars(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

vocab_train = common_voice_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_train.column_names,)
vocab_test = common_voice_test.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_test.column_names,)
vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
vocab_list = [char for char in vocab_list if not char.isascii()]
vocab_list.append(" ")

vocab_dict = {v: k for k, v in enumerate(vocab_list)}
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

with open("vocab.json", "w") as vocab_file:
    json.dump(vocab_dict, vocab_file)

tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True,)

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
processor.save_pretrained("./wav2vec2-large-xlsr-cantonese")

resamplers = {
    48000: torchaudio.transforms.Resample(48000, 16000),
    44100: torchaudio.transforms.Resample(44100, 16000),
}

def load_and_resample(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = resamplers[sampling_rate](speech_array).squeeze().numpy()
    batch["sampling_rate"] = 16_000
    batch["target_text"] = batch["sentence"]
    return batch

common_voice_train = common_voice_train.map(load_and_resample, remove_columns=common_voice_train.column_names,)
common_voice_test = common_voice_test.map(load_and_resample, remove_columns=common_voice_test.column_names,)

def prepare_dataset(batch):
    batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch

common_voice_train = common_voice_train.map(prepare_dataset, remove_columns=common_voice_train.column_names, batch_size=-1, num_proc=10, batched=True,)
common_voice_test = common_voice_test.map(prepare_dataset, remove_columns=common_voice_test.column_names, batch_size=-1, num_proc=10, batched=True,)

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
cer_metric = load_metric("./cer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}

model = Wav2Vec2ForCTC.from_pretrained(
    args.model,
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    gradient_checkpointing=True,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)

if not args.unfreeze:
    model.freeze_feature_extractor()

training_args = TrainingArguments(
    output_dir="./wav2vec2-large-xlsr-cantonese",
    group_by_length=True,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    eval_steps=400,
    num_train_epochs=40,
    fp16=True,
    fp16_backend="amp",
    logging_strategy="steps",
    logging_steps=400,
    learning_rate=args.lr,
    warmup_steps=100,
    save_steps=2376,
    save_total_limit=3,
    dataloader_num_workers=20,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=common_voice_train,
    eval_dataset=common_voice_test,
    tokenizer=processor.feature_extractor,
)

trainer.train()
