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
from trad2simp import convert_to_simplified  # 导入转换函数
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default="/scratch/s5600502/thesis_project/mandarin_cantonese/cantonese-tokenized-wav2vec2-model")
parser.add_argument('--processor_dir', type=str, default="/scratch/s5600502/thesis_project/mandarin_cantonese/cantonese-tokenized-wav2vec2-processor")
parser.add_argument('--unfreeze', action='store_true')
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--warmup', type=float, default=500)
parser.add_argument('--num_epochs', type=int, default=40)  # 新增参数以指定运行多少个 epoch
args = parser.parse_args()

print(f"args: {args}")

# 从本地磁盘加载数据集
common_voice_train = load_from_disk("/scratch/s5600502/thesis_project/baseline/common_voice_train")
common_voice_test = load_from_disk("/scratch/s5600502/thesis_project/baseline/common_voice_test")

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
    batch["sentence"] = convert_to_simplified(sen)  # 转换为简体粤语
    return batch

# 检查数据集是否存在缺失值或无效值
def check_dataset(batch):
    for path in batch["path"]:
        assert isinstance(path, str), "Audio path should be a string"
        assert os.path.exists(path), f"Audio file {path} does not exist"
    return batch

common_voice_train = common_voice_train.map(check_dataset, batched=True)
common_voice_test = common_voice_test.map(check_dataset, batched=True)

common_voice_train = common_voice_train.map(remove_special_characters, batched=True)
common_voice_test = common_voice_test.map(remove_special_characters, batched=True)

def extract_all_chars(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

vocab_train = common_voice_train.map(extract_all_chars, batched=True, batch_size=1, keep_in_memory=True, remove_columns=common_voice_train.column_names)
vocab_test = common_voice_test.map(extract_all_chars, batched=True, batch_size=1, keep_in_memory=True, remove_columns=common_voice_test.column_names)
vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
vocab_list = [char for char in vocab_list if ord(char) > 127]
vocab_list.append(" ")

vocab_dict = {v: k for k, v in enumerate(vocab_list)}
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

with open("vocab.json", "w") as vocab_file:
    json.dump(vocab_dict, vocab_file)

tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
processor.save_pretrained("./wav2vec2-large-xlsr-cantonese")

resamplers = {
    48000: torchaudio.transforms.Resample(48000, 16000),
    44100: torchaudio.transforms.Resample(44100, 16000),
    32000: torchaudio.transforms.Resample(32000, 16000),
    16000: torchaudio.transforms.Resample(16000, 16000),
}

def load_and_resample(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = resamplers[sampling_rate](speech_array).squeeze().numpy()
    batch["sampling_rate"] = 16_000
    batch["target_text"] = batch["sentence"]
    return batch

# 处理数据集，删除临时文件以节省空间
temp_file_train = "/tmp/common_voice_train_temp.arrow"
temp_file_test = "/tmp/common_voice_test_temp.arrow"
prepared_file_train = "/tmp/common_voice_train_prepared_temp.arrow"
prepared_file_test = "/tmp/common_voice_test_prepared_temp.arrow"

common_voice_train = common_voice_train.map(load_and_resample, remove_columns=common_voice_train.column_names, batch_size=1, num_proc=1, cache_file_name=temp_file_train)
common_voice_test = common_voice_test.map(load_and_resample, remove_columns=common_voice_test.column_names, batch_size=1, num_proc=1, cache_file_name=temp_file_test)

def prepare_dataset(batch):
    batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"]).input_values
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch

common_voice_train = common_voice_train.map(prepare_dataset, remove_columns=common_voice_train.column_names, batch_size=1, num_proc=1, batched=True, cache_file_name=prepared_file_train)
common_voice_test = common_voice_test.map(prepare_dataset, remove_columns=common_voice_test.column_names, batch_size=1, num_proc=1, batched=True, cache_file_name=prepared_file_test)

# 删除临时文件
os.remove(temp_file_train)
os.remove(temp_file_test)

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
cer_metric = load_metric("cer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}

model = Wav2Vec2ForCTC.from_pretrained(args.model_dir)

model.config.gradient_checkpointing = True
model.config.attention_dropout = 0.1
model.config.hidden_dropout = 0.1
model.config.feat_proj_dropout = 0.0
model.config.mask_time_prob = 0.05
model.config.layerdrop = 0.1
model.config.ctc_loss_reduction = "mean"
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = len(processor.tokenizer)

if not args.unfreeze:
    model.freeze_feature_extractor()

checkpoint_dir = "./wav2vec2-large-xlsr-cantonese/checkpoints"

training_args = TrainingArguments(
    output_dir=checkpoint_dir,
    group_by_length=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    evaluation_strategy="epoch",
    num_train_epochs=args.num_epochs,
    fp16=True,
    fp16_backend="amp",
    logging_strategy="epoch",
    logging_dir="./logs",
    learning_rate=args.lr,
    warmup_steps=args.warmup,
    save_steps=10 * len(common_voice_train) // 1,
    save_total_limit=3,
    dataloader_num_workers=20,
    report_to=["tensorboard"],  # 添加 TensorBoard 支持
)

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_cer = float("inf")

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_output = super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        cer = eval_output[f"{metric_key_prefix}_cer"]
        print(f"\nEpoch {self.state.epoch}: Validation CER: {cer:.4f}")
        if cer < self.best_cer:
            self.best_cer = cer
            self.save_model(os.path.join(self.args.output_dir, "best_model"))
            print(f"Best model saved with CER: {cer:.4f}")
        return eval_output

    def log(self, logs: Dict[str, float]) -> None:
        super().log(logs)
        if "loss" in logs:
            print(f"Epoch {self.state.epoch}: Training Loss: {logs['loss']:.4f}")
        if "eval_loss" in logs:
            print(f"Epoch {self.state.epoch}: Validation Loss: {logs['eval_loss']:.4f}")

trainer = CustomTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=common_voice_train,
    eval_dataset=common_voice_test,
    tokenizer=processor.feature_extractor,
)

# 增加初始模型参数检查
initial_params = list(model.parameters())
print(f"Initial model parameters: {initial_params[:5]}")  # 只打印前5个参数

trainer.train()

# 检查训练结束后的模型参数变化
final_params = list(model.parameters())
print(f"Final model parameters: {final_params[:5]}")  # 只打印前5个参数
