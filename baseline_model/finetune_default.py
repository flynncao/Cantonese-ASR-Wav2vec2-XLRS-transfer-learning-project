
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

print(torch.cuda.is_available())
print(torch.cuda.device_count())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))


import argparse
parser = argparse.ArgumentParser() 
parser.add_argument('--model', type=str, default="facebook/wav2vec2-large-xlsr-53")
parser.add_argument('--unfreeze', action='store_true')
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--warmup', type=float, default=500)
parser.add_argument('-f', '--fff', help="dummy argument to avoid error in Jupyter", default="dummy_value")
args = parser.parse_args()

print(f"args: {args}")




# 从本地磁盘加载数据集 Load Cantonese language only 
common_voice_train = load_dataset("mozilla-foundation/common_voice_13_0", "zh-HK", split="train")
common_voice_test = load_dataset("mozilla-foundation/common_voice_13_0", "zh-HK", split="test")

unused_cols = ["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"]
common_voice_train = common_voice_train.remove_columns(unused_cols)
common_voice_test = common_voice_test.remove_columns(unused_cols)






# data preprocessing

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



# load datasets and resampling, the modern way
from datasets import Audio
common_voice_train = common_voice_train.cast_column("audio", Audio(sampling_rate=16000)) # reinterpret this column ("audio") as a certain type, with new settings
common_voice_test = common_voice_test.cast_column("audio", Audio(sampling_rate=16000))



# --- 3. Define the prepare_dataset function (like your Whisper one) ---
tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True,)

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
processor.save_pretrained("./wav2vec2-large-xlsr-cantonese")




def prepare_dataset_for_batching(batch, processor_obj=None):
    
    # Calculate durations first to filter
    audio_arrays = [item["array"] for item in batch["audio"]]
    sampling_rates = [item["sampling_rate"] for item in batch["audio"]]
    durations = [len(arr) / sr for arr, sr in zip(audio_arrays, sampling_rates)]
    
    # Create mask for items <= 15 seconds
    valid_mask = [d <= 10.0 for d in durations]
    
    # Filter all batch components
    audio_arrays = [arr for arr, keep in zip(audio_arrays, valid_mask) if keep]
    sampling_rates = [sr for sr, keep in zip(sampling_rates, valid_mask) if keep]
    sentences = [sent for sent, keep in zip(batch["sentence"], valid_mask) if keep]
    
    # Skip processing if no valid items
    if not audio_arrays:
        return {
            "input_values": [],
            "labels": [],
            "input_length": []
        }
    
    # Process audio inputs
    model_inputs = processor_obj(
        audio_arrays,
        sampling_rate=sampling_rates[0],
        padding=False,
        return_tensors=None,
    )

    # Process labels
    with processor_obj.as_target_processor():
        labels = processor_obj.tokenizer(
            sentences,
            add_special_tokens=False,
            padding=False,
        ).input_ids

    # Return only the three required lists
    return {
        "input_values": model_inputs.input_values,
        "labels": labels,
        "input_length": [d for d, keep in zip(durations, valid_mask) if keep]
    }
    

original_train_len = len(common_voice_train)
original_test_len = len(common_voice_test)

# Then process datasets
common_voice_train = common_voice_train.map(
    prepare_dataset_for_batching,
    num_proc=5,
    batched=True,
    fn_kwargs={"processor_obj": processor},
    load_from_cache_file=True,
    remove_columns=common_voice_train.column_names
)

common_voice_test = common_voice_test.map(
    prepare_dataset_for_batching,
    num_proc=5,
    batched=True,
    fn_kwargs={"processor_obj": processor},
    load_from_cache_file=True,
    remove_columns=common_voice_test.column_names
)

# Calculate filtered counts
train_filtered = original_train_len - len(common_voice_train)
test_filtered = original_test_len - len(common_voice_test)

print(f"Filtered {train_filtered} items from training set")
print(f"Filtered {test_filtered} items from test set")

print(common_voice_train[0])
print(common_voice_test[0])


# Define a data collator for CTC with padding and masking
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


# Metrics and model initialization, feature extractor, and model loading
import evaluate
data_collator = DataCollatorCTCWithPadding(
    processor=processor,
    padding=True,
    max_length=int(16_000 * 15),     # cap at 15s
    max_length_labels=512,
    pad_to_multiple_of=16,
    pad_to_multiple_of_labels=8,
)
# Load the built-in CER metric
# cer_metric = load_metric("cer")
cer_metric = evaluate.load("cer")


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    # Avoid in-place modification
    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, group_tokens=False, skip_special_tokens=True)

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


device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# # Configure PyTorch Dynamo to suppress errors during optimization
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

training_args = TrainingArguments(
    output_dir="./wav2vec2-large-xlsr-cantonese",
    group_by_length=True,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    evaluation_strategy="steps",
    eval_steps=400,
    num_train_epochs=10,
    fp16=True,
    fp16_backend="amp",
    fp16_full_eval=True,
    logging_strategy="steps",
    logging_steps=400,
    learning_rate=args.lr,
    warmup_steps=500,
    save_steps=2376,
    save_total_limit=3,
    dataloader_num_workers=0,
    optim="adamw_8bit",
    remove_unused_columns=False,
    torch_compile=False,
)

trainer = Trainer(
    model=model.to(device),
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=common_voice_train,
    eval_dataset=common_voice_test,
    tokenizer=processor.feature_extractor,
)

trainer.train()



