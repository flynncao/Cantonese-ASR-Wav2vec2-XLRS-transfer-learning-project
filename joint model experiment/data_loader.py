import os
import re
import torch
import librosa
import string
from torch.utils.data import Dataset, DataLoader
import json
import torchaudio
from transformers import Wav2Vec2Processor
from data_collator import DataCollatorCTCWithPadding
from inspect_data import inspect_data

class CommonVoiceDataset(Dataset):
    def __init__(self, data_path, tsv_file, processor, target_sampling_rate=16000):
        self.data_path = data_path
        self.tsv_file = tsv_file
        self.processor = processor

        target_sampling_rate = 16000
        self.target_sampling_rate = target_sampling_rate
        
        # 加载vocab
        with open("/scratch/s5600502/thesis_project/mandarin_cantonese/vocab.json", 'r') as vocab_file:
            self.vocab_dict = json.load(vocab_file)
        
        # 读取 TSV 文件并提取音频文件路径和转录文本
        self.data = self._load_data()

    def _remove_special_characters(self, text):
        # 定义要忽略的字符
        chars_to_ignore_regex = '[\丶\,\?\.\!\-\;\:"\“\%\‘\”\�\．\⋯\！\－\：\–\。\》\,\）\,\？\；\～\~\…\︰\，\（\」\‧\《\﹔\、\—\／\,\「\﹖\·\']'  # 这里可以根据需要添加更多字符

        # 移除特定字符
        text = re.sub(chars_to_ignore_regex, '', text).lower()

        # 针对特定的简单条件进行字符替换，例如将 'd' 替换为 '啲'
        # 注意：这里的逻辑是如果 'd' 是单独出现的字母，则替换为 '啲'
        # 如果您有其他复杂逻辑，可能需要更详细的判断条件
        if "d" in text:
            if len([c for c in text if c in string.ascii_lowercase]) == 1:
                text = text.replace("d", "啲")
        
        return text

    def _load_data(self):
        data = []
        with open(os.path.join(self.data_path, self.tsv_file), "r", encoding="utf-8") as f:
            next(f)  # 跳过标题行
            for line in f:
                cols = line.strip().split("\t")
                audio_file = os.path.join(self.data_path, "clips", cols[1])
                text = self._remove_special_characters(cols[3])  # 假设文本在第四列
                data.append((audio_file, text))
        
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        audio_file, text = self.data[idx]
        
        # 加载音频文件并重新采样
        speech_array, sampling_rate = torchaudio.load(audio_file)
        if sampling_rate != self.target_sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=self.target_sampling_rate)
            speech_array = resampler(speech_array)

        # 确保音频数据是单通道的
        if speech_array.ndim > 1:
            speech_array = speech_array.mean(dim=0)  # 取均值合并声道

        # 音频预处理
        input_values = self.processor(speech_array.squeeze(), sampling_rate=self.target_sampling_rate, return_tensors="pt").input_values

        print("Type of input_values:", type(input_values))
        print("Content of input_values:", input_values)  # 查看内容

        # 文本预处理
        with self.processor.as_target_processor():
            labels = self.processor(text, return_tensors="pt").input_ids
        
        # 确保输入值是张量
        if not isinstance(input_values, torch.Tensor):
            raise ValueError("Input values should be a torch.Tensor")
        
        return {
            'input_values': input_values.squeeze(),
            'labels': labels.squeeze()
        }



# 初始化处理器
processor = Wav2Vec2Processor.from_pretrained("/scratch/s5600502/thesis_project/mandarin_cantonese/cantonese-tokenized-wav2vec2-processor/")

def get_dataloader(data_path, tsv_file, processor, batch_size, shuffle=True):
    dataset = CommonVoiceDataset(data_path, tsv_file, processor)
    # 使用 DataCollatorCTCWithPadding 处理数据填充
    data_collator = DataCollatorCTCWithPadding(processor=processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=data_collator)
    return dataloader

# # 使用数据加载器
# data_path = "/scratch/s5600502/thesis_project/common_voice_cantonese/cv-corpus-17.0-2024-03-15/yue"
# train_tsv = "train.tsv"

# processor_path = "/scratch/s5600502/thesis_project/mandarin_cantonese/cantonese-tokenized-wav2vec2-processor/"
# wav2vec2_processor = Wav2Vec2Processor.from_pretrained(processor_path)
# train_dataloader = get_dataloader(data_path, train_tsv, wav2vec2_processor, batch_size=1)

