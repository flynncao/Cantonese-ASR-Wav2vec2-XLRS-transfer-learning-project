import json
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch

# 加载简体中文Wav2Vec2模型
model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
processor = Wav2Vec2Processor.from_pretrained(model_name)
tokenizer = processor.tokenizer

# 读取从文件中提取的简体粤语字符
input_file_path = "/scratch/s5600502/thesis_project/cantonese-nlp/Text Preprocessing/cantonese_simplified_characters.json"
output_file_path = "/scratch/s5600502/thesis_project/cantonese-nlp/Text Preprocessing/newly_added_characters.json"

with open(input_file_path, 'r', encoding='utf-8') as file:
    cantonese_chars = json.load(file)

# 获取现有的tokenizer词汇表
existing_vocab = set(tokenizer.get_vocab().keys())

# 查找并添加新的字符
new_chars = [char for char in cantonese_chars if char not in existing_vocab]
print(f"New characters to be added: {new_chars}")
num_added_tokens = tokenizer.add_tokens(new_chars)

# 保存新添加的字符到一个新文件中
with open(output_file_path, 'w', encoding='utf-8') as file:
    json.dump(new_chars, file, ensure_ascii=False, indent=2)

print(f"Added {num_added_tokens} new tokens")
print(f"Newly added characters have been saved to {output_file_path}")

# 加载模型并调整嵌入大小
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# 获取当前的嵌入层
old_embedding_size = model.lm_head.out_features
new_embedding_size = len(tokenizer)

# 创建新的嵌入层并复制旧的嵌入
new_lm_head = torch.nn.Linear(model.lm_head.in_features, new_embedding_size)
new_lm_head.weight.data[:old_embedding_size, :] = model.lm_head.weight.data

# 替换模型的嵌入层
model.lm_head = new_lm_head

# 保存修改后的模型和 tokenizer
processor.save_pretrained("/scratch/s5600502/thesis_project/mandarin_cantonese/cantonese-tokenized-wav2vec2-processor")
model.save_pretrained("/scratch/s5600502/thesis_project/mandarin_cantonese/cantonese-tokenized-wav2vec2-model")

print("Model and tokenizer saved successfully.")
