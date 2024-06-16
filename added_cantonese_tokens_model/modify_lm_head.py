import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# 加载处理器
processor = Wav2Vec2Processor.from_pretrained("/scratch/s5600502/thesis_project/mandarin_cantonese/cantonese-tokenized-wav2vec2-processor")

# 加载预训练模型，忽略 lm_head 层的大小不匹配
model = Wav2Vec2ForCTC.from_pretrained(
    "/scratch/s5600502/thesis_project/mandarin_cantonese/cantonese-tokenized-wav2vec2-model",
    ignore_mismatched_sizes=True
)

# 获取词汇表大小
vocab_size = len(processor.tokenizer)
print("vocab_size")
print(vocab_size)

# 调整 lm_head 层
model.lm_head = torch.nn.Linear(model.config.hidden_size, vocab_size)
# 初始化新的 lm_head 层的权重
model.lm_head.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
model.lm_head.bias.data.zero_()

# 打印 lm_head 层的大小
print("lm_head weight shape:", model.lm_head.weight.shape)
print("lm_head bias shape:", model.lm_head.bias.shape)

# 保存调整后的模型和处理器到新目录
model.save_pretrained("./wav2vec2-large-xlsr-cantonese")
processor.save_pretrained("./wav2vec2-large-xlsr-cantonese")

print("Model and processor saved successfully.")

# 加载保存的模型和处理器
model = Wav2Vec2ForCTC.from_pretrained("./wav2vec2-large-xlsr-cantonese")
processor = Wav2Vec2Processor.from_pretrained("./wav2vec2-large-xlsr-cantonese")

# 打印 lm_head 层的大小以验证
print(f"lm_head weight shape: {model.lm_head.weight.shape}")
print(f"lm_head bias shape: {model.lm_head.bias.shape}")
