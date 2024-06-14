import os
import shutil
import torch
from wav2vec2_model import load_wav2vec2_model
from bert_model import load_bert_model
from joint_model import JointModel
from data_loader import get_dataloader
from datasets import load_metric
from transformers import Wav2Vec2Processor, BertTokenizer
from trad2simp import convert_to_simplified
# import Levenshtein
from ctc_alignment import ctc_alignment

# 设置模型和数据路径
wav2vec2_model_path = "/scratch/s5600502/thesis_project/mandarin_cantonese/cantonese-tokenized-wav2vec2-model/pytorch_model.bin"
wav2vec2_config_path = "/scratch/s5600502/thesis_project/mandarin_cantonese/cantonese-tokenized-wav2vec2-model/config.json"
bert_model_path = "/scratch/s5600502/thesis_project/mandarin_cantonese/language_model/pytorch_model.bin"
bert_config_path = "/scratch/s5600502/thesis_project/mandarin_cantonese/language_model/config.json"
tokenizer_path = "/scratch/s5600502/thesis_project/mandarin_cantonese/lang_tokenizer"
processor_path = "/scratch/s5600502/thesis_project/mandarin_cantonese/cantonese-tokenized-wav2vec2-processor/"
joint_model_path = "/scratch/s5600502/thesis_project/mandarin_cantonese/joint_model/joint_model.pt"
save_joint_model_epoch = "/scratch/s5600502/thesis_project/mandarin_cantonese/train_joint_model/joint_model_epochs"
data_path = "/scratch/s5600502/thesis_project/common_voice_cantonese/cv-corpus-17.0-2024-03-15/yue"
save_tokenizer_path = "/scratch/s5600502/thesis_project/mandarin_cantonese/saved_tokenizer"

print("Loading Wav2Vec2 model and processor...")
# 加载 Wav2Vec2 模型、配置和处理器
wav2vec2_model = load_wav2vec2_model(wav2vec2_model_path, wav2vec2_config_path)
wav2vec2_config = wav2vec2_model.config
wav2vec2_processor = Wav2Vec2Processor.from_pretrained(processor_path)
print("Loaded Wav2Vec2 model and processor.")

print("Loading BERT model, config, and tokenizer...")
# 加载 BERT 模型、配置和标记器
bert_model = load_bert_model(bert_model_path, bert_config_path)
bert_config = bert_model.config
bert_tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
print("Loaded BERT model, config, and tokenizer.")

print("Creating joint model...")
# 创建联合模型
hidden_dim = wav2vec2_config.hidden_size
bert_embed_dim = bert_config.hidden_size
joint_model = JointModel(wav2vec2_model, bert_model, hidden_dim, bert_embed_dim)
print("Created joint model.")

# 加载联合模型权重（如果存在）
if os.path.exists(joint_model_path):
    print("Loading joint model weights...")
    joint_model.load_state_dict(torch.load(joint_model_path), strict=False)
    print("Loaded joint model weights.")

print("Preparing datasets...")
# 准备数据集
train_tsv = "train.tsv"
dev_tsv = "dev.tsv"
train_dataloader = get_dataloader(data_path, train_tsv, wav2vec2_processor, batch_size=1)
val_dataloader = get_dataloader(data_path, dev_tsv, wav2vec2_processor, batch_size=1, shuffle=False)
print("Prepared datasets.")

print("Setting up optimizer and scheduler...")
# 设置优化器和学习率调度器
optimizer = torch.optim.AdamW(joint_model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, steps_per_epoch=len(train_dataloader), epochs=10)
print("Set up optimizer and scheduler.")

# 加载 CER 指标
cer_metric = load_metric("cer")

print("Starting training loop...")
# 训练循环
num_epochs = 101
best_val_cer = float('inf')

for epoch in range(num_epochs):
    print(f"Starting epoch {epoch + 1}/{num_epochs}")
    joint_model.train()
    for i, batch in enumerate(train_dataloader):
        print(f"Training batch {i + 1}/{len(train_dataloader)}")
        input_values, attention_mask, labels = batch['input_values'], batch['attention_mask'], batch['labels']
        try:
            loss = joint_model(input_values, labels=labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        except Exception as e:
            print(f"Error during training batch {i + 1}/{len(train_dataloader)}: {e}")
            continue
    print(f"Completed epoch {epoch + 1}/{num_epochs} training")

    # 在每个 epoch 结束后评估模型
    joint_model.eval()
    pred_texts_all = []
    label_texts_all = []

    print(f"Evaluating epoch {epoch + 1}/{num_epochs}")
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            print(f"Evaluating batch {i + 1}/{len(val_dataloader)}")
            try:
                input_values, attention_mask, labels = batch['input_values'], batch['attention_mask'], batch['labels']
                pred_ids = joint_model.predict(input_values)
                aligned_pred_ids = [ctc_alignment(pred, label) for pred, label in zip(pred_ids, labels)]
                pred_texts = wav2vec2_processor.batch_decode(aligned_pred_ids)
                label_texts = wav2vec2_processor.batch_decode(labels)
                pred_texts = [convert_to_simplified(text) for text in pred_texts]
                label_texts = [convert_to_simplified(text) for text in label_texts]
                pred_texts_all.extend(pred_texts)
                label_texts_all.extend(label_texts)
            except Exception as e:
                print(f"Error during evaluation batch {i + 1}/{len(val_dataloader)}: {e}")
                continue

    # 确保有数据可计算 CER
    if len(pred_texts_all) > 0 and len(label_texts_all) > 0:
        cer = cer_metric.compute(predictions=pred_texts_all, references=label_texts_all)
        print(f"Epoch {epoch + 1}, Validation CER: {cer:.4f}")
        # 检查是否为最佳模型
        if cer < best_val_cer:
            best_val_cer = cer
            best_model_path = os.path.join(os.path.dirname(save_joint_model_epoch), f"best_joint_model.pt")
            torch.save(joint_model.state_dict(), best_model_path)
            print(f"Epoch {epoch + 1}: Best model saved with Validation CER: {cer:.4f}")
    else:
        print(f"Epoch {epoch + 1}, Validation CER: No data to compute CER")
    
    # 每隔 10 个 epoch 保存一次模型
    if (epoch + 1) % 10 == 0:
        model_path = os.path.join(os.path.dirname(save_joint_model_epoch), f"joint_model_epoch_{epoch + 1}.pt")
        torch.save(joint_model.state_dict(), model_path)
        print(f"Epoch {epoch + 1}: Model saved as {model_path}")

# 训练结束后保存最终模型
final_model_path = os.path.join(os.path.dirname(save_joint_model_epoch), f"joint_model_final.pt")
torch.save(joint_model.state_dict(), final_model_path)
print(f"Training completed. Final model saved as {final_model_path}")

# 保存分词器相关文件
if not os.path.exists(save_tokenizer_path):
    os.makedirs(save_tokenizer_path)

tokenizer_files = ["added_tokens.json", "special_tokens_map.json", "tokenizer_config.json", "vocab_rm_line.txt", "vocab.txt"]
for file_name in tokenizer_files:
    src_file = os.path.join(tokenizer_path, file_name)
    dst_file = os.path.join(save_tokenizer_path, file_name)
    if os.path.exists(src_file):
        shutil.copy(src_file, dst_file)
        print(f"Saved {file_name} to {save_tokenizer_path}")
print("All tokenizer files have been saved.")
