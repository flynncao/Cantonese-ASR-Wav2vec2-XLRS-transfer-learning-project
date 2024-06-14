import torch
import torchaudio
from datasets import load_metric, load_from_disk
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import re
import string
from trad2simp import convert_to_simplified

# 替换为你的模型路径
model_id = "./wav2vec2-large-xlsr-cantonese"
model_path = "./cantonese-tokenized-wav2vec2-model"  # 使用最新的模型

chars_to_ignore_regex = '[\,\?\.\!\-\;\:"\"\%\'\"\�\．\⋯\！\－\：\–\。\》\,\）\,\？\；\～\~\…\︰\，\（\」\‧\《\﹔\、\—\／\,\「\﹖\·\']'

# 从磁盘加载测试数据集
common_voice_test_path = "/scratch/s5600502/thesis_project/baseline/common_voice_test"
test_dataset = load_from_disk(common_voice_test_path)

cer_metric = load_metric("cer")
processor = Wav2Vec2Processor.from_pretrained(f"{model_id}")
model = Wav2Vec2ForCTC.from_pretrained(f"{model_path}")
model.to("cuda")

resamplers = {
    48000: torchaudio.transforms.Resample(48000, 16000),
    44100: torchaudio.transforms.Resample(44100, 16000),
    32000: torchaudio.transforms.Resample(32000, 16000),
    16000: torchaudio.transforms.Resample(16000, 16000),
}

# 预处理数据集
def speech_file_to_array_fn(batch):
    sen = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
    if "d" in sen:
        if len([c for c in sen if c in string.ascii_lowercase]) == 1:
            sen = sen.replace("d", "啲")
    
    batch["sentence"] = convert_to_simplified(sen)
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = resamplers[sampling_rate](speech_array).squeeze().numpy()
    return batch

test_dataset = test_dataset.map(speech_file_to_array_fn)

# 评估数据集
def evaluate(batch):
    inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_strings"] = processor.batch_decode(pred_ids)
    return batch

result = test_dataset.map(evaluate, batched=True, batch_size=16)

# 计算并打印 CER
result["sentence"] = [convert_to_simplified(sen) for sen in result["sentence"]]  # 转换参考文本为简体粤语
result["pred_strings"] = [convert_to_simplified(sen) for sen in result["pred_strings"]]
cer_value = cer_metric.compute(predictions=result["pred_strings"], references=result["sentence"])
print(f"CER: {cer_value:.2f}")

# 将结果写入文件
with open("evaluation_results.txt", "w", encoding="utf-8") as f:
    f.write(f"CER: {cer_value:.2f}\n\n")
    for pred, ref in zip(result["pred_strings"], result["sentence"]):
        f.write(f"Prediction: {pred}\n")
        f.write(f"Reference: {ref}\n")
        f.write("\n")
