import torch
import torch.nn as nn
from wav2vec2_model import load_wav2vec2_model
from bert_model import load_bert_model
from transformers import BertTokenizer, Wav2Vec2Processor

class TransformerLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TransformerLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, hidden_states):
        transformed_states = self.linear(hidden_states)
        return transformed_states

class JointModel(nn.Module):
    def __init__(self, wav2vec2_model, bert_model, hidden_dim, bert_embed_dim):
        super(JointModel, self).__init__()
        self.wav2vec2_model = wav2vec2_model
        self.bert_model = bert_model
        self.transformer_layer = TransformerLayer(hidden_dim, bert_embed_dim)
    
    def forward(self, audio_input, labels=None):
        # 将音频输入传递给 Wav2Vec2 模型，得到隐藏状态表示
        hidden_states = self.wav2vec2_model(audio_input).last_hidden_state
        
        # 使用转换层将隐藏状态表示转换为与 BERT 词嵌入维度相同的表示
        transformed_states = self.transformer_layer(hidden_states)
        
        # 将转换后的表示作为输入传递给 BERT 模型
        bert_output = self.bert_model(inputs_embeds=transformed_states)

        # 提取 BERT 的输出
        last_hidden_states = bert_output.last_hidden_state
        
        if labels is not None:
            # 计算损失
            loss_fn = nn.CrossEntropyLoss()
            # 注意: 这里假设 labels 的形状是 [batch_size, seq_len]
            # 并且 last_hidden_states 的形状是 [batch_size, seq_len, hidden_size]
            # 将 logits 和 labels 展平以计算损失
            loss = loss_fn(last_hidden_states.view(-1, last_hidden_states.size(-1)), labels.view(-1))
            return loss
        
        # 对 BERT 的输出进行解码，得到最终的预测文本
        predicted_text = self.process_bert_output(last_hidden_states)
        return predicted_text
    
    def process_bert_output(self, bert_output):
        # 获取 BERT 模型的预测结果
        predictions = bert_output
        
        # 对预测结果进行解码，得到预测的文本
        predicted_tokens = torch.argmax(predictions, dim=-1)

        # 加载 BERT 分词器
        bert_tokenizer = BertTokenizer.from_pretrained(pretrained_lan_model_path)
        predicted_text = bert_tokenizer.decode(predicted_tokens)
        
        return predicted_text
    
    def predict(self, input_values):
        # 将音频输入传递给 Wav2Vec2 模型，得到隐藏状态表示
        hidden_states = self.wav2vec2_model(input_values).last_hidden_state
        
        # 使用转换层将隐藏状态表示转换为与 BERT 词嵌入维度相同的表示
        transformed_states = self.transformer_layer(hidden_states)
        
        # 将转换后的表示作为输入传递给 BERT 模型
        bert_output = self.bert_model(inputs_embeds=transformed_states)
        
        # 提取 BERT 的输出
        last_hidden_states = bert_output.last_hidden_state
        
        # 对 BERT 的输出进行解码，得到最终的预测文本
        predicted_text = self.process_bert_output(last_hidden_states)
        return predicted_text


# 预训练模型路径
wav2vec2_model_path = "/scratch/s5600502/thesis_project/mandarin_cantonese/cantonese-tokenized-wav2vec2-model/pytorch_model.bin"
wav2vec2_config_path = "/scratch/s5600502/thesis_project/mandarin_cantonese/cantonese-tokenized-wav2vec2-model/config.json"
processor_path = "/scratch/s5600502/thesis_project/mandarin_cantonese/cantonese-tokenized-wav2vec2-processor/"

# 加载 Wav2Vec2 模型和处理器
wav2vec2_model = load_wav2vec2_model(wav2vec2_model_path, wav2vec2_config_path)
wav2vec2_processor = Wav2Vec2Processor.from_pretrained(processor_path)

# 预训练模型路径
pretrained_lan_model_path = "/scratch/s5600502/thesis_project/mandarin_cantonese/language_model"
config_lan_path = "/scratch/s5600502/thesis_project/mandarin_cantonese/language_model/config.json"
vocab_lan_file = "/scratch/s5600502/thesis_project/mandarin_cantonese/lang_tokenizer/vocab.txt"

# 加载 BERT 模型
bert_model = load_bert_model(pretrained_lan_model_path, config_lan_path)

# 定义 Wav2Vec2 隐藏状态的维度和 BERT 词嵌入的维度
# 加载 Wav2Vec2 模型配置
wav2vec2_config = wav2vec2_model.config
hidden_dim = wav2vec2_config.hidden_size

# 加载 BERT 模型配置
bert_config = bert_model.config
bert_embed_dim = bert_config.hidden_size

# 创建联合模型实例
joint_model = JointModel(wav2vec2_model, bert_model, hidden_dim, bert_embed_dim)

# 保存完整的联合模型对象
model_save_path = "/scratch/s5600502/thesis_project/mandarin_cantonese/train joint model/joint_model/joint_model.pt"
torch.save(joint_model, model_save_path)

# 保存联合模型的状态字典
model_save_path = "/scratch/s5600502/thesis_project/mandarin_cantonese/train joint model/joint_model/joint_model_state_dict.pt"

print("Saving joint model state dict...")
torch.save(joint_model.state_dict(), model_save_path)
print(f"Joint model state dict saved to: {model_save_path}")
