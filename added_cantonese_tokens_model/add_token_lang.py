import json
from transformers import BertTokenizer, BertModel

def extend_language_model_vocab(added_tokens_file, language_model_tokenizer, language_model):
    # 读取added_tokens.json文件
    with open(added_tokens_file, 'r', encoding='utf-8') as f:
        added_tokens = json.load(f)
    
    # 将新添加的token添加到语言模型词表中
    num_added_tokens = language_model_tokenizer.add_tokens(list(added_tokens.keys()))
    
    print(f"Added {num_added_tokens} new tokens to the language model vocabulary.")
    
    # 如果有新的token被添加,则调整语言模型的嵌入层大小
    if num_added_tokens > 0:
        language_model.resize_token_embeddings(len(language_model_tokenizer))
    
    return language_model_tokenizer, language_model

# 示例用法
added_tokens_file = "./cantonese-tokenized-wav2vec2-processor/added_tokens.json"
language_model_tokenizer = BertTokenizer.from_pretrained("indiejoseph/bert-base-cantonese")
language_model = BertModel.from_pretrained("indiejoseph/bert-base-cantonese")

updated_tokenizer, updated_language_model = extend_language_model_vocab(added_tokens_file, language_model_tokenizer, language_model)

# 保存更新后的tokenizer和语言模型
updated_tokenizer.save_pretrained("./lang_tokenizer")
updated_language_model.save_pretrained("./language_model")