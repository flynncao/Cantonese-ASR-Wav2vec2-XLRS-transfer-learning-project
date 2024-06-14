from transformers import BertModel, BertConfig, AutoTokenizer

def load_bert_model(model_path, config_path):
    model = BertModel.from_pretrained(model_path, config=config_path)
    return model