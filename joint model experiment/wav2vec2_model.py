from transformers import Wav2Vec2Model

def load_wav2vec2_model(model_path, config_path):
    model = Wav2Vec2Model.from_pretrained(model_path, config=config_path)
    return model
