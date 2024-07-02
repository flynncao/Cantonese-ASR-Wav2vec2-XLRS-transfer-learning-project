from opencc import OpenCC

def convert_to_simplified(text: str) -> str:
    converter = OpenCC('t2s')  # t2s 表示从繁体转到简体
    simplified_text = converter.convert(text)
    return simplified_text