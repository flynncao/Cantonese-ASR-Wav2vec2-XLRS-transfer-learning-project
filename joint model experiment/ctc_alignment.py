def ctc_alignment(predictions, targets, blank_id=0):
    """
    进行 CTC 对齐。
    
    参数:
        predictions (list): 模型预测的字符 ID 列表。
        targets (list): 真实标签的字符 ID 列表。
        blank_id (int): 空白标签的 ID,默认为 0。
        
    返回:
        list: 对齐后的预测字符 ID 列表。
    """
    aligned_predictions = []
    previous_char = None
    
    for pred_char, target_char in zip(predictions, targets):
        if pred_char != blank_id and pred_char != previous_char:
            aligned_predictions.append(pred_char)
        previous_char = pred_char
    
    return aligned_predictions
