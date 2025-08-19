from django import template

register = template.Library()

@register.filter
def to_percent(value):
    """把 0-1 范围的数值转换为 0-100 范围的浮点数."""
    try:
        v = float(value or 0)
    except Exception:
        return 0.0
    return v * 100.0

@register.simple_tag
def get_max_confidence_info(diagnosis):
    """
    获取诊断结果中置信度信息。
    含：返回一个字典，包
      - primary: {'index': int, 'label': str, 'confidence': float(百分比)} (始终存在，为最大置信度)
      - positives: [ {'index': int, 'label': str, 'confidence': float(百分比)}, ... ] （置信度 > 0.5 的所有类别，如果没有则为空）
    兼容旧模板的用法（仍可访问 primary.index 和 primary.confidence）。
    """
    # 读取8个置信度，默认0
    confidences = [
        float(getattr(diagnosis, 'confidence_0', 0) or 0),
        float(getattr(diagnosis, 'confidence_1', 0) or 0),
        float(getattr(diagnosis, 'confidence_2', 0) or 0),
        float(getattr(diagnosis, 'confidence_3', 0) or 0),
        float(getattr(diagnosis, 'confidence_4', 0) or 0),
        float(getattr(diagnosis, 'confidence_5', 0) or 0),
        float(getattr(diagnosis, 'confidence_6', 0) or 0),
        float(getattr(diagnosis, 'confidence_7', 0) or 0),
    ]

    # 标签映射
    injury_names = {
        0: '无外伤',
        1: '腹盆腔或腹膜后积血/血肿',
        2: '肝脏损伤',
        3: '脾脏损伤',
        4: '右肾损伤',
        5: '左肾损伤',
        6: '右肾上腺损伤',
        7: '胰腺损伤'
    }

    # primary: 最大置信度
    max_confidence = max(confidences)
    max_index = confidences.index(max_confidence)
    primary = {
        'index': max_index,
        'label': injury_names.get(max_index, '未知类型'),
        'confidence': float(max_confidence * 100)
    }

    # positives: 所有置信度 > 0.5（注意：confidences 存储为 0-1）
    positives = []
    threshold = 0.5
    for i, c in enumerate(confidences):
        if c > threshold:
            positives.append({
                'index': i,
                'label': injury_names.get(i, '未知类型'),
                'confidence': float(c * 100)
            })

    return {
        'primary': primary,
        'positives': positives
    }