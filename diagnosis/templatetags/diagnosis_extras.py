from django import template

register = template.Library()

@register.simple_tag
def get_max_confidence_info(diagnosis):
    """
    获取诊断结果中置信度最高的类别信息
    返回包含index和confidence的字典
    """
    confidences = [
        getattr(diagnosis, 'confidence_0', 0),
        getattr(diagnosis, 'confidence_1', 0),
        getattr(diagnosis, 'confidence_2', 0),
        getattr(diagnosis, 'confidence_3', 0),
        getattr(diagnosis, 'confidence_4', 0),
        getattr(diagnosis, 'confidence_5', 0),
        getattr(diagnosis, 'confidence_6', 0),
        getattr(diagnosis, 'confidence_7', 0),
    ]
    
    max_confidence = max(confidences)
    max_index = confidences.index(max_confidence)
    
    return {
        'index': max_index,
        'confidence': max_confidence
    }