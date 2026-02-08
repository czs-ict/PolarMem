import numpy as np
from skimage.filters import threshold_otsu
from typing import Dict, Tuple, List

def calculate_dynamic_threshold(concept_scores: Dict[str, float], fallback_threshold: float = 0.5) -> Tuple[List[str], List[str]]:
    """
    基于 Otsu 算法计算动态阈值，并过滤正负样本。
    
    Args:
        concept_scores: 字典 {concept_name: score_float}
        fallback_threshold: 当 Otsu 失败时的默认阈值 (default 0.5)
    
    Returns:
        pos_concepts: List of names (Score > threshold + 0.05)
        neg_concepts: List of names (Score < threshold - 0.05) -> 视为幻觉
    """
    if not concept_scores:
        return [], []

    scores = np.array(list(concept_scores.values()))
    
    # 默认值
    tau = fallback_threshold
    
    # 尝试使用 Otsu 计算动态阈值
    # 条件：样本 > 1 且 方差足够大
    if len(scores) > 1 and np.var(scores) > 1e-5:
        try:
            tau = threshold_otsu(scores)
        except Exception as e:
            print(f"Otsu failed, using fallback {fallback_threshold}: {e}")
            tau = fallback_threshold
    
    # 基于阈值进行分类
    pos_concepts = []
    neg_concepts = []
    
    # Project Rules: tau +/- 0.05 margin
    high_thresh = tau + 0.05
    low_thresh = tau - 0.05
    
    for concept, score in concept_scores.items():
        if score > high_thresh:
            pos_concepts.append(concept)
        elif score < low_thresh:
            neg_concepts.append(concept)
            
    return pos_concepts, neg_concepts
