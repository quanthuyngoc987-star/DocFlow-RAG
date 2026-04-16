"""
重排序器 —— 对检索结果进行二次精排

学习要点：
- 两阶段检索（Recall + Rerank）是工业界常用的范式
- Recall 阶段用高效检索（FAISS/BM25）从大量文档中召回候选
- Rerank 阶段用更精确的模型（交叉编码器/LLM）对候选精排
- 交叉编码器比双塔模型更精确，但速度更慢（适合对少量候选精排）
"""

import logging
import threading
from config import RERANK_METHOD

# 交叉编码器（懒加载 + 线程安全）
_cross_encoder = None
_cross_encoder_lock = threading.Lock()


def get_cross_encoder():
    """懒加载交叉编码器模型（双重检查锁定，线程安全）"""
    global _cross_encoder
    if _cross_encoder is None:
        with _cross_encoder_lock:
            if _cross_encoder is None:
                try:
                    from sentence_transformers import CrossEncoder
                    _cross_encoder = CrossEncoder(
                        'sentence-transformers/distiluse-base-multilingual-cased-v2',
                        local_files_only=True
                    )
                    logging.info("交叉编码器加载成功（本地缓存）")
                except Exception as e:
                    logging.warning(f"交叉编码器本地缓存不可用，直接回退不重排序: {str(e)}")
                    _cross_encoder = None
    return _cross_encoder


def rerank_with_cross_encoder(query, docs, doc_ids, metadata_list, top_k=5):
    """使用交叉编码器对检索结果进行重排序"""
    if not docs:
        return []

    encoder = get_cross_encoder()
    if encoder is None:
        logging.warning("交叉编码器不可用，跳过重排序")
        return _fallback_results(doc_ids, docs, metadata_list)

    cross_inputs = [[query, doc] for doc in docs]
    try:
        scores = encoder.predict(cross_inputs)
        results = [
            (doc_id, {'content': doc, 'metadata': meta, 'score': float(score)})
            for doc_id, doc, meta, score in zip(doc_ids, docs, metadata_list, scores)
        ]
        results = sorted(results, key=lambda x: x[1]['score'], reverse=True)
        return results[:top_k]
    except Exception as e:
        logging.error(f"交叉编码器重排序失败: {str(e)}")
        return _fallback_results(doc_ids, docs, metadata_list)


def rerank_with_llm(query, docs, doc_ids, metadata_list, top_k=5):
    """兼容旧配置入口，内部回退到交叉编码器重排序。"""
    return rerank_with_cross_encoder(query, docs, doc_ids, metadata_list, top_k)


def rerank_results(query, docs, doc_ids, metadata_list, method=None, top_k=5):
    """对检索结果进行重排序（统一入口）"""
    if method is None:
        method = RERANK_METHOD

    if method == "llm":
        return rerank_with_llm(query, docs, doc_ids, metadata_list, top_k)
    elif method == "cross_encoder":
        return rerank_with_cross_encoder(query, docs, doc_ids, metadata_list, top_k)
    else:
        return _fallback_results(doc_ids, docs, metadata_list)


def _fallback_results(doc_ids, docs, metadata_list):
    """回退方案：按原始顺序返回"""
    return [(doc_id, {'content': doc, 'metadata': meta, 'score': 1.0 - idx / len(docs)})
            for idx, (doc_id, doc, meta) in enumerate(zip(doc_ids, docs, metadata_list))]
