"""
LLM 调用 —— 大模型回答生成（DeepSeek）

学习要点：
- Prompt Engineering：如何构建高质量的提示词模板
- 流式输出 vs 非流式输出的区别
- 云端 DeepSeek API 的对接
"""

import json
import logging
import requests
from typing import Any, List, Optional
from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_API_URL,
    DEEPSEEK_MODEL_NAME
)
from core.retriever import recursive_retrieval
from core.vector_store import vector_store
from features.conflict_detector import detect_conflicts, evaluate_source_credibility
from features.thinking_chain import process_thinking_content

CONNECT_TIMEOUT = 8
READ_TIMEOUT = 45


def call_deepseek_api(prompt, temperature=0.7, max_tokens=1024):
    """调用 DeepSeek 云端 API 获取回答"""
    if not DEEPSEEK_API_KEY:
        logging.error("未设置 DEEPSEEK_API_KEY")
        return "错误：未配置 DeepSeek API 密钥。"

    try:
        payload = {
            "model": DEEPSEEK_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False, "max_tokens": max_tokens,
            "temperature": temperature, "top_p": 0.7, "top_k": 50,
            "frequency_penalty": 0.5, "n": 1,
            "response_format": {"type": "text"}
        }
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY.strip()}",
            "Content-Type": "application/json; charset=utf-8"
        }
        json_payload = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        response = requests.post(
            DEEPSEEK_API_URL,
            data=json_payload,
            headers=headers,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
        )
        response.raise_for_status()
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0]["message"]
            content = message.get("content", "")
            reasoning = message.get("reasoning_content", "")
            if reasoning:
                return f"{content}<think>{reasoning}</think>"
            return content
        return "API返回结果格式异常"

    except requests.exceptions.RequestException as e:
        logging.error(f"调用DeepSeek API时出错: {str(e)}")
        return f"调用API时出错: {str(e)}"
    except Exception as e:
        logging.error(f"DeepSeek API 未知错误: {str(e)}")
        return f"发生未知错误: {str(e)}"


def call_llm_simple(prompt, model_choice="deepseek"):
    """简单的 LLM 调用（用于递归检索中的查询改写判断）。"""
    # 查询改写对当前云端路径的收益有限，默认跳过以减少额外等待。
    return "不需要进一步查询"


def _is_error_text(text):
    """判断返回内容是否为后端错误提示文本。"""
    if not isinstance(text, str):
        return False
    keywords = ["调用API时出错", "系统错误", "HTTPConnectionPool", "Max retries exceeded", "未配置"]
    return any(k in text for k in keywords)


def _extract_message_text(content: Any) -> str:
    """兼容 Gradio Chatbot 多种 content 结构，提取纯文本。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                if text:
                    parts.append(str(text).strip())
            elif isinstance(item, str):
                parts.append(item.strip())
        return "\n".join([p for p in parts if p]).strip()
    return str(content).strip()


def _normalize_chat_history(chat_history: Optional[List[dict]], max_messages: int = 8) -> List[dict]:
    """将历史对话标准化为 role/content 文本结构，仅保留最近若干条消息。"""
    if not chat_history or not isinstance(chat_history, list):
        return []

    normalized = []
    for msg in chat_history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        text = _extract_message_text(msg.get("content", ""))
        if not text:
            continue
        normalized.append({"role": role, "content": text})

    return normalized[-max_messages:]


def _build_history_text(chat_history: Optional[List[dict]], max_chars: int = 1400) -> str:
    """构建用于 Prompt 的历史对话文本。"""
    normalized = _normalize_chat_history(chat_history)
    if not normalized:
        return "无"

    lines = []
    for item in normalized:
        prefix = "用户" if item["role"] == "user" else "助手"
        lines.append(f"{prefix}: {item['content']}")

    history_text = "\n".join(lines).strip()
    if len(history_text) > max_chars:
        history_text = history_text[-max_chars:]
    return history_text


def _build_retrieval_query(question: str, chat_history: Optional[List[dict]]) -> str:
    """基于最近用户消息构建历史感知检索查询。"""
    normalized = _normalize_chat_history(chat_history)
    if not normalized:
        return question

    recent_user_msgs = [m["content"] for m in normalized if m["role"] == "user" and m["content"].strip()]
    recent_user_msgs = recent_user_msgs[-2:]
    if not recent_user_msgs:
        return question

    context = "；".join([m for m in recent_user_msgs if m.strip()])
    if context:
        return f"{context}；{question}"
    return question


def _build_prompt(question, context, enable_web_search, knowledge_base_exists,
                  time_sensitive, conflict_detected, history_text="无"):
    """构建提示词"""
    prompt_template = """作为一个专业的问答助手，你需要基于以下{context_type}回答用户问题。

历史对话（用于理解代词、省略和上下文）：
{history_text}

提供的参考内容：
{context}

用户问题：{question}

请遵循以下回答原则：
1. 仅基于提供的参考内容回答问题，不要使用你自己的知识
2. 如果参考内容中没有足够信息，请坦诚告知你无法回答
3. 回答应该全面、准确、有条理，并使用适当的段落和结构
4. 请用中文回答
5. 在回答末尾标注信息来源{time_instruction}{conflict_instruction}

请现在开始回答："""

    return prompt_template.format(
        context_type="本地文档和网络搜索结果" if enable_web_search and knowledge_base_exists else (
            "网络搜索结果" if enable_web_search else "本地文档"),
        history_text=history_text,
        context=context if context else (
            "网络搜索结果将用于回答。" if enable_web_search and not knowledge_base_exists else "知识库为空或未找到相关内容。"),
        question=question,
        time_instruction="，优先使用最新的信息" if time_sensitive and enable_web_search else "",
        conflict_instruction="，并明确指出不同来源的差异" if conflict_detected else ""
    )


def _build_context(all_contexts, all_doc_ids, all_metadata, enable_web_search):
    """构建上下文和来源信息"""
    context_parts = []
    sources_for_conflict = []

    for doc, doc_id, metadata in zip(all_contexts, all_doc_ids, all_metadata):
        source_type = metadata.get('source', '本地文档')
        source_item = {'text': doc, 'type': source_type}

        if source_type == 'web':
            url = metadata.get('url', '未知URL')
            title = metadata.get('title', '未知标题')
            context_parts.append(f"[网络来源: {title}] (URL: {url})\n{doc}")
            source_item['url'] = url
            source_item['title'] = title
        else:
            source = metadata.get('source', '未知来源')
            context_parts.append(f"[本地文档: {source}]\n{doc}")
            source_item['source'] = source

        sources_for_conflict.append(source_item)

    return "\n\n".join(context_parts), sources_for_conflict


def query_answer(question, enable_web_search=False, model_choice="deepseek", progress=None, chat_history=None):
    """
    问答处理主流程（非流式）

    完整流程：递归检索 → 构建上下文 → 矛盾检测 → 构建Prompt → LLM生成
    """
    try:
        knowledge_base_exists = vector_store.is_ready
        if not knowledge_base_exists and not enable_web_search:
            return "⚠️ 知识库为空，请先上传文档。"

        if progress:
            progress(0.3, desc="执行递归检索...")

        retrieval_query = _build_retrieval_query(question, chat_history)
        all_contexts, all_doc_ids, all_metadata = recursive_retrieval(
            initial_query=retrieval_query, enable_web_search=enable_web_search, model_choice=model_choice
        )

        context, sources = _build_context(all_contexts, all_doc_ids, all_metadata, enable_web_search)
        conflict_detected = detect_conflicts(sources)
        time_sensitive = any(w in question for w in ["最新", "今年", "当前", "最近", "刚刚"])
        history_text = _build_history_text(chat_history)

        prompt = _build_prompt(question, context, enable_web_search,
                               knowledge_base_exists, time_sensitive, conflict_detected, history_text)

        if progress:
            progress(0.8, desc="生成回答...")

        if model_choice != "deepseek":
            logging.info(f"模型选项 {model_choice} 已回退为 deepseek")
        result = call_deepseek_api(prompt, temperature=0.7, max_tokens=1536)

        if _is_error_text(result):
            return result

        return process_thinking_content(result)

    except json.JSONDecodeError:
        return "响应解析失败，请重试"
    except Exception as e:
        return f"系统错误: {str(e)}"


def stream_answer(question, enable_web_search=False, model_choice="deepseek", progress=None, chat_history=None):
    """问答处理主流程（流式，用于 Gradio generator 模式）"""
    try:
        knowledge_base_exists = vector_store.is_ready
        if not knowledge_base_exists and not enable_web_search:
            yield "⚠️ 知识库为空，请先上传文档。", "遇到错误"
            return

        if progress:
            progress(0.3, desc="执行递归检索...")

        retrieval_query = _build_retrieval_query(question, chat_history)
        all_contexts, all_doc_ids, all_metadata = recursive_retrieval(
            initial_query=retrieval_query, enable_web_search=enable_web_search, model_choice=model_choice
        )

        context, sources = _build_context(all_contexts, all_doc_ids, all_metadata, enable_web_search)
        conflict_detected = detect_conflicts(sources)
        time_sensitive = any(w in question for w in ["最新", "今年", "当前", "最近", "刚刚"])
        history_text = _build_history_text(chat_history)

        prompt = _build_prompt(question, context, enable_web_search,
                               knowledge_base_exists, time_sensitive, conflict_detected, history_text)

        if model_choice != "deepseek":
            logging.info(f"模型选项 {model_choice} 已回退为 deepseek")
        full_answer = call_deepseek_api(prompt, temperature=0.7, max_tokens=1536)
        if _is_error_text(full_answer):
            yield full_answer, "遇到错误"
            return
        yield process_thinking_content(full_answer), "完成!"

    except Exception as e:
        yield f"系统错误: {str(e)}", "遇到错误"
