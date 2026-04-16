"""
配置中心 —— 环境变量加载、模型参数、自动检测机制

学习要点：
- 了解如何通过 .env 文件管理敏感配置（API Key）
- 了解 RAG 系统中的关键超参数及其作用
- 理解 LLM 后端的自动检测与回退机制
"""

import os
import logging
import requests
from pathlib import Path
from dotenv import load_dotenv

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第一步：加载环境变量
# 优先加载 .env（用户配置），不存在则回退到 example.env（示例配置）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
dotenv_path = Path(__file__).parent / ".env"
if not dotenv_path.exists():
    dotenv_path = Path(__file__).parent / "example.env"
    logging.warning("⚠️ 未找到 .env 文件，已回退加载 example.env。建议：cp example.env .env 并填入真实 API Key")
load_dotenv(dotenv_path)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第二步：API 密钥配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SEARCH_ENGINE = "google"

# 主命名使用 DeepSeek；保留 SiliconFlow 变量名兼容旧配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", os.getenv("SILICONFLOW_API_KEY"))
DEEPSEEK_API_URL = os.getenv(
    "DEEPSEEK_API_URL",
    os.getenv("SILICONFLOW_API_URL", "https://api.deepseek.com/chat/completions")
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第三步：模型名称配置
# DeepSeek API 格式: deepseek-chat
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEEPSEEK_MODEL_NAME = os.getenv(
    "DEEPSEEK_MODEL_NAME",
    os.getenv("SILICONFLOW_MODEL_NAME", "deepseek-chat")
)
RERANK_METHOD = os.getenv("RERANK_METHOD", "cross_encoder")

# 兼容旧变量名导入（避免现有模块立即失效）
SILICONFLOW_API_KEY = DEEPSEEK_API_KEY
SILICONFLOW_API_URL = DEEPSEEK_API_URL
SILICONFLOW_MODEL_NAME = DEEPSEEK_MODEL_NAME

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第四步：RAG 超参数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHUNK_SIZE = 400          # 文本分块大小（字符数）
CHUNK_OVERLAP = 40        # 相邻分块的重叠字符数
HYBRID_ALPHA = 0.7        # 混合检索中语义检索的权重（0-1）
RETRIEVAL_TOP_K = 10      # 检索返回的候选文档数量
RERANK_TOP_K = 5          # 重排序后保留的文档数量
MAX_RETRIEVAL_ITERATIONS = 3  # 递归检索的最大迭代轮数

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第五步：运行时环境配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
requests.adapters.DEFAULT_RETRIES = 3

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第六步：LLM 后端自动检测
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def detect_default_model():
    """
    当前版本固定使用 DeepSeek 云端后端
    """
    if DEEPSEEK_API_KEY and DEEPSEEK_API_KEY.strip() and not DEEPSEEK_API_KEY.startswith("Your"):
        logging.info("✅ 检测到 DeepSeek API Key，默认使用云端模型")
    else:
        logging.warning("⚠️ 未配置可用的 DeepSeek API Key，请在 .env 中设置 DEEPSEEK_API_KEY（或兼容的 SILICONFLOW_API_KEY）")
    return "deepseek"

DEFAULT_MODEL_CHOICE = detect_default_model()
