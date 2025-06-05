from llama_cloud.client import LlamaCloud

import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv(
    dotenv_path="/Users/wingzheng/Desktop/github/ParseDoc/llama_index/.env",
    override=True,
)

# 从环境变量中获取 LlamaParse 的配置（来自“已知1”）
base_url = os.getenv("LLAMA_CLOUD_US_BASE_URL", "https://api.cloud.llamaindex.ai")
api_key = os.getenv("LLAMA_CLOUD_API_KEY")

# LlamaCloud和LlamaParse一样，核心只需要传入这2个参数即可
client = LlamaCloud(
    token=api_key,
    base_url=base_url,
)
try:
    pipelines = client.pipelines.search_pipelines(project_name="Default")
    print("LlamaCloud API key is valid")
    print(f"Found pipelines: {pipelines}")
except Exception as e:
    print(f"LlamaCloud API key error: {e}")
