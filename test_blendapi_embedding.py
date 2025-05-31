import httpx
import json
import os
from dotenv import load_dotenv
import logging
import datetime
import time

logging.basicConfig(level=logging.DEBUG)
load_dotenv()

api_key = os.getenv("BLENDAPI_API_KEY")
api_base = os.getenv("BLENDAPI_API_ENDPOINT")

print(f"DEBUG--BLENDAPI_API_KEY: {api_key}")
print(f"DEBUG--BLENDAPI_API_ENDPOINT: {api_base}")

if not api_key or not api_base:
    raise ValueError(
        "BLENDAPI_API_KEY or BLENDAPI_API_ENDPOINT is not set in .env file"
    )

url = f"{api_base.rstrip('/')}/embeddings"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "User-Agent": "LlamaCloud-Test-Client/1.0",
}
data = {
    "input": [
        "test1",
        "test2",
        "test3",
    ],  # input：三个测试文本 ["test1", "test2", "test3"]，用于批量生成嵌入。
    "model": "text-embedding-3-small",
    "dimensions": 1536,
    "encoding_format": "float",
}

retries = 3  # 重试次数
try:
    with httpx.Client(timeout=30.0) as client:  # 设置 30 秒超时
        print(
            f"DEBUG--Sending request with {len(data['input'])} inputs: {data['input']}"
        )  # DEBUG--Sending request with 3 inputs: ['test1', 'test2', 'test3']
        for attempt in range(retries):
            try:
                response = client.post(url, headers=headers, json=data)
                response.raise_for_status()  # 检查响应状态码
                break
            except httpx.HTTPStatusError as e:
                if attempt < retries - 1:
                    print(f"DEBUG--Retry {attempt + 1}/{retries} after error: {e}")
                    time.sleep(2)
                else:
                    raise
        print(f"DEBUG--Response Status Code: {response.status_code}")
        print(f"DEBUG--Response Headers: {response.headers}")
        response_json = (
            response.json()
        )  # 解析响应为 JSON 格式 提取模型名称、嵌入数量（data 字段的长度）。
        print(f"DEBUG--Model: {response_json.get('model')}")
        print(f"DEBUG--Embedding Count: {len(response_json.get('data', []))}")
        print(f"DEBUG--Expected Embedding Count: {len(data['input'])}")
        for i, emb in enumerate(response_json.get("data", [])):
            print(
                f"DEBUG--Embedding {i} Index: {emb.get('index')}, Length: {len(emb.get('embedding', []))}"
            )
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"blendapi_test_log_{timestamp}.json", "w") as f:
            json.dump(response_json, f, indent=2)
        print(f"DEBUG--Response saved to blendapi_test_log_{timestamp}.json")
        # 检查批量处理问题 验证 BlendAPI 批量处理稳定性的核心逻辑，直接检测 API 是否正确处理所有输入。
        if len(response_json.get("data", [])) != len(data["input"]):
            print(
                f"WARNING--Batch processing issue: Expected {len(data['input'])} embeddings, got {len(response_json.get('data', []))}"
            )
except httpx.HTTPStatusError as e:
    print(f"DEBUG--HTTP Error: {e}")
    print(f"DEBUG--Response Status Code: {e.response.status_code}")
    print(
        f"DEBUG--Error Message: {e.response.json().get('error', {}).get('message', 'No error message')}"
    )
    with open(f"blendapi_test_error_log_{timestamp}.json", "w") as f:
        json.dump(e.response.json(), f, indent=2)
    print(f"DEBUG--Error response saved to blendapi_test_error_log_{timestamp}.json")
except httpx.RequestError as e:
    print(f"DEBUG--Request Error: {e}")
except Exception as e:
    print(f"DEBUG--Unexpected Error: {e}")
