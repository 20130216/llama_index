from llama_cloud.client import LlamaCloud

client = LlamaCloud(
    token="llx-qm4E2vTT3LUNgPYiSdu9xDhGUtNEC5QYXyJCodDofkYLwJfY",
    base_url="https://api.cloud.llamaindex.ai",
)
try:
    pipelines = client.pipelines.search_pipelines(project_name="Default")
    print("LlamaCloud API key is valid")
    print(f"Found pipelines: {pipelines}")
except Exception as e:
    print(f"LlamaCloud API key error: {e}")
