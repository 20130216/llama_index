from llama_parse import LlamaParse
import os
from dotenv import load_dotenv


import asyncio

# 非常安全且健壮的写法，它能够确保无论运行环境中是否存在事件循环冲突，或者是否安装了 nest_asyncio，脚本都能正常运行，而不会引入额外的麻烦
# 可选：保留 nest_asyncio 以确保跨环境兼容性 通过 try-except 确保即使未安装也不影响运行。
try:
    # nest_asyncio.apply() 修补 asyncio，允许在已有事件循环的环境中嵌套运行异步任务（如 LlamaParse.load_data）
    # 它解决了 Jupyter Notebook、VS Code 终端等环境中的事件循环冲突，确保异步任务正常完成
    import nest_asyncio

    # 应用 nest_asyncio 以支持嵌套事件循环
    # 建议保留：为确保脚本在多种环境（Notebook、终端、CI/CD）中的兼容性，保留 nest_asyncio.apply() 是安全的防御措施。
    nest_asyncio.apply()
except ImportError:
    print("nest_asyncio not installed, skipping")

# 调试事件循环
try:
    print(f"DEBUG--Current event loop: {asyncio.get_event_loop()}")
    print(f"DEBUG--Is loop running: {asyncio.get_event_loop().is_running()}")
except Exception as e:
    print(f"DEBUG--Event loop error: {e}")

# 清理 LLAMA_CLOUD_BASE_URL
# 单独通过“unset LLAMA_CLOUD_BASE_URL”去除该环境变量的值才可不用此段代码，“printenv | grep LLAMA”查看环境变量验证
# if "LLAMA_CLOUD_BASE_URL" in os.environ:
#     del os.environ["LLAMA_CLOUD_BASE_URL"]

# 加载 .env 文件中的环境变量
load_dotenv(
    dotenv_path="/Users/wingzheng/Desktop/github/ParseDoc/llama_index/.env",
    override=True,
)

# 从环境变量中获取 LlamaParse 的配置（来自“已知1”）
base_url = os.getenv("LLAMA_CLOUD_US_BASE_URL", "https://api.cloud.llamaindex.ai")
api_key = os.getenv("LLAMA_CLOUD_API_KEY")
organization_id = os.getenv("LLAMA_CLOUD_ORGANIZATION_ID", None)
project_name = os.getenv("LLAMA_CLOUD_PROJECT_NAME", "Default")

# 验证环境变量
if not base_url or not base_url.startswith(("http://", "https://")):
    raise ValueError(f"Invalid base_url: {base_url}")
if not api_key:
    raise ValueError("API key is missing")

# print("DEBUG--All environment variables:", os.environ)
# 打印调试信息
print(f"DEBUG--LLAMA_CLOUD_US_BASE_URL is: {base_url}")
print(f"DEBUG--LLAMA_CLOUD_API_KEY is: {api_key}")
print(f"DEBUG--organization_id is: {organization_id}")
print(f"DEBUG--project_name is: {project_name}\n")

# 初始化 LlamaParse 实例，传入配置参数（结合“已知2”）
parser = LlamaParse(
    api_key=api_key,
    # base_url="https://api.cloud.llamaindex.ai",  # 硬编码正确 URL
    base_url=base_url.rstrip("/"),  # 确保移除末尾斜杠
    # organization_id=organization_id,   # 经测试，完全没必要用这两个参数
    # project_id=project_name,
    # result_type="text",
    result_type="markdown",  # 改为 markdown，生成更结构化的输出，减少碎片化
    # split_by_page=False,  # 禁用按页面分割，即不按页面分割，一整页解析； 设置为ture或不写该参数，则按单页分割（建议）；
    fast_mode=True,  # 快速模式跳过所有计算密集型后处理步骤 — 无 OCR、无布局重建、无图像提取、无表格或标题检测。
    # preserve_layout_alignment_across_pages=True # （VIP功能）保持跨页面对齐 对于跨页具有连续表格/对齐方式的文档很有用
    # target_pages="0-2,6-22,33", # 将特定页码作为逗号分隔列表传递来指定要解析的页面
    # user_prompt="If output is not in english, translate it in english." # system_prompt_append
    # do_not_cache=True, # （VIP功能）避免缓存敏感文档
    max_timeout=7200,  # 增加超时到 2 小时， 以支持大型 PDF 的解析。
    check_interval=1,
    verbose=True,  # 启用详细日志
)
# parser.base_url = "https://api.cloud.llamaindex.ai"  # 强制覆盖
print(f"DEBUG--Final base_url used by LlamaParse: {parser.base_url}")
# 设置文件路径
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.abspath(
    "docs/docs/examples/structured_outputs/data/apple_2021_10k.pdf"
)
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File {file_path} does not exist")


# 加载 PDF 文件
try:
    orig_docs = parser.load_data(file_path)
    print(f"DEBUG--Documents loaded: {len(orig_docs)} documents")
    # 改进：添加字符计数和短文本标记
    for i, doc in enumerate(orig_docs):
        if (
            hasattr(doc, "text_resource")
            and doc.text_resource
            and hasattr(doc.text_resource, "text")
            and doc.text_resource.text
        ):
            text = doc.text_resource.text.strip()
            text_preview = text[:10] if text else "Empty text after stripping"
            # 添加字符计数和完整文本标记
            preview_length = len(text_preview)
            is_full_text = len(text) <= 10 and text
            status = " (full text)" if is_full_text else f" ({preview_length} chars)"
        else:
            text_preview = "No text resource"
            status = ""
        print(
            f"DEBUG--Document {i + 1}: ID={doc.id_}, Text Preview={text_preview}{status}"
        )

    # 创建文件夹（如果不存在）
    output_dir = "test-new-add-python-output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 保存解析结果为 Markdown 文件
    # with open("parsed_output_split_by_page_true.md", "w", encoding="utf-8") as f:
    with open(
        os.path.join(output_dir, "parsed_output_split_by_page_true.md"),
        "w",
        encoding="utf-8",
    ) as f:
        for i, doc in enumerate(orig_docs):
            text = (
                doc.text_resource.text.strip()
                if hasattr(doc, "text_resource")
                and doc.text_resource
                and hasattr(doc.text_resource, "text")
                and doc.text_resource.text
                else ""
            )
            f.write(f"## Document {i + 1}\n\n{text}\n\n")
    print("DEBUG--Saved parsed output to parsed_output_split_by_page_false.md")
except Exception as e:
    print(f"Error during load_data: {e}")
    import traceback

    traceback.print_exc()
# orig_docs = LlamaParse(result_type="text").load_data(
#     "docs/docs/examples/structured_outputs/data/apple_2021_10k.pdf"
# )

# 添加统计，分析 Document 文本长度分布 ；帮助判断是否过多短文本。
# 举例：DEBUG--Text length stats: min=1192, max=8840, avg=4196.6
# 解读“举例”：每个 Document 的文本长度在 1192 到 8840 字符之间，平均约 4196.6 字符。
text_lengths = [
    len(doc.text_resource.text.strip())
    for doc in orig_docs
    if hasattr(doc, "text_resource")
    and doc.text_resource
    and hasattr(doc.text_resource, "text")
    and doc.text_resource.text
]
print(
    f"DEBUG--Text length stats: min={min(text_lengths)}, max={max(text_lengths)}, avg={sum(text_lengths) / len(text_lengths):.1f}"
)

# 返回加载的文档
orig_docs
