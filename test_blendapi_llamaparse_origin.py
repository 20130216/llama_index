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
# organization_id = os.getenv("LLAMA_CLOUD_ORGANIZATION_ID", None)
# project_name = os.getenv("LLAMA_CLOUD_PROJECT_NAME", "Default")

# 验证环境变量
if not base_url or not base_url.startswith(("http://", "https://")):
    raise ValueError(f"Invalid base_url: {base_url}")
if not api_key:
    raise ValueError("API key is missing")

# print("DEBUG--All environment variables:", os.environ)
# 打印调试信息
print(f"DEBUG--LLAMA_CLOUD_US_BASE_URL is: {base_url}")
print(f"DEBUG--LLAMA_CLOUD_API_KEY is: {api_key}")
# print(f"DEBUG--organization_id is: {organization_id}")
# print(f"DEBUG--project_name is: {project_name}\n")

# 初始化 LlamaParse 实例，传入配置参数（结合“已知2”）
parser = LlamaParse(
    api_key=api_key,
    # base_url="https://api.cloud.llamaindex.ai",  # 硬编码正确 URL
    base_url=base_url.rstrip("/"),  # 确保移除末尾斜杠
    # organization_id=organization_id  # 经测试，完全没必要用这两个参数
    # project_id=project_name,
    language="ch_sim",  # 支持多语言 "en,fr,de"；仅影响从图像中提取的文本。 测试下来不支持多语言 "ch_sim,en"报错
    # disable_ocr=True # 禁用OCR；默认情况下，LlamaParse 对文档中嵌入的图像运行 OCR
    # fast_mode=True,  # （实操不推荐）快速模式跳过所有计算密集型后处理步骤 — 无 OCR、无布局重建、无图像提取、无表格或标题检测。
    ## 页面独立 单页提取"parse_page_..."   跨页连续：“"parse_document_...”
    parse_mode="parse_page_with_llm",  # 默认；优先级1 花费3；不设置 等同于显式设置（推荐设置） 默认是“平衡模式”
    # parser = LlamaParse(auto_mode=True) # 优先级2；默认和agent之间选择 3-45
    # parse_mode="parse_page_with_agent"  # 优先级3；页面独立但复杂，45
    # parse_mode="parse_document_with_agent"    # 优先级4:好但贵：跨页连续，结构复杂 90
    # parse_mode="parse_page_with_layout_agent" # 优先级4:页面独立且需布局保真，45
    ## 广义，优化整个文档的解析逻辑，包括标题、章节、表格和全局结构；提供全局文档解析，优化跨页表格和结构连续性。
    # parse_mode="parse_document_with_llm", # 优先级2；跨页连续，结构简单 30；
    ## 狭义，专注于跨页的视觉和布局对齐，主要影响文本和表格的格式；在逐页或局部解析的基础上，额外调整跨页对齐。
    preserve_layout_alignment_across_pages=True,  # （VIP功能）保持跨页面对齐 对于跨页具有连续表格/对齐方式的文档很有用
    spreadsheet_extract_sub_tables=True,  # （VIP参数设置，针对表格，强烈建议设置为ture，默认为False）尝试识别多个逻辑子表；输出仍可能合并到一个文件中，具体取决于result_type 或 structured_output。
    # target_pages="0-2,6-22,33", # 将特定页码作为逗号分隔列表传递来指定要解析的页面
    # max_pages=25 # 限制要解析的最大页数,在指定页面后停止解析文档。
    # split_by_page=False,  # 禁用按页面分割，即不按页面分割，一整页解析； 设置为ture或不写该参数，则按单页分割（建议）；
    # RAG去噪声 更适合去掉页面标签，除非单独强调页面，否则不建议把页码植入
    # page_separator="\n== {pageNumber} ==\n", # （半VIP参数设置）
    # page_prefix="START OF PAGE: {pageNumber}\n",
    # page_suffix="\nEND OF PAGE: {pageNumber}",
    # user_prompt="Summarize the content in 100 words or less." # 内容摘要：在解析时提取文档的关键信息并生成摘要
    user_prompt="针对本文件，排除所有页码并确保表格完整。",
    # result_type="text", # 此时不一定必须result_type="markdown"
    # system_prompt_append="For tables, add a caption above each table describing its content." # 增强表格格式：要求表格在 Markdown 中使用特定的对齐方式或额外说明
    system_prompt_append="仅提取内容，排除页码、页眉、页脚或类似 '结束页面' 或 '# 1-1-156' 的标记，将跨页表格合并为单一表格。",
    result_type="markdown",  # 需要用到system_prompt_append时必须要如此设置；默认解析模式 明确提到默认输出为md格式 “ parse_mode="parse_page_with_llm””这里提及；
    ## 需要图像
    disable_image_extraction=False,  # 默认为false；此处显式设定
    save_images=True,  # 提取并保存文档中的所有图像；场景：需要图像内容进行分析或存档。
    ## 不需要图像：两者互斥，不能同时设置。前者完全不提取，设置之后不需要再设置save_images=True。
    # disable_image_extraction=True # 禁用图像提取，完全不提取；文档中的图像不会被处理。场景：无需图像数据，节省处理时间。
    # structured_output=True  # 默认格式：JSON格式；其结构由 LlamaParse 的默认解析模式决定
    ## 允许用户提供一个自定义的 JSON 架构（JSON Schema）
    # structured_output_json_schema='A JSON SCHEMA'
    ## 3个预定义架构
    # structured_output_json_schema_name="resume" # 简历 此时建议显式设定structured_output=True
    # structured_output_json_schema_name="invoice" # 发票
    # structured_output_json_schema_name="imFeelingLucky" # 允许 LlamaParse 推断输出格式的通配符架构
    # do_not_cache=True, # （VIP功能）避免缓存敏感文档 ；应确保不使用缓存，每次解析都重新处理
    num_workers=10,  # 控制用于发送 API 请求进行解析的工作线程数量  推荐从 4-10 开始测试，视情况调整。
    max_timeout=7200,  # 7200---增加超时到 2 小时， 以支持大型 PDF 的解析。
    check_interval=10,  # Python 将轮询以检查作业的状态。默认值为 1 秒。建议：5-10 秒较为平衡
    verbose=True,  # 启用详细日志
)
# parser.base_url = "https://api.cloud.llamaindex.ai"  # 强制覆盖
print(f"DEBUG--Final base_url used by LlamaParse: {parser.base_url}")

# 设置文件路径
script_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.abspath(
    # "docs/docs/examples/structured_outputs/data/apple_2021_10k.pdf"
    r"/Users/wingzheng/Downloads/解析结果评测/测试集收集/olmocr 测试文档集2- 精心收集：连续性+签名+多列等/签名/（脱敏版）测试1财务表格6签名页面：税友招股书_（去除打印的名字）.pdf"
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
