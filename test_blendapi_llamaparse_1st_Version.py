from llama_parse import LlamaParse
import os
import asyncio
import argparse
from dotenv import load_dotenv
import datetime
import shutil
import requests
import hashlib
from pathlib import Path
from PyPDF2 import PdfReader
import traceback
import time

# 确保事件循环兼容性
try:
    import nest_asyncio

    nest_asyncio.apply()
except ImportError:
    print("nest_asyncio not installed, skipping")

# 调试事件循环
try:
    print(f"DEBUG--Current event loop: {asyncio.get_event_loop()}")
    print(f"DEBUG--Is loop running: {asyncio.get_event_loop().is_running()}")
except Exception as e:
    print(f"DEBUG--Event loop error: {e}")

# 加载环境变量
load_dotenv(
    dotenv_path="/Users/wingzheng/Desktop/github/ParseDoc/llama_index/.env",
    override=True,
)

# 获取 LlamaParse 配置
base_url = os.getenv("LLAMA_CLOUD_US_BASE_URL", "https://api.cloud.llamaindex.ai")
api_key = os.getenv("LLAMA_CLOUD_API_KEY")

# 验证环境变量
if not base_url or not base_url.startswith(("http://", "https://")):
    raise ValueError(f"Invalid base_url: {base_url}")
if not api_key:
    raise ValueError("API key is missing")

print(f"DEBUG--LLAMA_CLOUD_US_BASE_URL is: {base_url}")
print(f"DEBUG--LLAMA_CLOUD_API_KEY is: {api_key}")

# 初始化 LlamaParse
parser = LlamaParse(
    api_key=api_key,
    base_url=base_url.rstrip("/"),
    language="ch_sim",
    # parse_mode="parse_document_with_llm",
    parse_mode="parse_page_with_llm",
    preserve_layout_alignment_across_pages=True,
    spreadsheet_extract_sub_tables=True,
    result_type="markdown",
    # disable_image_extraction=False,
    # save_images=True,
    disable_image_extraction=True,
    user_prompt="针对本文件，排除所有页码、页眉、页脚，并确保跨页表格合并为单一表格，仅保留一个表头。",
    system_prompt_append="仅提取内容，排除页码、页眉、页脚或类似 '结束页面' 或 '# 1-1-156' 的标记，将跨页表格合并为单一表格，仅保留第一个表头，忽略后续重复表头。",
    num_workers=10,
    max_timeout=3600,
    check_interval=3,  # 优化轮询
    verbose=True,
    do_not_cache=True,
)
print(f"DEBUG--Final base_url used by LlamaParse: {parser.base_url}")


def get_file_size_mb(file_path: Path) -> float:
    """获取文件大小（MB）。."""
    return file_path.stat().st_size / (1024 * 1024)


def should_use_sync(file_paths: list[Path]) -> bool:
    """判断是否使用同步解析。."""
    return False  # 强制异步解析


def sync_parse_with_new_loop(parser, file_path):
    """为同步解析创建新的事件循环。."""
    start_time = time.time()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    print(f"DEBUG--Created new event loop for {file_path}: {loop}")
    try:
        docs = parser.load_data(file_path)
        print(f"DEBUG--Sync parse completed in {time.time() - start_time:.2f} seconds")
        return docs
    except Exception as e:
        print(f"DEBUG--Sync parse failed: {type(e).__name__}: {e}")
        print(f"Stack trace: {traceback.format_exc()}")
        raise
    finally:
        print(f"DEBUG--Closing event loop for {file_path}: {loop}")
        loop.close()


async def parse_files(
    file_paths: list[Path], output_dir: Path, input_root: Path, use_sync: bool = False
) -> list:
    """
    解析文件列表，逐个处理，确保文件名与内容匹配，保留目录结构。
    Args:
        file_paths: PDF 文件路径列表。
        output_dir: 输出根目录。
        input_root: 输入文件夹根目录。
        use_sync: 是否使用同步解析。
    Returns:
        List of parsed documents.
    """
    all_docs = []
    output_dir.mkdir(parents=True, exist_ok=True)

    # 预创建所有子目录
    for file_path in file_paths:
        relative_path = file_path.relative_to(input_root)
        (output_dir / relative_path.parent).mkdir(parents=True, exist_ok=True)

    for file_path in file_paths:
        try:
            print(f"DEBUG--Parsing file: {file_path}")
            print(f"DEBUG--File size: {get_file_size_mb(file_path):.2f} MB")

            # 验证 PDF 文件有效性并记录元数据
            try:
                reader = PdfReader(str(file_path))
                print(f"DEBUG--File pages: {len(reader.pages)}")
                # 检查是否有图像
                image_count = 0
                for page in reader.pages:
                    resources = page.get("/Resources", {})
                    if isinstance(resources, dict):
                        if "/XObject" in resources:
                            image_count += 1
                    elif hasattr(resources, "get_object"):  # 处理 IndirectObject
                        resolved_resources = resources.get_object()
                        if (
                            isinstance(resolved_resources, dict)
                            and "/XObject" in resolved_resources
                        ):
                            image_count += 1
                print(f"DEBUG--Image count: {image_count}")
            except Exception as e:
                print(f"WARNING--Invalid PDF file {file_path}: {e}")
                print(f"Stack trace: {traceback.format_exc()}")
                continue

            docs = []
            if use_sync:
                print(f"DEBUG--Using synchronous parsing for {file_path}")
                for attempt in range(3):
                    try:
                        docs = sync_parse_with_new_loop(parser, str(file_path))
                        break
                    except Exception as e:
                        if attempt < 2:
                            print(
                                f"DEBUG--Retrying {file_path} (attempt {attempt + 2}/3): {type(e).__name__}: {e}"
                            )
                            print(f"Stack trace: {traceback.format_exc()}")
                            time.sleep(2)
                        else:
                            print(
                                f"WARNING--Sync failed after 3 attempts for {file_path}: {e}"
                            )
                            print(f"Stack trace: {traceback.format_exc()}")
                            docs = []
            else:
                print(f"DEBUG--Using asynchronous parsing for {file_path}")
                start_time = time.time()
                for attempt in range(3):
                    try:
                        print(f"DEBUG--Attempt {attempt + 1}/3: Starting async parse")
                        docs = await asyncio.ensure_future(
                            parser.aload_data(str(file_path))
                        )
                        print(
                            f"DEBUG--Async parse completed in {time.time() - start_time:.2f} seconds"
                        )
                        break
                    except (RuntimeError, asyncio.CancelledError) as e:
                        if attempt < 2:
                            print(
                                f"DEBUG--Retrying {file_path} (attempt {attempt + 2}/3): {type(e).__name__}: {e}"
                            )
                            print(f"Stack trace: {traceback.format_exc()}")
                            await asyncio.sleep(2)
                        else:
                            print(
                                f"WARNING--Async failed after 3 attempts for {file_path}: {e}"
                            )
                            print(f"Stack trace: {traceback.format_exc()}")
                            docs = []
                    except Exception as e:
                        print(
                            f"WARNING--Unexpected error during async parsing {file_path}: {e}"
                        )
                        print(f"Stack trace: {traceback.format_exc()}")
                        docs = []
                        break

            all_docs.extend(docs)

            # 计算输出路径
            relative_path = file_path.relative_to(input_root)
            output_file = output_dir / relative_path.with_suffix(".md")

            # 检查有效内容
            if docs and any(hasattr(doc, "text") and doc.text.strip() for doc in docs):
                with open(output_file, "w", encoding="utf-8") as f:
                    for i, doc in enumerate(docs):
                        text = (
                            doc.text.strip()
                            if hasattr(doc, "text") and doc.text
                            else ""
                        )
                        f.write(f"## Page {i + 1}\n\n{text}\n\n")
                print(f"DEBUG--Saved output to {output_file}")
                with open(output_file, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                print(f"DEBUG--Output file hash: {file_hash}")
            else:
                print(f"WARNING--No valid content for {file_path}, skipping output")

        except RuntimeError as e:
            print(f"WARNING--Event loop error for {file_path}: {e}")
            print(f"Stack trace: {traceback.format_exc()}")
            continue
        except requests.exceptions.ConnectionError:
            print(
                f"Network error: Failed to connect to {parser.base_url}. Please check VPN."
            )
            raise
        except requests.exceptions.Timeout:
            print(f"Timeout error for {file_path}: Request timed out.")
            raise
        except Exception as e:
            print(f"WARNING--Error processing {file_path}: {e}")
            print(f"Stack trace: {traceback.format_exc()}")
            continue

    return all_docs


async def main():
    # 记录开始时间
    start_time = datetime.datetime.now()
    print(f"INFO--Parsing started at {start_time}")

    # 解析命令行参数
    arg_parser = argparse.ArgumentParser(description="Parse PDF files using LlamaParse")
    arg_parser.add_argument("path", type=str, help="Path to PDF file or folder")
    args = arg_parser.parse_args()

    input_path = Path(args.path).resolve()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")  # 修改为 YYYYMMDD_HHMM

    if not input_path.exists():
        raise FileNotFoundError(f"Path {input_path} does not exist")

    # 处理单个文件
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"File {input_path} is not a PDF file")

        print(f"Processing file: {input_path}")
        output_dir = input_path.parent
        output_file = output_dir / f"{input_path.stem}_{timestamp}.md"

        try:
            print(f"DEBUG--File size: {get_file_size_mb(input_path):.2f} MB")
            try:
                reader = PdfReader(str(input_path))
                print(f"DEBUG--File pages: {len(reader.pages)}")
                # 检查是否有图像
                image_count = 0
                for page in reader.pages:
                    resources = page.get("/Resources", {})
                    if isinstance(resources, dict):
                        if "/XObject" in resources:
                            image_count += 1
                    elif hasattr(resources, "get_object"):  # 处理 IndirectObject
                        resolved_resources = resources.get_object()
                        if (
                            isinstance(resolved_resources, dict)
                            and "/XObject" in resolved_resources
                        ):
                            image_count += 1
                print(f"DEBUG--Image count: {image_count}")
            except Exception as e:
                print(f"WARNING--Invalid PDF file {input_path}: {e}")
                print(f"Stack trace: {traceback.format_exc()}")
                raise

            use_sync = should_use_sync([input_path])
            docs = []
            if use_sync:
                print(f"DEBUG--Using synchronous parsing for {input_path}")
                for attempt in range(3):
                    try:
                        docs = sync_parse_with_new_loop(parser, str(input_path))
                        break
                    except Exception as e:
                        if attempt < 2:
                            print(
                                f"DEBUG--Retrying {input_path} (attempt {attempt + 2}/3): {type(e).__name__}: {e}"
                            )
                            print(f"Stack trace: {traceback.format_exc()}")
                            time.sleep(2)
                        else:
                            print(
                                f"WARNING--Sync failed after 3 attempts for {input_path}: {e}"
                            )
                            print(f"Stack trace: {traceback.format_exc()}")
                            docs = []
            else:
                print(f"DEBUG--Using asynchronous parsing for {input_path}")
                start_time = time.time()
                for attempt in range(3):
                    try:
                        print(f"DEBUG--Attempt {attempt + 1}/3: Starting async parse")
                        docs = await asyncio.ensure_future(
                            parser.aload_data(str(input_path))
                        )
                        print(
                            f"DEBUG--Async parse completed in {time.time() - start_time:.2f} seconds"
                        )
                        break
                    except (RuntimeError, asyncio.CancelledError) as e:
                        if attempt < 2:
                            print(
                                f"DEBUG--Retrying {input_path} (attempt {attempt + 2}/3): {type(e).__name__}: {e}"
                            )
                            print(f"Stack trace: {traceback.format_exc()}")
                            await asyncio.sleep(2)
                        else:
                            print(
                                f"WARNING--Async failed after 3 attempts for {input_path}: {e}"
                            )
                            print(f"Stack trace: {traceback.format_exc()}")
                            docs = []
                    except Exception as e:
                        print(
                            f"WARNING--Unexpected error during async parsing {input_path}: {e}"
                        )
                        print(f"Stack trace: {traceback.format_exc()}")
                        docs = []
                        break

            if docs and any(hasattr(doc, "text") and doc.text.strip() for doc in docs):
                with open(output_file, "w", encoding="utf-8") as f:
                    for i, doc in enumerate(docs):
                        text = (
                            doc.text.strip()
                            if hasattr(doc, "text") and doc.text
                            else ""
                        )
                        f.write(f"## Page {i + 1}\n\n{text}\n\n")
                print(f"Output saved to: {output_file}")
                with open(output_file, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                print(f"DEBUG--Output file hash: {file_hash}")
            else:
                print(f"WARNING--No valid content for {input_path}, skipping output")

            # 调试信息
            for i, doc in enumerate(docs):
                text = doc.text.strip() if hasattr(doc, "text") and doc.text else ""
                text_preview = text[:10] if text else "Empty text"
                status = (
                    " (full text)"
                    if len(text) <= 10
                    else f" ({len(text_preview)} chars)"
                )
                print(
                    f"Document {i + 1}: ID={doc.id_}, Text Preview={text_preview}{status}"
                )

            # 文本长度统计
            text_lengths = [
                len(doc.text.strip())
                for doc in docs
                if hasattr(doc, "text") and doc.text
            ]
            if text_lengths:
                print(
                    f"Text stats: min={min(text_lengths)}, max={max(text_lengths)}, avg={sum(text_lengths) / len(text_lengths):.2f}"
                )
            else:
                print("No text extracted, stats unavailable")

        except RuntimeError as e:
            print(f"WARNING--Event loop error for {input_path}: {e}")
            print(f"Stack trace: {traceback.format_exc()}")
            raise
        except requests.exceptions.ConnectionError:
            print(
                f"Network error: Failed to connect to {parser.base_url}. Please check VPN."
            )
            raise
        except requests.exceptions.Timeout:
            print(f"Timeout error: Request timed out for {input_path}.")
            raise
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            print(f"Stack trace: {traceback.format_exc()}")
            raise

    # 处理文件夹
    elif input_path.is_dir():
        print(f"Processing folder: {input_path}")
        pdf_files = sorted(input_path.rglob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in {input_path}")

        output_dir = input_path.parent / f"{input_path.name}_{timestamp}"
        if output_dir.exists():
            print(f"Overwriting existing folder: {output_dir}")
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

        use_sync = should_use_sync(pdf_files)
        all_docs = await parse_files(
            pdf_files, output_dir, input_root=input_path, use_sync=use_sync
        )

        # 文本长度统计
        text_lengths = [
            len(doc.text.strip())
            for doc in all_docs
            if hasattr(doc, "text") and doc.text
        ]
        if text_lengths:
            print(
                f"Stats: min={min(text_lengths)}, max={max(text_lengths)}, text avg={sum(text_lengths) / len(text_lengths):.2f}"
            )
        else:
            print("No text extracted, stats unavailable")

    # 记录结束时间和总耗时
    end_time = datetime.datetime.now()
    total_time = end_time - start_time
    print(f"INFO--Parsing completed at {end_time}")
    print(f"INFO--Total parsing time: {total_time.total_seconds():.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())
