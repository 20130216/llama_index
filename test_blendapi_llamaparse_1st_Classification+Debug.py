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
import pdfplumber
from difflib import SequenceMatcher
import json
from typing import List, Dict, Tuple

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
print(f"DEBUG--LLAMA_CLOUD_API_KEY is: {api_key[:4]}...{api_key[-4:]}")

# 统一提示词
USER_PROMPT = (
    "针对本文件，排除所有页码、页眉、页脚，并确保跨页表格合并为单一表格，仅保留一个表头，"
    "完整提取所有表格行数、嵌入式图表和图片内容，避免截断段落或遗漏小字。"
)
SYSTEM_PROMPT_APPEND = (
    "仅提取核心内容，排除页码、页眉、页脚或类似标记（如 '结束页面'、'# 1-1-156'、'Page 1'），"
    "支持多语言噪声，合并跨页表格，仅保留首个表头，忽略重复表头和嵌套表格冗余，"
    "确保完整提取所有表格行数、嵌入式图表和图片内容，避免截断段落或遗漏小字。"
)


def get_file_size_mb(file_path: Path) -> float:
    """获取文件大小（MB）。."""
    return file_path.stat().st_size / (1024 * 1024)


def analyze_table_similarity(tables: List[List[str]]) -> float:
    """计算表格表头相似度。."""
    if len(tables) < 2:
        return 0.0
    similarities = []
    for i in range(len(tables) - 1):
        header1 = " ".join(tables[i][0] if tables[i] else [])
        header2 = " ".join(tables[i + 1][0] if tables[i + 1] else [])
        if header1 and header2:
            similarity = SequenceMatcher(None, header1, header2).ratio()
            similarities.append(similarity)
    return sum(similarities) / len(similarities) if similarities else 0.0


def analyze_pdf_complexity(
    file_path: Path, cache_dir: Path
) -> Tuple[float, bool, bool, Dict]:
    """分析PDF复杂性，返回复杂度得分、是否连续、是否视觉密集及分析细节。."""
    cache_file = cache_dir / f"{file_path.stem}_complexity.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            cached = json.load(f)
        print(f"DEBUG--Loaded complexity cache for {file_path}")
        return (
            cached["score"],
            cached["is_continuous"],
            cached["is_visually_dense"],
            cached["details"],
        )

    try:
        with pdfplumber.open(file_path) as pdf:
            num_pages = len(pdf.pages)
            image_count = 0
            table_count = 0
            multi_column_count = 0
            text_continuity_score = 0
            table_continuity_score = 0
            table_density = 0
            prev_text = ""
            all_tables: List[List[str]] = []

            for i, page in enumerate(pdf.pages):
                # 图像比例
                images = page.images
                image_count += len(images)

                # 表格检测
                tables = page.extract_tables()
                table_count += len(tables)
                for table in tables:
                    if table:
                        all_tables.append(table)
                table_density += sum(
                    len(table) * len(table[0]) if table else 0 for table in tables
                ) / (page.width * page.height + 1e-6)

                # 多列检测
                words = page.extract_words()
                if words:
                    x_coords = [word["x0"] for word in words]
                    if max(x_coords) - min(x_coords) > page.width * 0.5:
                        multi_column_count += 1

                # 文本连续性
                text = page.extract_text() or ""
                if prev_text and text:
                    similarity = SequenceMatcher(None, prev_text, text).ratio()
                    text_continuity_score += similarity
                prev_text = text

            # 表格连续性
            table_continuity_score = analyze_table_similarity(all_tables)

            # 计算复杂度得分
            page_score = min(num_pages / 20, 1) * 10  # 页面数 (10%)
            image_score = min(image_count / (num_pages * 3), 1) * 20  # 图像比例 (20%)
            continuity_score = (
                text_continuity_score / max(1, num_pages - 1) * 0.5
                + table_continuity_score * 0.5
            ) * 40  # 连续性 (40%)
            layout_score = (
                min(table_count / num_pages, 1) * 0.4
                + min(multi_column_count / num_pages, 1) * 0.3
                + min(table_density, 1) * 0.3
            ) * 30  # 布局复杂性 (30%)

            total_score = page_score + image_score + continuity_score + layout_score
            is_continuous = continuity_score > 0.7 * 40  # 连续性得分>70%
            is_visually_dense = (
                image_score > 0.3 * 20 or layout_score > 0.5 * 30
            )  # 图像或布局复杂

            details = {
                "num_pages": num_pages,
                "image_count": image_count,
                "table_count": table_count,
                "multi_column_count": multi_column_count,
                "text_continuity": text_continuity_score / max(1, num_pages - 1),
                "table_continuity": table_continuity_score,
                "table_density": table_density,
                "page_score": page_score,
                "image_score": image_score,
                "continuity_score": continuity_score,
                "layout_score": layout_score,
            }

            # 缓存结果
            cache_data = {
                "score": total_score,
                "is_continuous": is_continuous,
                "is_visually_dense": is_visually_dense,
                "details": details,
            }
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

            print(
                f"DEBUG--Complexity analysis: score={total_score:.2f}, continuous={is_continuous}, visually_dense={is_visually_dense}, details={details}"
            )
            return total_score, is_continuous, is_visually_dense, details

    except Exception as e:
        print(f"WARNING--Complexity analysis failed for {file_path}: {e}")
        return 35, False, False, {"error": str(e)}


def get_parser_config(
    score: float, is_continuous: bool, is_visually_dense: bool
) -> LlamaParse:
    """根据复杂性得分选择解析模式和配置。."""
    parse_mode = "parse_page_with_llm"
    disable_image_extraction = True
    save_images = False

    if score > 65:
        if is_continuous:
            parse_mode = "parse_document_with_llm"
        else:
            parse_mode = (
                "parse_page_with_layout_agent"
                if is_visually_dense
                else "parse_page_with_agent"
            )
            disable_image_extraction = not is_visually_dense
            save_images = is_visually_dense
    elif score > 35:
        parse_mode = (
            "parse_page_with_llm"
            if not is_visually_dense
            else "parse_page_with_layout_agent"
        )
        disable_image_extraction = not is_visually_dense
        save_images = is_visually_dense
    else:
        parse_mode = "parse_page_with_llm"

    print(
        f"DEBUG--Selected parse mode: {parse_mode}, disable_image_extraction={disable_image_extraction}, save_images={save_images}"
    )
    return LlamaParse(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        language="ch_sim",
        parse_mode=parse_mode,
        preserve_layout_alignment_across_pages=True,  # 初始化参数
        spreadsheet_extract_sub_tables=True,
        result_type="markdown",
        disable_image_extraction=disable_image_extraction,
        save_images=save_images,
        user_prompt=USER_PROMPT,
        system_prompt_append=SYSTEM_PROMPT_APPEND,
        num_workers=4,
        max_timeout=7200,
        check_interval=5,  # 延长轮询间隔
        verbose=True,
        do_not_cache=True,
    )


def should_use_sync(file_paths: List[Path]) -> bool:
    """判断是否使用同步解析。."""
    return False  # 强制异步解析


def sync_parse_with_new_loop(parser: LlamaParse, file_path: str) -> List:
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
    file_paths: List[Path], output_dir: Path, input_root: Path, use_sync: bool = False
) -> Tuple[List, Dict]:
    """解析文件列表，逐个处理，确保文件名与内容匹配，保留目录结构。."""
    all_docs = []
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "complexity_cache"
    report = {"files": [], "total_time": 0, "failures": []}

    # 预创建所有子目录
    for file_path in file_paths:
        relative_path = file_path.relative_to(input_root)
        (output_dir / relative_path.parent).mkdir(parents=True, exist_ok=True)

    for file_path in file_paths:
        try:
            start_time = time.time()
            print(f"DEBUG--Parsing file: {file_path}")
            print(f"DEBUG--File size: {get_file_size_mb(file_path):.2f} MB")

            # 复杂性分析
            complexity_score, is_continuous, is_visually_dense, analysis_details = (
                analyze_pdf_complexity(file_path, cache_dir)
            )
            report["files"].append(
                {"file": str(file_path), "complexity_details": analysis_details}
            )

            # 初始化解析器
            parser = get_parser_config(
                complexity_score, is_continuous, is_visually_dense
            )

            # 验证 PDF 文件有效性并记录元数据
            try:
                reader = PdfReader(str(file_path))
                print(f"DEBUG--File pages: {len(reader.pages)}")
                image_count = 0
                for page in reader.pages:
                    resources = page.get("/Resources", {})
                    if isinstance(resources, dict):
                        if "/XObject" in resources:
                            image_count += 1
                    elif hasattr(resources, "get_object"):
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
                report["failures"].append(
                    {"file": str(file_path), "error": f"PDF validation failed: {e!s}"}
                )
                continue

            docs = []
            parse_success = False
            fallback_mode = (
                "parse_page_with_llm"
                if parser.parse_mode != "parse_page_with_llm"
                else None
            )

            for attempt in range(3):
                try:
                    print(
                        f"DEBUG--Attempt {attempt + 1}/3: Starting async parse with mode {parser.parse_mode}"
                    )
                    docs = await parser.aload_data(str(file_path))
                    print(
                        f"DEBUG--Async parse completed in {time.time() - start_time:.2f} seconds"
                    )
                    parse_success = True
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
                        report["failures"].append(
                            {
                                "file": str(file_path),
                                "error": f"Async parse failed: {e!s}",
                            }
                        )
                except Exception as e:
                    print(
                        f"WARNING--Unexpected error during async parsing {file_path}: {e}"
                    )
                    print(f"Stack trace: {traceback.format_exc()}")
                    report["failures"].append(
                        {"file": str(file_path), "error": f"Unexpected error: {e!s}"}
                    )
                    if fallback_mode and attempt == 0:
                        print(f"DEBUG--Falling back to {fallback_mode} for {file_path}")
                        parser = LlamaParse(
                            api_key=api_key,
                            base_url=base_url.rstrip("/"),
                            language="ch_sim",
                            parse_mode=fallback_mode,
                            preserve_layout_alignment_across_pages=True,
                            spreadsheet_extract_sub_tables=True,
                            result_type="markdown",
                            disable_image_extraction=True,
                            save_images=False,
                            user_prompt=USER_PROMPT,
                            system_prompt_append=SYSTEM_PROMPT_APPEND,
                            num_workers=4,
                            max_timeout=7200,
                            check_interval=5,
                            verbose=True,
                            do_not_cache=True,
                        )
                        continue
                    break

            if parse_success:
                all_docs.extend(docs)

            # 计算输出路径
            relative_path = file_path.relative_to(input_root)
            output_file = output_dir / relative_path.with_suffix(".md")

            # 检查有效内容并保存
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
                report["failures"].append(
                    {"file": str(file_path), "error": "No valid content"}
                )

            # 更新报告
            report["files"][-1].update(
                {
                    "complexity_score": complexity_score,
                    "parse_mode": parser.parse_mode,
                    "time": time.time() - start_time,
                    "success": parse_success,
                }
            )

        except RuntimeError as e:
            print(f"WARNING--Event loop error for {file_path}: {e}")
            print(f"Stack trace: {traceback.format_exc()}")
            report["failures"].append(
                {"file": str(file_path), "error": f"Event loop error: {e!s}"}
            )
            continue
        except requests.exceptions.ConnectionError as e:
            print(
                f"Network error: Failed to connect to {parser.base_url}. Please check VPN."
            )
            report["failures"].append(
                {"file": str(file_path), "error": f"Network error: {e!s}"}
            )
            raise
        except requests.exceptions.Timeout as e:
            print(f"Timeout error for {file_path}: Request timed out.")
            report["failures"].append(
                {"file": str(file_path), "error": f"Timeout error: {e!s}"}
            )
            raise
        except Exception as e:
            print(f"WARNING--Error processing {file_path}: {e}")
            print(f"Stack trace: {traceback.format_exc()}")
            report["failures"].append(
                {"file": str(file_path), "error": f"Unexpected error: {e!s}"}
            )
            continue

    return all_docs, report


async def main():
    # 记录开始时间
    start_time = datetime.datetime.now()
    print(f"INFO--Parsing started at {start_time}")

    # 解析命令行参数
    arg_parser = argparse.ArgumentParser(description="Parse PDF files using LlamaParse")
    arg_parser.add_argument("path", type=str, help="Path to PDF file or folder")
    arg_parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of parallel workers"
    )
    arg_parser.add_argument(
        "--max_timeout", type=int, default=7200, help="Maximum timeout in seconds"
    )
    args = arg_parser.parse_args()

    input_path = Path(args.path).resolve()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

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
            # 复杂性分析
            cache_dir = output_dir / "complexity_cache"
            complexity_score, is_continuous, is_visually_dense, analysis_details = (
                analyze_pdf_complexity(input_path, cache_dir)
            )

            # 初始化解析器
            parser = get_parser_config(
                complexity_score, is_continuous, is_visually_dense
            )

            # 验证 PDF 文件有效性
            try:
                reader = PdfReader(str(input_path))
                print(f"DEBUG--File pages: {len(reader.pages)}")
                image_count = 0
                for page in reader.pages:
                    resources = page.get("/Resources", {})
                    if isinstance(resources, dict):
                        if "/XObject" in resources:
                            image_count += 1
                    elif hasattr(resources, "get_object"):
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
            report = {"files": [], "total_time": 0, "failures": []}
            parse_success = False
            fallback_mode = (
                "parse_page_with_llm"
                if parser.parse_mode != "parse_page_with_llm"
                else None
            )

            if use_sync:
                print(f"DEBUG--Using synchronous parsing for {input_path}")
                for attempt in range(3):
                    try:
                        docs = sync_parse_with_new_loop(parser, str(input_path))
                        parse_success = True
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
                            report["failures"].append(
                                {
                                    "file": str(input_path),
                                    "error": f"Sync parse failed: {e!s}",
                                }
                            )
            else:
                print(f"DEBUG--Using asynchronous parsing for {input_path}")
                start_time = time.time()
                for attempt in range(3):
                    try:
                        print(
                            f"DEBUG--Attempt {attempt + 1}/3: Starting async parse with mode {parser.parse_mode}"
                        )
                        docs = await parser.aload_data(str(input_path))
                        print(
                            f"DEBUG--Async parse completed in {time.time() - start_time:.2f} seconds"
                        )
                        parse_success = True
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
                            report["failures"].append(
                                {
                                    "file": str(input_path),
                                    "error": f"Async parse failed: {e!s}",
                                }
                            )
                    except Exception as e:
                        print(
                            f"WARNING--Unexpected error during async parsing {input_path}: {e}"
                        )
                        print(f"Stack trace: {traceback.format_exc()}")
                        report["failures"].append(
                            {
                                "file": str(input_path),
                                "error": f"Unexpected error: {e!s}",
                            }
                        )
                        if fallback_mode and attempt == 0:
                            print(
                                f"DEBUG--Falling back to {fallback_mode} for {input_path}"
                            )
                            parser = LlamaParse(
                                api_key=api_key,
                                base_url=base_url.rstrip("/"),
                                language="ch_sim",
                                parse_mode=fallback_mode,
                                preserve_layout_alignment_across_pages=True,
                                spreadsheet_extract_sub_tables=True,
                                result_type="markdown",
                                disable_image_extraction=True,
                                save_images=False,
                                user_prompt=USER_PROMPT,
                                system_prompt_append=SYSTEM_PROMPT_APPEND,
                                num_workers=4,
                                max_timeout=7200,
                                check_interval=5,
                                verbose=True,
                                do_not_cache=True,
                            )
                            continue
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
                report["failures"].append(
                    {"file": str(input_path), "error": "No valid content"}
                )

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

            # 记录报告
            report["files"].append(
                {
                    "file": str(input_path),
                    "complexity_score": complexity_score,
                    "complexity_details": analysis_details,
                    "parse_mode": parser.parse_mode,
                    "time": time.time() - start_time,
                    "success": parse_success,
                }
            )

        except RuntimeError as e:
            print(f"WARNING--Event loop error for {input_path}: {e}")
            print(f"Stack trace: {traceback.format_exc()}")
            report["failures"].append(
                {"file": str(input_path), "error": f"Event loop error: {e!s}"}
            )
            raise
        except requests.exceptions.ConnectionError as e:
            print(
                f"Network error: Failed to connect to {parser.base_url}. Please check VPN."
            )
            report["failures"].append(
                {"file": str(input_path), "error": f"Network error: {e!s}"}
            )
            raise
        except requests.exceptions.Timeout as e:
            print(f"Timeout error: Request timed out for {input_path}.")
            report["failures"].append(
                {"file": str(input_path), "error": f"Timeout error: {e!s}"}
            )
            raise
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            print(f"Stack trace: {traceback.format_exc()}")
            report["failures"].append(
                {"file": str(input_path), "error": f"Unexpected error: {e!s}"}
            )
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
        all_docs, report = await parse_files(
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
    report["total_time"] = total_time.total_seconds()
    print(f"INFO--Parsing completed at {end_time}")
    print(f"INFO--Total parsing time: {total_time.total_seconds():.2f} seconds")

    # 保存解析报告
    report_file = output_dir / f"parse_report_{timestamp}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"INFO--Parse report saved to {report_file}")


if __name__ == "__main__":
    asyncio.run(main())
