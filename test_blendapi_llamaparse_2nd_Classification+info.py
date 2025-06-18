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
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("parse_debug.log", encoding="utf-8"),
    ],
)
logging.getLogger("PyPDF2").setLevel(logging.INFO)
logging.getLogger("pdfplumber").setLevel(logging.INFO)
logger = logging.getLogger(__name__)
# 确保事件循环兼容性
try:
    import nest_asyncio

    nest_asyncio.apply()
except ImportError:
    logger.warning("nest_asyncio not installed, skipping")

# 调试事件循环
try:
    logger.debug(f"Current event loop: {asyncio.get_event_loop()}")
    logger.debug(f"Is loop running: {asyncio.get_event_loop().is_running()}")
except Exception as e:
    logger.error(f"Event loop error: {e}")

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

logger.debug(f"LLAMA_CLOUD_US_BASE_URL: {base_url}")
logger.debug(f"LLAMA_CLOUD_API_KEY: {api_key[:4]}...{api_key[-4:]}")

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


def analyze_table_similarity(tables: List[List[List[str]]]) -> float:
    """计算表格表头和结构相似度。."""
    if len(tables) < 2:
        return 0.0
    similarities = []
    for i in range(len(tables) - 1):
        table1, table2 = tables[i], tables[i + 1]
        if not table1 or not table2 or not table1[0] or not table2[0]:
            continue
        # 表头相似度
        header1 = [" " if cell is None else str(cell) for cell in table1[0]]
        header2 = [" " if cell is None else str(cell) for cell in table2[0]]
        header_similarity = SequenceMatcher(
            None, " ".join(header1), " ".join(header2)
        ).ratio()
        # 结构相似度（列数）
        col_similarity = 1.0 if len(header1) == len(header2) else 0.5
        similarity = 0.7 * header_similarity + 0.3 * col_similarity
        similarities.append(similarity)
        # logger.debug(f"Table {i} vs {i+1}: header1={header1[:50]}, header2={header2[:50]}, similarity={similarity:.2f}")
    return sum(similarities) / len(similarities) if similarities else 0.0


def analyze_pdf_complexity(
    file_path: Path, cache_dir: Path
) -> Tuple[float, bool, bool, Dict]:
    """分析PDF复杂性，返回复杂度得分、是否连续、是否视觉密集及分析细节。."""
    cache_file = cache_dir / f"{file_path.stem}_complexity.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            cached = json.load(f)
        logger.debug(f"Loaded complexity cache for {file_path}")
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
            all_tables: List[List[List[str]]] = []
            page_details = []

            for i, page in enumerate(pdf.pages):
                page_info = {
                    "page": i + 1,
                    "text_length": 0,
                    "table_rows": 0,
                    "images": 0,
                }
                # 图像比例
                images = page.images
                image_count += len(images)
                page_info["images"] = len(images)

                # 表格检测
                tables = page.extract_tables(
                    table_settings={
                        "vertical_strategy": "lines",
                        "horizontal_strategy": "lines",
                    }
                )
                table_count += len(tables)
                valid_tables = []
                for j, table in enumerate(tables):
                    if table and table[0]:
                        try:
                            valid_tables.append(
                                [
                                    [" " if cell is None else str(cell) for cell in row]
                                    for row in table
                                ]
                            )
                            page_info["table_rows"] += len(table)
                        except Exception as e:
                            logger.warning(f"Table {j} on page {i + 1} invalid: {e}")
                all_tables.extend(valid_tables)
                table_density += sum(
                    len(table) * len(table[0]) if table else 0 for table in tables
                ) / (page.width * page.height + 1e-6)
                page_info["table_count"] = len(valid_tables)

                # 多列检测
                words = page.extract_words()
                if words:
                    x_coords = [word["x0"] for word in words]
                    if max(x_coords) - min(x_coords) > page.width * 0.5:
                        multi_column_count += 1
                page_info["multi_column"] = (
                    max(x_coords) - min(x_coords) > page.width * 0.5 if words else False
                )

                # 文本连续性
                text = page.extract_text() or ""
                page_info["text_length"] = len(text)
                if prev_text and text:
                    similarity = SequenceMatcher(None, prev_text, text).ratio()
                    text_continuity_score += similarity
                    page_info["text_similarity"] = similarity
                prev_text = text
                page_details.append(page_info)
                logger.debug(f"Page {i + 1}: {page_info}")

            # 表格连续性
            table_continuity_score = analyze_table_similarity(all_tables)

            # 计算复杂度得分
            page_score = min(num_pages / 20, 1) * 15  # 页面数 (15%)
            image_score = min(image_count / (num_pages * 3), 1) * 15  # 图像比例 (15%)
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
            is_continuous = continuity_score > 0.7 * 40
            is_visually_dense = image_score > 0.3 * 15 or layout_score > 0.5 * 30

            details = {
                "num_pages": num_pages,
                "image_count": image_count,
                "table_count": table_count,
                "multi_column_count": multi_column_count,
                "text_continuity": text_continuity_score / max(1, num_pages - 1)
                if num_pages > 1
                else 0,
                "table_continuity": table_continuity_score,
                "table_density": table_density,
                "page_score": page_score,
                "image_score": image_score,
                "continuity_score": continuity_score,
                "layout_score": layout_score,
                "page_details": page_details,
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

            # logger.debug(f"Complexity analysis: score={total_score:.2f}, continuous={is_continuous}, visually_dense={is_visually_dense}, details={details}")
            logger.info(
                f"Complexity analysis: score={total_score:.2f}, continuous={is_continuous}, visually_dense={is_visually_dense}"
            )
            return total_score, is_continuous, is_visually_dense, details

    except Exception as e:
        logger.error(f"Complexity analysis failed for {file_path}: {e}")
        details = {
            "error": str(e),
            "page_details": page_details if "page_details" in locals() else [],
        }
        return 60, True, True, details  # 假设复杂文档


def get_parser_config(
    score: float, is_continuous: bool, is_visually_dense: bool
) -> LlamaParse:
    """根据复杂性得分选择解析模式和配置。."""
    parse_mode = "parse_page_with_llm"
    disable_image_extraction = True
    save_images = False

    if score > 60:
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
    elif score > 30:
        parse_mode = (
            "parse_page_with_layout_agent"
            if is_visually_dense
            else "parse_page_with_llm"
        )
        disable_image_extraction = not is_visually_dense
        save_images = is_visually_dense
    else:
        parse_mode = "parse_page_with_llm"

    logger.debug(
        f"Selected parse mode: {parse_mode}, disable_image_extraction={disable_image_extraction}, save_images={save_images}"
    )
    return LlamaParse(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        language="ch_sim",
        parse_mode=parse_mode,
        preserve_layout_alignment_across_pages=True,
        spreadsheet_extract_sub_tables=True,
        result_type="markdown",
        disable_image_extraction=disable_image_extraction,
        save_images=save_images,
        user_prompt=USER_PROMPT,
        system_prompt_append=SYSTEM_PROMPT_APPEND,
        num_workers=4,
        max_timeout=7200,
        check_interval=5,
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
    logger.debug(f"Created new event loop for {file_path}: {loop}")
    try:
        docs = parser.load_data(file_path)
        logger.debug(f"Sync parse completed in {time.time() - start_time:.2f} seconds")
        return docs
    except Exception as e:
        logger.error(f"Sync parse failed: {type(e).__name__}: {e}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise
    finally:
        logger.debug(f"Closing event loop for {file_path}: {loop}")
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
            logger.debug(f"Parsing file: {file_path}")
            logger.debug(f"File size: {get_file_size_mb(file_path):.2f} MB")

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
                logger.debug(f"File pages: {len(reader.pages)}")
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
                logger.debug(f"Image count: {image_count}")
            except Exception as e:
                logger.warning(f"Invalid PDF file {file_path}: {e}")
                logger.warning(f"Stack trace: {traceback.format_exc()}")
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
                    logger.debug(
                        f"Attempt {attempt + 1}/3: Starting async parse with mode {parser.parse_mode}"
                    )
                    docs = await parser.aload_data(str(file_path))
                    logger.debug(
                        f"Async parse completed in {time.time() - start_time:.2f} seconds"
                    )
                    parse_success = True
                    logger.debug(
                        f"Parsed docs: {len(docs)} pages, first doc text preview: {docs[0].text[:50] if docs and hasattr(docs[0], 'text') else 'N/A'}"
                    )
                    break
                except (RuntimeError, asyncio.CancelledError) as e:
                    if attempt < 2:
                        logger.debug(
                            f"Retrying {file_path} (attempt {attempt + 2}/3): {type(e).__name__}: {e}"
                        )
                        logger.debug(f"Stack trace: {traceback.format_exc()}")
                        await asyncio.sleep(2)
                    else:
                        logger.warning(
                            f"Async failed after 3 attempts for {file_path}: {e}"
                        )
                        logger.warning(f"Stack trace: {traceback.format_exc()}")
                        report["failures"].append(
                            {
                                "file": str(file_path),
                                "error": f"Async parse failed: {e!s}",
                            }
                        )
                except Exception as e:
                    logger.warning(
                        f"Unexpected error during async parsing {file_path}: {e}"
                    )
                    logger.warning(f"Stack trace: {traceback.format_exc()}")
                    report["failures"].append(
                        {"file": str(file_path), "error": f"Unexpected error: {e!s}"}
                    )
                    if fallback_mode and attempt == 0:
                        logger.debug(f"Falling back to {fallback_mode} for {file_path}")
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
                logger.debug(f"Saved output to {output_file}")
                with open(output_file, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                logger.debug(f"Output file hash: {file_hash}")
            else:
                logger.warning(f"No valid content for {file_path}, skipping output")
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
                    "doc_count": len(docs),
                }
            )

        except RuntimeError as e:
            logger.warning(f"Event loop error for {file_path}: {e}")
            logger.warning(f"Stack trace: {traceback.format_exc()}")
            report["failures"].append(
                {"file": str(file_path), "error": f"Event loop error: {e!s}"}
            )
            continue
        except requests.exceptions.ConnectionError as e:
            logger.error(
                f"Network error: Failed to connect to {parser.base_url}. Please check VPN."
            )
            report["failures"].append(
                {"file": str(file_path), "error": f"Network error: {e!s}"}
            )
            raise
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout error for {file_path}: Request timed out.")
            report["failures"].append(
                {"file": str(file_path), "error": f"Timeout error: {e!s}"}
            )
            raise
        except Exception as e:
            logger.warning(f"Error processing {file_path}: {e}")
            logger.warning(f"Stack trace: {traceback.format_exc()}")
            report["failures"].append(
                {"file": str(file_path), "error": f"Unexpected error: {e!s}"}
            )
            continue

    return all_docs, report


async def main():
    # 记录开始时间
    start_time = datetime.datetime.now()
    logger.info(f"Parsing started at {start_time}")

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

        logger.info(f"Processing file: {input_path}")
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
                logger.debug(f"File pages: {len(reader.pages)}")
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
                logger.debug(f"Image count: {image_count}")
            except Exception as e:
                logger.warning(f"Invalid PDF file {input_path}: {e}")
                logger.warning(f"Stack trace: {traceback.format_exc()}")
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
                logger.debug(f"Using synchronous parsing for {input_path}")
                for attempt in range(3):
                    try:
                        docs = sync_parse_with_new_loop(parser, str(input_path))
                        parse_success = True
                        break
                    except Exception as e:
                        if attempt < 2:
                            logger.debug(
                                f"Retrying {input_path} (attempt {attempt + 2}/3): {type(e).__name__}: {e}"
                            )
                            logger.debug(f"Stack trace: {traceback.format_exc()}")
                            time.sleep(2)
                        else:
                            logger.warning(
                                f"Sync failed after 3 attempts for {input_path}: {e}"
                            )
                            logger.warning(f"Stack trace: {traceback.format_exc()}")
                            report["failures"].append(
                                {
                                    "file": str(input_path),
                                    "error": f"Sync parse failed: {e!s}",
                                }
                            )
            else:
                logger.debug(f"Using asynchronous parsing for {input_path}")
                start_time = time.time()
                for attempt in range(3):
                    try:
                        logger.debug(
                            f"Attempt {attempt + 1}/3: Starting async parse with mode {parser.parse_mode}"
                        )
                        docs = await parser.aload_data(str(input_path))
                        logger.debug(
                            f"Async parse completed in {time.time() - start_time:.2f} seconds"
                        )
                        parse_success = True
                        logger.debug(
                            f"Parsed docs: {len(docs)} pages, first doc text preview: {docs[0].text[:50] if docs and hasattr(docs[0], 'text') else 'N/A'}"
                        )
                        break
                    except (RuntimeError, asyncio.CancelledError) as e:
                        if attempt < 2:
                            logger.debug(
                                f"Retrying {input_path} (attempt {attempt + 2}/3): {type(e).__name__}: {e}"
                            )
                            logger.debug(f"Stack trace: {traceback.format_exc()}")
                            await asyncio.sleep(2)
                        else:
                            logger.warning(
                                f"Async failed after 3 attempts for {input_path}: {e}"
                            )
                            logger.warning(f"Stack trace: {traceback.format_exc()}")
                            report["failures"].append(
                                {
                                    "file": str(input_path),
                                    "error": f"Async parse failed: {e!s}",
                                }
                            )
                    except Exception as e:
                        logger.warning(
                            f"Unexpected error during async parsing {input_path}: {e}"
                        )
                        logger.warning(f"Stack trace: {traceback.format_exc()}")
                        report["failures"].append(
                            {
                                "file": str(input_path),
                                "error": f"Unexpected error: {e!s}",
                            }
                        )
                        if fallback_mode and attempt == 0:
                            logger.debug(
                                f"Falling back to {fallback_mode} for {input_path}"
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
                logger.info(f"Output saved to: {output_file}")
                with open(output_file, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                logger.debug(f"Output file hash: {file_hash}")
            else:
                logger.warning(f"No valid content for {input_path}, skipping output")
                report["failures"].append(
                    {"file": str(input_path), "error": "No valid content"}
                )

            # 调试信息
            for i, doc in enumerate(docs):
                text = doc.text.strip() if hasattr(doc, "text") and doc.text else ""
                text_preview = text[:50] if text else "Empty text"
                status = (
                    " (full text)"
                    if len(text) <= 50
                    else f" ({len(text_preview)} chars)"
                )
                logger.debug(
                    f"Document {i + 1}: ID={doc.id_}, Text Preview={text_preview}{status}"
                )

            # 文本长度统计
            text_lengths = [
                len(doc.text.strip())
                for doc in docs
                if hasattr(doc, "text") and doc.text
            ]
            if text_lengths:
                logger.info(
                    f"Text stats: min={min(text_lengths)}, max={max(text_lengths)}, avg={sum(text_lengths) / len(text_lengths):.2f}"
                )
            else:
                logger.info("No text extracted, stats unavailable")

            # 记录报告
            report["files"].append(
                {
                    "file": str(input_path),
                    "complexity_score": complexity_score,
                    "complexity_details": analysis_details,
                    "parse_mode": parser.parse_mode,
                    "time": time.time() - start_time,
                    "success": parse_success,
                    "doc_count": len(docs),
                }
            )

        except RuntimeError as e:
            logger.error(f"Event loop error for {input_path}: {e}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            report["files"].append(
                {"file": str(input_path), "error": f"Event loop error: {e!s}"}
            )
        except requests.exceptions.ConnectionError as e:
            logger.error(
                f"Network error: Failed to connect to {parser.base_url}. Please check VPN."
            )
            report["failures"].append(
                {"file": str(input_path), "error": f"Network error: {e!s}"}
            )
            raise
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout error: Request timed out for {input_path}.")
            report["failures"].append(
                {"file": str(input_path), "error": f"Timeout error: {e!s}"}
            )
            raise
        except Exception as e:
            logger.error(f"Error processing {input_path}: {e}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            report["failures"].append(
                {"file": str(input_path), "error": f"Unexpected error: {e!s}"}
            )
            raise

    # 处理文件夹
    elif input_path.is_dir():
        logger.info(f"Processing folder: {input_path}")
        pdf_files = sorted(input_path.rglob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in {input_path}")

        output_dir = input_path.parent / f"{input_path.name}_{timestamp}"
        if output_dir.exists():
            logger.info(f"Overwriting existing folder: {output_dir}")
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
            logger.info(
                f"Stats: min={min(text_lengths)}, max={max(text_lengths)}, text avg={sum(text_lengths) / len(text_lengths):.2f}"
            )
        else:
            logger.info("No text extracted, stats unavailable")

    # 记录结束时间和总耗时
    end_time = datetime.datetime.now()
    total_time = end_time - start_time
    report["total_time"] = total_time.total_seconds()
    logger.info(f"Parsing completed at {end_time}")
    logger.info(f"Total parsing time: {total_time.total_seconds():.2f} seconds")

    # 保存解析报告
    report_file = output_dir / f"parse_report_{timestamp}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"Parse report saved to {report_file}")


if __name__ == "__main__":
    asyncio.run(main())
