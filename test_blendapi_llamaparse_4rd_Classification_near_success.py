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
import time
import pdfplumber
from difflib import SequenceMatcher
import re
import json
from typing import List, Dict, Tuple
import logging
from tabulate import tabulate
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("parse_debug.log", encoding="utf-8"),
    ],
)
logging.getLogger("PyPDF2").setLevel(logging.INFO)
logging.getLogger("pdfplumber").setLevel(logging.INFO)
logging.getLogger("llama_parse").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


# 自动下载 NLTK 资源
def download_nltk_resources():
    """下载必要的 NLTK 资源，确保 punkt 和 punkt_tab 可用。."""
    try:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("averaged_perceptron_tagger", quiet=True)
        logger.info("NLTK resources downloaded successfully")
    except Exception as e:
        logger.error(f"Failed to download NLTK resources: {e}")
        logger.error("Please run 'import nltk; nltk.download('punkt_tab')' manually")


download_nltk_resources()

# 确保事件循环兼容性
try:
    import nest_asyncio

    nest_asyncio.apply()
except ImportError:
    logger.warning("nest_asyncio not installed, skipping")

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


# 验证配置文件
def validate_config(config: Dict) -> Dict:
    """验证配置文件，确保权重和阈值合法。."""
    weights = config.get("weights", {})
    thresholds = config.get("thresholds", {})
    continuity = config.get("continuity", {})
    layout = config.get("layout", {})
    config.get("image", {})

    # 验证权重和
    total_weight = sum(
        weights.get(key, 0)
        for key in ["page", "image", "continuity", "layout", "text_complexity"]
    )
    if not 0.95 <= total_weight <= 1.05:
        logger.warning(f"Total weights sum {total_weight} not ≈ 1, normalizing")
        for key in weights:
            weights[key] /= total_weight

    # 验证阈值递增
    if not (
        thresholds["simple"] < thresholds["medium_low"] < thresholds["medium_high"]
    ):
        raise ValueError(f"Invalid thresholds: {thresholds}")

    # 验证子权重
    if (
        not 0.95
        <= sum(continuity.get(key, 0) for key in ["text_weight", "table_weight"])
        <= 1.05
    ):
        raise ValueError(f"Invalid continuity weights: {continuity}")
    if (
        not 0.95
        <= sum(
            layout.get(key, 0)
            for key in ["table_weight", "multi_column_weight", "density_weight"]
        )
        <= 1.05
    ):
        raise ValueError(f"Invalid layout weights: {layout}")

    return config


# 加载配置文件
def load_config(config_path: str = "config.json") -> Dict:
    """加载配置文件，验证格式并返回默认配置（若文件不存在）。."""
    config_dir = Path("test_blendapi_config")
    config_file = Path(config_path)

    # 优先检查 test_blendapi_config 文件夹
    if config_dir.exists() and (config_dir / config_file).exists():
        config_file = config_dir / config_file
    elif not config_file.exists():
        logger.warning(
            f"Config file {config_path} not found, using default configuration"
        )
        logger.info(
            "Please check 'test_blendapi_config' folder for available configs (e.g., config1.json for financial tables)"
        )
        config = {
            "weights": {
                "page": 0.20,
                "image": 0.20,
                "continuity": 0.30,
                "layout": 0.30,
                "text_complexity": 0.05,
            },
            "thresholds": {"simple": 20, "medium_low": 35, "medium_high": 50},
            "continuity": {"text_weight": 0.4, "table_weight": 0.6},
            "layout": {
                "table_weight": 0.45,
                "multi_column_weight": 0.25,
                "density_weight": 0.30,
            },
            "image": {"images_per_page_cap": 2, "chart_weight": 1.5},
        }
        return validate_config(config)

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        config = validate_config(config)
        logger.info(f"Loaded configuration from {config_file}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config {config_file}: {e}")
        raise


config = load_config()

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
        header1 = [" " if cell is None else str(cell) for cell in table1[0]]
        header2 = [" " if cell is None else str(cell) for cell in table2[0]]
        header_similarity = SequenceMatcher(
            None, " ".join(header1), " ".join(header2)
        ).ratio()
        col_similarity = 1.0 if len(header1) == len(header2) else 0.5
        similarity = 0.7 * header_similarity + 0.3 * col_similarity
        similarities.append(similarity)
    return sum(similarities) / len(similarities) if similarities else 0.0


def compute_text_complexity(text: str) -> float:
    """计算文本复杂度，基于句子长度和词汇多样性，带回退机制。."""
    try:
        sentences = sent_tokenize(text)
        if not sentences:
            return 0.5  # 回退默认值
        avg_sentence_length = sum(len(word_tokenize(sent)) for sent in sentences) / len(
            sentences
        )
        words = word_tokenize(text.lower())
        unique_words = len(set(words))
        vocab_diversity = unique_words / len(words) if words else 0.0
        return (
            min(avg_sentence_length / 20, 1) * 0.6 + min(vocab_diversity / 0.5, 1) * 0.4
        )
    except Exception as e:
        logger.warning(f"Text complexity computation failed: {e}")
        logger.info(
            "Try running 'import nltk; nltk.download('punkt_tab')' to resolve NLTK resource issues"
        )
        return 0.5  # 回退默认值


def analyze_pdf_complexity(
    file_path: Path, cache_dir: Path
) -> Tuple[float, bool, bool, Dict]:
    """分析PDF复杂性，返回复杂度得分、是否连续、是否视觉密集及分析细节。."""
    try:
        with pdfplumber.open(file_path) as pdf:
            num_pages = len(pdf.pages)
            image_count = 0
            chart_count = 0
            table_count = 0
            multi_column_count = 0
            text_continuity_score = 0
            table_continuity_score = 0
            table_density = 0
            text_complexity_score = 0
            prev_text = ""
            all_tables: List[List[List[str]]] = []
            page_details = []

            for i, page in enumerate(pdf.pages):
                page_info = {
                    "page": i + 1,
                    "text_length": 0,
                    "table_rows": 0,
                    "images": 0,
                    "charts": 0,
                }
                images = page.images
                image_count += len(images)
                page_info["images"] = len(images)

                # 检测复杂图表
                for img in images:
                    img_bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
                    img_text = page.within_bbox(img_bbox).extract_text() or ""
                    if (
                        re.search(r"\b\d+\b.*\b\d+\b", img_text)
                        or len(img_text.split()) > 5
                    ):
                        chart_count += 1
                        page_info["charts"] += 1

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

                words = page.extract_words()
                if words:
                    x_coords = [word["x0"] for word in words]
                    if max(x_coords) - min(x_coords) > page.width * 0.5:
                        multi_column_count += 1
                page_info["multi_column"] = (
                    max(x_coords) - min(x_coords) > page.width * 0.5 if words else False
                )

                text = page.extract_text() or ""
                page_info["text_length"] = len(text)
                text_complexity_score += compute_text_complexity(text)
                if prev_text and text:
                    prev_words = set(word_tokenize(text.lower()))
                    curr_words = set(word_tokenize(text.lower()))
                    keyword_overlap = (
                        len(prev_words & curr_words) / len(prev_words | curr_words)
                        if prev_words | curr_words
                        else 0
                    )
                    similarity = SequenceMatcher(None, prev_text, text).ratio()
                    text_continuity_score += 0.7 * similarity + 0.3 * keyword_overlap
                prev_text = text
                page_details.append(page_info)

            table_continuity_score = analyze_table_similarity(all_tables)
            table_density = table_density / num_pages if num_pages else 0
            text_complexity_score = (
                text_complexity_score / num_pages if num_pages else 0
            )

            page_score = min(num_pages / 20, 1) * config["weights"]["page"] * 100
            image_score = (
                min(
                    (image_count + config["image"]["chart_weight"] * chart_count)
                    / (num_pages * config["image"]["images_per_page_cap"]),
                    1,
                )
                * config["weights"]["image"]
                * 100
            )
            continuity_score = (
                (
                    text_continuity_score
                    / max(1, num_pages - 1)
                    * config["continuity"]["text_weight"]
                    + table_continuity_score * config["continuity"]["table_weight"]
                )
                * config["weights"]["continuity"]
                * 100
            )
            layout_score = (
                (
                    min(table_count / num_pages, 1) * config["layout"]["table_weight"]
                    + min(multi_column_count / num_pages, 1)
                    * config["layout"]["multi_column_weight"]
                    + min(table_density, 1) * config["layout"]["density_weight"]
                )
                * config["weights"]["layout"]
                * 100
            )
            text_complexity_score = (
                min(text_complexity_score, 1)
                * config["weights"]["text_complexity"]
                * 100
            )

            total_score = (
                page_score
                + image_score
                + continuity_score
                + layout_score
                + text_complexity_score
            )
            is_continuous = (
                continuity_score > 0.7 * config["weights"]["continuity"] * 100
            )
            is_visually_dense = (
                image_score > 0.3 * config["weights"]["image"] * 100
                or layout_score > 0.5 * config["weights"]["layout"] * 100
            )

            # 强制最低分值
            if is_visually_dense and total_score < config["thresholds"]["simple"]:
                total_score = config["thresholds"]["simple"]
            if is_continuous and total_score < config["thresholds"]["medium_low"]:
                total_score = config["thresholds"]["medium_low"]

            details = {
                "num_pages": num_pages,
                "image_count": image_count,
                "chart_count": chart_count,
                "table_count": table_count,
                "multi_column_count": multi_column_count,
                "text_continuity": text_continuity_score / max(1, num_pages - 1)
                if num_pages > 1
                else 0,
                "table_continuity": table_continuity_score,
                "table_density": table_density,
                "text_complexity": text_complexity_score,
                "page_score": page_score,
                "image_score": image_score,
                "continuity_score": continuity_score,
                "layout_score": layout_score,
                "text_complexity_score": text_complexity_score,
                "page_details": page_details,
                "is_continuous": is_continuous,
                "is_visually_dense": is_visually_dense,
            }

            logger.info(
                f"Complexity analysis for {file_path.name}: score={total_score:.2f}, continuous={is_continuous}, visually_dense={is_visually_dense}"
            )
            logger.debug(
                f"Score breakdown: page={page_score:.2f}, image={image_score:.2f}, continuity={continuity_score:.2f}, layout={layout_score:.2f}, text_complexity={text_complexity_score:.2f}"
            )

            return total_score, is_continuous, is_visually_dense, details

    except Exception as e:
        logger.error(f"Complexity analysis failed for {file_path}: {e}")
        file_size_mb = get_file_size_mb(file_path)
        try:
            with PdfReader(str(file_path)) as reader:
                num_pages = len(reader.pages)
        except Exception:
            num_pages = 1
        estimated_score = min(30 + file_size_mb * 5 + num_pages * 2, 60)
        details = {
            "error": str(e),
            "page_details": [],
            "estimated_score": estimated_score,
            "num_pages": num_pages,
            "file_size_mb": file_size_mb,
        }
        logger.info(
            f"Estimated complexity score for {file_path}: {estimated_score:.2f}"
        )
        return estimated_score, True, True, details


def get_parser_config(
    score: float, is_continuous: bool, is_visually_dense: bool
) -> Tuple[LlamaParse, str]:
    """根据复杂性得分选择解析模式和配置，返回解析器和模式理由。."""
    parser_mode = "parse_page_with_llm"
    disable_image_extraction = True
    save_images = False
    reason = ""

    # 边界平滑过渡
    if (
        config["thresholds"]["simple"] - 2
        <= score
        <= config["thresholds"]["simple"] + 2
    ):
        if is_visually_dense or is_continuous:
            score = config["thresholds"]["simple"] + 0.1
    if (
        config["thresholds"]["medium_high"] - 2
        <= score
        <= config["thresholds"]["medium_high"] + 2
    ):
        if is_visually_dense or is_continuous:
            score = config["thresholds"]["medium_high"] + 0.1

    if score > config["thresholds"]["medium_high"]:
        if is_continuous:
            parser_mode = "parse_document_with_llm"
            reason = f"Score {score:.2f} > {config['thresholds']['medium_high']}, continuous=True"
        else:
            parser_mode = (
                "parse_page_with_layout_agent"
                if is_visually_dense
                else "parse_page_with_agent"
            )
            disable_image_extraction = not is_visually_dense
            save_images = is_visually_dense
            reason = f"Score {score:.2f} > {config['thresholds']['medium_high']}, continuous=False, visually_dense={is_visually_dense}"
    elif score > config["thresholds"]["medium_low"]:
        parser_mode = (
            "parse_page_with_layout_agent"
            if is_visually_dense
            else "parse_page_with_agent"
        )
        disable_image_extraction = not is_visually_dense
        save_images = is_visually_dense
        reason = f"Score {score:.2f} in ({config['thresholds']['medium_low']}, {config['thresholds']['medium_high']}], visually_dense={is_visually_dense}"
    elif score > config["thresholds"]["simple"]:
        parser_mode = (
            "parse_page_with_layout_agent"
            if is_visually_dense
            else "parse_page_with_llm"
        )
        disable_image_extraction = not is_visually_dense
        save_images = is_visually_dense
        reason = f"Score {score:.2f} in ({config['thresholds']['simple']}, {config['thresholds']['medium_low']}], visually_dense={is_visually_dense}"
    else:
        parser_mode = "parse_page_with_llm"
        reason = f"Score {score:.2f} <= {config['thresholds']['simple']}"

    logger.debug(
        f"Selected parser mode: {parser_mode}, disable_image_extraction={disable_image_extraction}, save_images={save_images}, reason: {reason}"
    )
    return LlamaParse(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        language="ch_sim",
        parse_mode=parser_mode,
        preserve_layout_alignment_across_pages=True,
        spreadsheet_extract_sub_tables=True,
        result_type="markdown",
        disable_image_extraction=disable_image_extraction,
        save_images=save_images,
        user_prompt=USER_PROMPT,
        system_prompt_append=SYSTEM_PROMPT_APPEND,
        num_workers=4,
        max_timeout=7200,
        check_interval=10,
        verbose=True,
        do_not_cache=True,
    ), reason


def should_use_sync(file_paths: List[Path]) -> bool:
    """判断是否使用同步解析。."""
    return False


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
        logger.error(f"Sync parse failed for {file_path}: {type(e).__name__}: {e}")
        raise
    finally:
        logger.debug(f"Closing event loop for {file_path}: {loop}")
        loop.close()


async def parse_files(
    file_paths: List[Path], output_dir: Path, input_root: Path, use_sync: bool = False
) -> Tuple[List, Dict, Dict]:
    """解析文件列表，逐个处理，保留目录结构，动态调整阈值。."""
    all_docs = []
    file_docs_map = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "complexity_cache"
    report = {
        "files": [],
        "total_time": 0,
        "failures": [],
        "thresholds": config["thresholds"].copy(),
    }

    # 预创建输出目录结构
    for file_path in file_paths:
        relative_path = file_path.relative_to(input_root)
        (output_dir / relative_path.parent).mkdir(parents=True, exist_ok=True)

    # 自适应阈值
    scores = []
    for file_path in file_paths:
        score, _, _, _ = analyze_pdf_complexity(file_path, cache_dir)
        scores.append(score)
    if scores:
        median_score = sorted(scores)[len(scores) // 2]
        if median_score < 50:
            scale = median_score / 50
            report["thresholds"] = {
                "simple": config["thresholds"]["simple"] * scale,
                "medium_low": config["thresholds"]["medium_low"] * scale,
                "medium_high": config["thresholds"]["medium_high"] * scale,
            }
            logger.info(
                f"Adjusted thresholds based on median score {median_score:.2f}: {report['thresholds']}"
            )

    for file_path in file_paths:
        try:
            start_time = time.time()
            logger.debug(f"Parsing file: {file_path}")
            logger.debug(f"File size: {get_file_size_mb(file_path):.2f} MB")

            complexity_score, is_continuous, is_visually_dense, analysis_details = (
                analyze_pdf_complexity(file_path, cache_dir)
            )
            parser, parse_reason = get_parser_config(
                complexity_score, is_continuous, is_visually_dense
            )
            analysis_details["parse_reason"] = parse_reason

            try:
                reader = PdfReader(str(file_path))
                num_pages = len(reader.pages)
                logger.debug(f"File pages: {num_pages}")
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
                        await asyncio.sleep(2)
                    else:
                        logger.warning(
                            f"Async failed after 3 attempts for {file_path}: {e}"
                        )
                        report["failures"].append(
                            {
                                "file": str(file_path),
                                "error": f"Async parse failed: {e!s}",
                            }
                        )
                except Exception as e:
                    if (
                        isinstance(e, httpx.HTTPStatusError)
                        and e.response.status_code == 429
                    ):
                        logger.error(
                            f"API rate limit exceeded for {file_path}: {e}. Please check your LlamaCloud account credit limit or upgrade your plan."
                        )
                        report["failures"].append(
                            {
                                "file": str(file_path),
                                "error": "API rate limit exceeded: check credit limit",
                            }
                        )
                    else:
                        logger.warning(
                            f"Unexpected error during async parsing {file_path}: {e}"
                        )
                        report["failures"].append(
                            {
                                "file": str(file_path),
                                "error": f"Unexpected error: {e!s}",
                            }
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
                            check_interval=10,
                            verbose=True,
                            do_not_cache=True,
                        )
                        continue
                    break

            if parse_success:
                all_docs.extend(docs)
                file_docs_map[str(file_path)] = docs

            relative_path = file_path.relative_to(input_root)
            output_file = output_dir / relative_path.with_suffix(".md")

            if docs and any(hasattr(doc, "text") and doc.text.strip() for doc in docs):
                with open(output_file, "w", encoding="utf-8") as f:
                    for i, doc in enumerate(docs):
                        text = doc.text.strip() if hasattr(doc, "text") else ""
                        f.write(f"## Page {i + 1}\n\n{text}\n\n")
                logger.debug(f"Saved to {output_file}")
                with open(output_file, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                logger.debug(f"Output file hash: {file_hash}")
            else:
                logger.warning(f"No valid content for {file_path}, skipping output")
                report["failures"].append(
                    {"file": str(file_path), "error": "No valid content"}
                )

            report["files"].append(
                {
                    "file": str(file_path),
                    "complexity_score": complexity_score,
                    "complexity_details": analysis_details,
                    "parse_mode": parser.parse_mode,
                    "time": time.time() - start_time,
                    "success": parse_success,
                    "doc_count": len(docs),
                }
            )

            logger.info(f"File parsing completed: {file_path}")

        except RuntimeError as e:
            logger.warning(f"Event loop error for {file_path}: {e}")
            report["failures"].append(
                {"file": str(file_path), "error": f"Event loop error: {e!s}"}
            )
            continue
        except requests.exceptions.ConnectionError as e:
            logger.error(
                f"Network error: Failed to connect to {base_url}. Please check VPN: {e}"
            )
            report["failures"].append(
                {"file": str(file_path), "error": f"Network error: {e!s}"}
            )
            raise
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout error for {file_path}: Request timed out: {e}")
            report["failures"].append(
                {"file": str(file_path), "error": f"Timeout error: {e!s}"}
            )
            raise
        except Exception as e:
            logger.warning(f"Error processing {file_path}: {e}")
            report["failures"].append(
                {"file": str(file_path), "error": f"Unexpected error: {e!s}"}
            )
            continue

    return all_docs, report, file_docs_map


async def main():
    """主函数，解析PDF文件或文件夹。."""
    start_time = datetime.datetime.now()
    logger.info(f"Starting parsing at {start_time}")

    arg_parser = argparse.ArgumentParser(description="Parse PDF files using LlamaParse")
    arg_parser.add_argument("path", type=str, help="Path to PDF file or folder")
    arg_parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of parallel workers"
    )
    arg_parser.add_argument(
        "--max-timeout", type=int, default=7200, help="Maximum timeout in seconds"
    )
    args = arg_parser.parse_args()

    input_path = Path(args.path).resolve()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    if not input_path.exists():
        raise FileNotFoundError(f"Path {input_path} does not exist")

    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"File {input_path} is not a PDF file")

        logger.info(f"Processing file: {input_path}")
        output_dir = input_path.parent
        output_file = output_dir / f"{input_path.stem}_{timestamp}.md"

        try:
            cache_dir = output_dir / "complexity_cache"
            complexity_score, is_continuous, is_visually_dense, analysis_details = (
                analyze_pdf_complexity(input_path, cache_dir)
            )
            parser, parse_reason = get_parser_config(
                complexity_score, is_continuous, is_visually_dense
            )
            analysis_details["parse_reason"] = parse_reason

            try:
                reader = PdfReader(str(input_path))
                num_pages = len(reader.pages)
                logger.debug(f"File pages: {num_pages}")
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
                raise

            use_sync = should_use_sync([input_path])
            docs = []
            file_docs_map = {}
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
                            time.sleep(2)
                        else:
                            logger.warning(
                                f"Sync failed after 3 attempts for {input_path}: {e}"
                            )
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
                            await asyncio.sleep(2)
                        else:
                            logger.warning(
                                f"Async failed after 3 attempts for {input_path}: {e}"
                            )
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
                                check_interval=10,
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
            file_docs_map[str(input_path)] = docs

        except RuntimeError as e:
            logger.error(f"Event loop error for {input_path}: {e}")
            report["failures"].append(
                {"file": str(input_path), "error": f"Event loop error: {e!s}"}
            )
        except requests.exceptions.ConnectionError as e:
            logger.error(
                f"Network error: Failed to connect to {base_url}. Please check VPN: {e}"
            )
            report["failures"].append(
                {"file": str(input_path), "error": f"Network error: {e!s}"}
            )
            raise
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout error for {input_path}: Request timed out: {e}")
            report["failures"].append(
                {"file": str(input_path), "error": f"Timeout error: {e!s}"}
            )
            raise
        except Exception as e:
            logger.error(f"Error processing {input_path}: {e}")
            report["failures"].append(
                {"file": str(input_path), "error": f"Unexpected error: {e!s}"}
            )
            raise

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
        all_docs, report, file_docs_map = await parse_files(
            pdf_files, output_dir, input_root=input_path, use_sync=use_sync
        )

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

        table_data = []
        for file_report in report["files"]:
            file_path = file_report["file"]
            docs = file_docs_map.get(file_path, [])
            char_count = (
                sum(
                    len(doc.text.strip())
                    for doc in docs
                    if hasattr(doc, "text") and doc.text
                )
                if file_report["success"]
                else 0
            )
            table_data.append(
                [
                    Path(file_path).name,
                    file_report["doc_count"],
                    char_count,
                    f"{file_report['time']:.2f}",
                    str(file_report["complexity_details"].get("is_continuous", False)),
                    str(
                        file_report["complexity_details"].get(
                            "is_visually_dense", False
                        )
                    ),
                    file_report["parse_mode"],
                    file_report["complexity_details"].get("parse_reason", ""),
                ]
            )
        headers = [
            "文件名",
            "页面数",
            "字符数",
            "解析时长(s)",
            "连续性",
            "视觉密集",
            "解析模式",
            "模式选择理由",
        ]
        logger.info("解析汇总：")
        logger.info(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

    end_time = datetime.datetime.now()
    total_time = end_time - start_time
    report["total_time"] = total_time.total_seconds()
    logger.info(f"解析完成于 {end_time}")
    logger.info(f"总解析时间: {total_time.total_seconds():.2f} 秒")


if __name__ == "__main__":
    asyncio.run(main())
