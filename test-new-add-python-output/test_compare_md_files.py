import difflib
import os

# 读取两个 Markdown 文件
file1_path = "parsed_output_split_by_page_true.md"
file2_path = "test-new-add-python-output/parsed_output_split_by_page_true.md"

if not os.path.exists(file1_path) or not os.path.exists(file2_path):
    print(
        "ERROR--One or both files do not exist. Please ensure both Markdown files are generated."
    )
    exit(1)

with (
    open(file1_path, "r", encoding="utf-8") as f1,
    open(file2_path, "r", encoding="utf-8") as f2,
):
    text1 = f1.readlines()
    text2 = f2.readlines()


# 基本统计信息
def get_stats(text_lines):
    char_count = sum(len(line) for line in text_lines)
    line_count = len(text_lines)
    heading_count = sum(1 for line in text_lines if line.startswith("##"))
    empty_line_count = sum(1 for line in text_lines if line.strip() == "")
    return char_count, line_count, heading_count, empty_line_count


char_count1, line_count1, heading_count1, empty_line_count1 = get_stats(text1)
char_count2, line_count2, heading_count2, empty_line_count2 = get_stats(text2)

print("=== 基本统计信息 ===")
print(f"文件 1 ({file1_path}):")
print(f"  - 字符数: {char_count1}")
print(f"  - 行数: {line_count1}")
print(f"  - 标题数: {heading_count1}")
print(f"  - 空行数: {empty_line_count1}")
print(f"文件 2 ({file2_path}):")
print(f"  - 字符数: {char_count2}")
print(f"  - 行数: {line_count2}")
print(f"  - 标题数: {heading_count2}")
print(f"  - 空行数: {empty_line_count2}")

# 差异对比
differ = difflib.Differ()
diff = list(differ.compare(text1, text2))

print("\n=== 差异报告 ===")
in_diff_section = False
for line in diff:
    if line.startswith("  "):  # 相同行
        if in_diff_section:
            print("  [无进一步差异]")
            in_diff_section = False
    elif line.startswith(("- ", "+ ")):  # 差异行
        if not in_diff_section:
            print("  [发现差异]:")
            in_diff_section = True
        print(f"  {line}", end="")
    else:
        continue

# 汇总差异
removed_lines = [line[2:] for line in diff if line.startswith("- ")]
added_lines = [line[2:] for line in diff if line.startswith("+ ")]
common_lines = [line[2:] for line in diff if line.startswith("  ")]

print("\n=== 差异汇总 ===")
print(f"  - 移除行数: {len(removed_lines)}")
print(f"  - 添加行数: {len(added_lines)}")
print(f"  - 共同行数: {len(common_lines)}")

# 评价（中文输出）
print("\n=== 评价 ===")
evaluation_lines = []
if (
    char_count1 == char_count2
    and line_count1 == line_count2
    and heading_count1 == heading_count2
):
    evaluation_lines.append(
        "评价：两个文件在字符数、行数和标题数方面完全相同。未检测到显著结构差异。"
    )
elif (
    abs(char_count1 - char_count2) / char_count1 < 0.01
    and abs(line_count1 - line_count2) / line_count1 < 0.01
):
    evaluation_lines.append(
        "评价：两个文件在内容和结构上高度相似，差异小于1%（字符数和行数变化）。差异可能源自格式调整或轻微文本修改。"
    )
else:
    evaluation_lines.append(
        "评价：两个文件在内容或结构上显示显著差异。这可能由于解析逻辑变化、缺失部分或添加内容引起。建议详细审查差异。"
    )
    if heading_count1 != heading_count2:
        evaluation_lines.append(
            f"  - 注意：标题数不同 ({heading_count1} vs {heading_count2})，表明可能存在结构变化。"
        )

for line in evaluation_lines:
    print(line)

# 可视化差异（可选，需安装 rich 库）
try:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="详细差异汇总")
    table.add_column("指标", justify="right")
    table.add_column("文件 1", justify="center")
    table.add_column("文件 2", justify="center")

    table.add_row("字符数", str(char_count1), str(char_count2))
    table.add_row("行数", str(line_count1), str(line_count2))
    table.add_row("标题数", str(heading_count1), str(heading_count2))
    table.add_row("空行数", str(empty_line_count1), str(empty_line_count2))
    table.add_row("移除行数", str(len(removed_lines)), "-")
    table.add_row("添加行数", "-", str(len(added_lines)))
    table.add_row("共同行数", str(len(common_lines)), str(len(common_lines)))

    console.print(table)
except ImportError:
    print("注意--请安装 'rich' 库以获得增强的视觉输出。")
    print("如果使用 hatch 管理环境，可运行：hatch run pip install rich")


# 创建文件夹（如果不存在）
output_dir = "test-new-add-python-output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# 保存解析结果为 Markdown 文件
# with open("parsed_output_split_by_page_true.md", "w", encoding="utf-8") as f:
# 保存差异报告到文件
with open(
    os.path.join(output_dir, "test_compare_diff_report.txt"), "w", encoding="utf-8"
) as f:
    f.write("=== 基本统计信息 ===\n")
    f.write(f"文件 1 ({file1_path}):\n")
    f.write(f"  - 字符数: {char_count1}\n")
    f.write(f"  - 行数: {line_count1}\n")
    f.write(f"  - 标题数: {heading_count1}\n")
    f.write(f"  - 空行数: {empty_line_count1}\n")
    f.write(f"文件 2 ({file2_path}):\n")
    f.write(f"  - 字符数: {char_count2}\n")
    f.write(f"  - 行数: {line_count2}\n")
    f.write(f"  - 标题数: {heading_count2}\n")
    f.write(f"  - 空行数: {empty_line_count2}\n")
    f.write("\n=== 差异报告 ===\n")
    f.write("\n".join(diff))
    f.write("\n=== 差异汇总 ===\n")
    f.write(f"  - 移除行数: {len(removed_lines)}\n")
    f.write(f"  - 添加行数: {len(added_lines)}\n")
    f.write(f"  - 共同行数: {len(common_lines)}\n")
    f.write("\n=== 评价 ===\n")
    for line in evaluation_lines:
        f.write(f"{line}\n")
print("DEBUG--已保存差异报告到 test_compare_diff_report.txt")
