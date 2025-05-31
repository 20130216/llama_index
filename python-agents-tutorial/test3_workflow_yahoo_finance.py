from dotenv import load_dotenv
import os
import asyncio
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec


# 定义基本工具
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product."""
    return a * b


def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum."""
    return a + b


# 加载 .env 文件中的环境变量
load_dotenv()

# 从环境变量中获取端点和密钥
api_endpoint = os.getenv("BLENDAPI_API_ENDPOINT")
api_key = os.getenv("BLENDAPI_API_KEY")

# 初始化 OpenAI 实例，指定自定义端点和密钥
llm = OpenAI(
    model="gpt-4.1",  # 与 curl 命令中一致的模型
    api_base=api_endpoint,  # 使用自定义端点
    api_key=api_key,  # 使用自定义密钥
)

# 初始化 Yahoo Finance 工具
finance_tools = YahooFinanceToolSpec().to_tool_list()

# 将自定义工具添加到工具列表;
# 为了展示如何将自定义工具与 LlamaHub 工具相结合，我们将保留 and 函数，即使我们在这里不需要它们
finance_tools.extend([multiply, add])

# 初始化代理
workflow = FunctionAgent(
    name="Agent",
    description="Useful for performing financial operations.",
    llm=llm,
    tools=finance_tools,
    # 原-备用1：system_prompt="You are a helpful assistant."
    # 优化：备用2:system_prompt = "你是一个金融助手，擅长使用 Yahoo Finance 工具查询股票信息。所有回答用中文，提供简洁准确的金融数据。"
    system_prompt="你是一个金融助手，使用 Yahoo Finance 工具查询股票信息。准确识别用户意图，选择最合适的工具（如股价用 stock_basic_info，新闻用 stock_news），并以中文回复。如果用户意图不明确，优先选择 stock_basic_info 或询问澄清。",
)


# 定义异步主函数
async def main():
    response = await workflow.run(user_msg="阿里巴巴目前的股价是多少?中文回复")
    print(response)


# 运行程序
if __name__ == "__main__":
    asyncio.run(main())
