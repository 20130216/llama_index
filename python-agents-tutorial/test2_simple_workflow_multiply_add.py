from dotenv import load_dotenv
import os
import asyncio
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import FunctionAgent


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
    model="gpt-4o-mini",  # 与 curl 命令中一致的模型
    api_base=api_endpoint,  # 使用自定义端点
    api_key=api_key,  # 使用自定义密钥
)

# 初始化代理
workflow = FunctionAgent(
    tools=[multiply, add],
    llm=llm,
    system_prompt="You are an agent that can perform basic mathematical operations using tools.",
)


# 定义异步主函数
async def main():
    response = await workflow.run(user_msg="What is 20+(2*4)?")
    print(response)


# 运行程序
if __name__ == "__main__":
    asyncio.run(main())
