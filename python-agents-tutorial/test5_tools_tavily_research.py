from dotenv import load_dotenv
import os
import asyncio
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import FunctionAgent, AgentStream
from llama_index.tools.tavily_research import TavilyToolSpec

# 加载 .env 文件中的环境变量
load_dotenv()

# 从环境变量中获取端点和密钥
api_endpoint = os.getenv("BLENDAPI_API_ENDPOINT")
api_key = os.getenv("BLENDAPI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# 初始化 OpenAI 实例，指定自定义端点和密钥
llm = OpenAI(
    model="gpt-4o-mini",  # 与 curl 命令中一致的模型
    api_base=api_endpoint,  # 使用自定义端点
    api_key=api_key,  # 使用自定义密钥
)

# 初始化 Tavily 搜索工具
tavily_tool = TavilyToolSpec(api_key=tavily_api_key)

# 初始化代理
workflow = FunctionAgent(
    tools=tavily_tool.to_tool_list(),
    llm=llm,
    system_prompt="You're a helpful assistant that can search the web for information.",
)


# 定义异步主函数
async def main():
    # 运行工作流并获取流式事件
    handler = workflow.run(user_msg="What's the weather like in San Francisco?")
    async for event in handler.stream_events():
        if isinstance(event, AgentStream):
            print(event.delta, end="", flush=True)


# 运行程序
if __name__ == "__main__":
    asyncio.run(main())
