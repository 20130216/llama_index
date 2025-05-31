from dotenv import load_dotenv
import os
import asyncio
import requests
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import FunctionAgent, AgentStream
from llama_index.tools.google import GoogleSearchToolSpec

# 加载 .env 文件中的环境变量
load_dotenv()

# 从环境变量中获取端点和密钥
api_endpoint = os.getenv("BLENDAPI_API_ENDPOINT")
api_key = os.getenv("BLENDAPI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
google_cse_id = os.getenv("GOOGLE_CUSTOM_SEARCH_ENGINE_ID")

# 验证环境变量是否正确加载
if not all([api_endpoint, api_key, google_api_key, google_cse_id]):
    raise ValueError("Missing required environment variables in .env file")

# 简单验证 Google API 密钥和 CSE ID 格式
if len(google_api_key) < 20 or len(google_cse_id) < 10:
    raise ValueError("Invalid GOOGLE_API_KEY or GOOGLE_CUSTOM_SEARCH_ENGINE_ID format")

# 初始化 OpenAI 实例，指定自定义端点和密钥
llm = OpenAI(
    model="gpt-4o-mini",  # 使用指定模型
    api_base=api_endpoint,  # 使用自定义端点
    api_key=api_key,  # 使用自定义密钥
)

# 初始化 Google 搜索工具，指定 key、engine 和 num
google_tool = GoogleSearchToolSpec(google_api_key, google_cse_id, num=3)

# 初始化代理
workflow = FunctionAgent(
    tools=google_tool.to_tool_list(),
    llm=llm,
    system_prompt="You're a helpful assistant that can search the web for information using Google search.",
)


# 定义异步主函数
async def main():
    try:
        # 运行工作流并获取流式事件，使用更具体的查询
        # 当输入是“San Francisco weather今天的天气如何,中文输出”，会反馈“抱歉，我无法实时获取天气信息。不过，您可以通过访问天气网站或使用天气应用程序来查看旧金山今天的天气情况。通常，这些平台会提供温度、降水概率和风速等详细信息。%”
        handler = workflow.run(user_msg="HangZhou weather today,中文输出")
        # 在前面的示例中，我们在 workflow.run 方法上使用了 await 来获取来自代理的最终响应。但是，如果我们不等待响应，我们会返回一个异步迭代器，我们可以迭代该迭代器以在事件传入时获取事件。这个迭代器将返回各种事件。我们将从一个 AgentStream 事件开始，该事件包含输出的“增量”（最新更改）。
        # 运行工作流并查找要输出的该类型的事件  handler.stream_events():
        async for event in handler.stream_events():
            if isinstance(event, AgentStream) and event.delta:
                print(event.delta, end="", flush=True)
    except requests.exceptions.RequestException as e:
        print(f"Google API request failed: {e}")
    except ValueError as e:
        print(f"Invalid search parameters: {e}")
    except Exception as e:
        print(f"Error during workflow execution: {e}")


# 运行程序
if __name__ == "__main__":
    asyncio.run(main())
