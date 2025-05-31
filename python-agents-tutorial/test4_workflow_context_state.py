from dotenv import load_dotenv
import os
import asyncio
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context

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
    name="Agent",
    description="Useful for answering general questions and maintaining conversation context.",
    llm=llm,
    system_prompt="You are a helpful assistant capable of remembering user information across interactions.",
)

# 创建 Context 对象以维护状态
ctx = Context(workflow)


# 定义异步主函数
async def main():
    # 第一次运行：提供用户信息
    # 在 workflow.run 方法上使用了 await 来获取来自代理的最终响应
    response = await workflow.run(user_msg="Hi, 我的命令叫张三!", ctx=ctx)
    print(response)

    # 第二次运行：询问用户姓名，检查是否记住
    response2 = await workflow.run(user_msg="你的名字叫啥?", ctx=ctx)
    print(response2)


# 运行程序
if __name__ == "__main__":
    asyncio.run(main())
