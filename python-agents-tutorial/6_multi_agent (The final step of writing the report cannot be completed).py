from dotenv import load_dotenv

load_dotenv()

from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import (
    AgentOutput,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.agent.workflow import FunctionAgent
import os
from llama_index.tools.google import GoogleSearchToolSpec

# 加载 .env 文件中的环境变量
load_dotenv()

# 从环境变量中获取端点和密钥
api_endpoint = os.getenv("API_ENDPOINT")
api_key = os.getenv("API_KEY")
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

search_web = google_tool.to_tool_list()[0]


async def record_notes(ctx: Context, notes: str, notes_title: str) -> str:
    """Useful for recording notes on a given topic."""
    current_state = await ctx.get("state")
    if "research_notes" not in current_state:
        current_state["research_notes"] = {}
    current_state["research_notes"][notes_title] = notes
    await ctx.set("state", current_state)
    return "Notes recorded."


async def write_report(ctx: Context, report_content: str) -> str:
    """Useful for writing a report on a given topic."""
    current_state = await ctx.get("state")
    current_state["report_content"] = report_content
    await ctx.set("state", current_state)
    return "Report written."


async def review_report(ctx: Context, review: str) -> str:
    """Useful for reviewing a report and providing feedback."""
    current_state = await ctx.get("state")
    current_state["review"] = review
    await ctx.set("state", current_state)
    return "Report reviewed."


research_agent = FunctionAgent(
    name="ResearchAgent",
    description="Useful for searching the web for information on a given topic and recording notes on the topic.",
    system_prompt=(
        "You are the ResearchAgent that can search the web for information on a given topic and record notes on the topic. "
        "Once notes are recorded and you are satisfied, you should hand off control to the WriteAgent to write a report on the topic."
    ),
    llm=llm,
    tools=[search_web, record_notes],
    can_handoff_to=["WriteAgent"],
)

write_agent = FunctionAgent(
    name="WriteAgent",
    description="Useful for writing a report on a given topic.",
    system_prompt=(
        "You are the WriteAgent that can write a report on a given topic. "
        "Your report should be in a markdown format. The content should be grounded in the research notes. "
        "Once the report is written, you should get feedback at least once from the ReviewAgent."
    ),
    llm=llm,
    tools=[write_report],
    can_handoff_to=["ReviewAgent", "ResearchAgent"],
)

review_agent = FunctionAgent(
    name="ReviewAgent",
    description="Useful for reviewing a report and providing feedback.",
    system_prompt=(
        "You are the ReviewAgent that can review a report and provide feedback. "
        "Your feedback should either approve the current report or request changes for the WriteAgent to implement."
    ),
    llm=llm,
    tools=[review_report],
    can_handoff_to=["WriteAgent"],
)

agent_workflow = AgentWorkflow(
    agents=[research_agent, write_agent, review_agent],
    root_agent=research_agent.name,
    initial_state={
        "research_notes": {},
        "report_content": "Not written yet.",
        "review": "Review required.",
    },
)


async def main():
    handler = agent_workflow.run(
        user_msg="""
        Write me a report on the history of the web. Briefly describe the history
        of the world wide web, including the development of the internet and the
        development of the web, including 21st century developments.
    """
    )

    current_agent = None
    current_tool_calls = ""
    async for event in handler.stream_events():
        if (
            hasattr(event, "current_agent_name")
            and event.current_agent_name != current_agent
        ):
            current_agent = event.current_agent_name
            print(f"\n{'=' * 50}")
            print(f"🤖 Agent: {current_agent}")
            print(f"{'=' * 50}\n")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("📤 Output:", event.response.content)
            if event.tool_calls:
                print(
                    "🛠️  Planning to use tools:",
                    [call.tool_name for call in event.tool_calls],
                )
        elif isinstance(event, ToolCallResult):
            print(f"🔧 Tool Result ({event.tool_name}):")
            print(f"  Arguments: {event.tool_kwargs}")
            print(f"  Output: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"🔨 Calling Tool: {event.tool_name}")
            print(f"  With arguments: {event.tool_kwargs}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
