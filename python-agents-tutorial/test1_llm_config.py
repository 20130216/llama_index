from dotenv import load_dotenv
import os
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import FunctionAgent

load_dotenv()

api_endpoint = os.getenv("BLENDAPI_API_ENDPOINT")
api_key = os.getenv("BLENDAPI_API_KEY")

llm = OpenAI(model="gpt-4o-mini", api_base=api_endpoint, api_key=api_key)

agent = FunctionAgent(llm=llm)

response = llm.complete("你是什么模型？数据截止到何时？")
print(response)
