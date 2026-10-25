from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

import os
os.environ["LANGSMITH_PROJECT"]="langsmith_demo"

import os

print("Tracing:", os.getenv("LANGSMITH_TRACING"))
print("Project:", os.getenv("LANGSMITH_PROJECT"))
print("LangSmith key loaded:", bool(os.getenv("LANGSMITH_API_KEY")))
print("OpenAI key loaded:", bool(os.getenv("OPENAI_API_KEY")))

model = init_chat_model(
    model = "gpt-4o-mini",
    max_tokens = 1000,
)

prompt = ("You are a helpful assistant."
           "Please answer to the questions."
           "Say don't know incase of no information."
)

agent = create_agent(
    model = model,
    system_prompt = prompt,
    tools=[],
)

result = agent.invoke(
    {"messages":[{"role":"user","content":"What's the name of France's capital"}]}
)

print(result["messages"][-1].content)


