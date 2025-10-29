from dotenv import load_dotenv
import os
from langchain_classic import hub
# runtime of the agent that will actually execute the agent's logic
from langchain_classic.agents import AgentExecutor
# special lang chain function to create REACT agent is a runnable langchain chain (referred as react agent)
from langchain_classic.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
# Its a tavily tool a lang chain tool that will give our llm capability to search web
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
@tool
def add(a:int , b:int) -> int:
    """Add two numbers together."""
    return a + b

load_dotenv()
tools = [TavilySearch()]
# llm  = ChatOpenAI(model="gpt-4")
llm = ChatOllama(model="gemma3:latest",temperature=0,num_ctx=128000)
react_prompt = hub.pull("hwchase17/react")
agent  = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt
)
agent_executor = AgentExecutor(agent=agent, tools=tools,verbose=True)
chain = agent_executor
def main():
    os.environ["LANGSMITH_PROJECT"] = "Search Agent Project"
    print("Hello Search Agent")
    result = chain.invoke(
        input={
            "input":"search for 3 best gaming keyboards in India that needs to be mechanical include result form meckeys also get therie rpices and compare it on amazon also."
        }
    )

if __name__ == "__main__":
    main()