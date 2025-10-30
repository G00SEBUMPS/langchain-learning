from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain_classic.agents import create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import AgentExecutor


load_dotenv()

import os

os.environ["LANGSMITH_PROJECT"] = "Function Calling Project"
print("Hello Function Calling Project")
@tool
def multiply(x:float, y:float) -> float:
    """Multiply two numbers."""
    return x * y

def main():
    prompt = ChatPromptTemplate.from_messages(
        [
           ("system", "You are a helpful assistant that can use tools."),
           ("human", "{input}"),
           ("placeholder", "{agent_scratchpad}"),
        ]
    )
    tools = [multiply, TavilySearch()]
    llm = ChatOllama(model="llama3.1:latest", temperature=0, num_ctx=8000)
    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    res  =  agent_executor.invoke(
        {
            "input": "Comapre the temperature of Dubai and San Francisco. Give output in celsius.",
        }
    )
    print(res)

if __name__ == "__main__":
    main()