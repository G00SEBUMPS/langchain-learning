import os

from dotenv import load_dotenv


# special lang chain function to create REACT agent is a runnable langchain chain (referred as react agent)

from langchain_core.tools import tool
from langchain_ollama import ChatOllama

# Its a tavily tool a lang chain tool that will give our llm capability to search web
from langchain_tavily import TavilySearch

from schemas import AgentResponse
from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS

from langchain.agents import create_agent


@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


load_dotenv()
tools = [TavilySearch()]
# llm  = ChatOpenAI(model="gpt-4")
# Using llama3.1 or llama3.2 which support tool calling (gemma3 does not support tools)
llm = ChatOllama(model="llama3.1:latest", temperature=0, num_ctx=8000)
# react_prompt = hub.pull("hwchase17/react")
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    os.environ["LANGSMITH_PROJECT"] = "Search Agent Project"
    print("Hello Search Agent")
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "search for 3 best gaming keyboards in India that needs to be mechanical.",
                }
            ]
        }
    )

    # Extract the final AI message from the messages list
    messages = result["messages"]
    final_message = messages[-1]  # Get the last message (AI's final response)

    print("\n" + "=" * 80)
    print("AGENT RESPONSE:")
    print("=" * 80)
    print(final_message.content)
    print("=" * 80 + "\n")

    # If you want the structured response (AgentResponse format), it should be parsed
    # from the final_message content if the response_format was properly applied
    # For now, we're just displaying the raw text response


if __name__ == "__main__":
    main()
