from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.tools import tool, render_text_description
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_classic.agents.format_scratchpad import format_log_to_str
from typing import Union
import os
from callbacks import AgentCallbackHandler

load_dotenv()


@tool
def count_characters(word: str) -> int:
    """Count the number of characters in a word."""
    return len(word.strip('"\n'))


tools = [count_characters]


def find_tool_by_names(tools, tool_name):
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found.")


def main():
    os.environ["LANGSMITH_PROJECT"] = "tool-def Project"
    print("Hello tool-def Project")
    prompt_template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}
    """

    llm = ChatOllama(
        model="gemma3:latest",
        temperature=0,
        num_ctx=128000,
        stop=["\nObservation:", "Observation:", "Observation"],
        callbacks=[AgentCallbackHandler()],
    )
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["input", "agent_scratchpad", "tool_names", "tools"],
    ).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )
    # Input is missing when we do chain .invoke we pass question as input
    # agent = prompt | llm
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    agent_step = ""
    intermediate_steps = []
    while not isinstance(agent_step, AgentFinish):
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "How many characters are there in the word 'encyclopedia'?",
                "agent_scratchpad": intermediate_steps,
            }
        )
        print("Agent Response:", agent_step)
        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_names(tools, tool_name)
            tool_input = agent_step.tool_input
            observation = tool_to_use.func(str(tool_input))
            intermediate_steps.append((agent_step, observation))
            print(f"Tool Observation: {observation}")
            print(f"Agent Step: {agent_step}")
    # agent_step:Union[AgentAction,AgentFinish] = agent.invoke({
    #     "input":"How many characters are there in the word 'encyclopedia'?",
    #     "agent_scratchpad": intermediate_steps
    # })
    print("Final Agent Response:", agent_step)
    if isinstance(agent_step, AgentFinish):
        print(f"Final Answer: {agent_step.return_values['output']}")


if __name__ == "__main__":
    main()
