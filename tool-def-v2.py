from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from typing import Any
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
    print("Hello tool-def Project (bind_tools + Messages)")
    # Use an Ollama model that supports tools (e.g., llama3.1). gemma3 does not support tools.
    llm = ChatOllama(
        model="llama3.1:latest",
        temperature=0,
        num_ctx=4096,
        callbacks=[AgentCallbackHandler()],
    )

    # Bind tools directly to the LLM
    llm_with_tools = llm.bind_tools(tools)

    # Start the conversation using LangChain Messages
    question = "How many characters are there in the word 'encyclopedia'?"
    messages: list[Any] = [HumanMessage(content=question)]

    max_loops = 5
    for _ in range(max_loops):
        try:
            ai_msg: AIMessage = llm_with_tools.invoke(messages)
        except Exception as e:
            print("Model/tool invocation error:", e)
            print(
                "Hint: Ensure your Ollama model supports tools (e.g., 'llama3.1:latest')."
            )
            return
        print("AI Message:", ai_msg)

        # If the model requests tool calls, execute them and append ToolMessages
        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        if tool_calls:
            for tc in tool_calls:
                # Handle both dict-style and object-style tool calls
                name = getattr(tc, "name", None) or (tc.get("name") if isinstance(tc, dict) else None)
                args = getattr(tc, "args", None) or (tc.get("args") if isinstance(tc, dict) else None)
                call_id = getattr(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else None)

                if not name:
                    continue

                try:
                    selected_tool = find_tool_by_names(tools, name)
                except Exception as e:
                    tool_result = f"Tool resolution error for '{name}': {e}"
                    # Try to attach a ToolMessage so the model can recover
                    try:
                        messages.append(ToolMessage(content=tool_result, tool_call_id=call_id or name))
                    except TypeError:
                        messages.append(ToolMessage(content=tool_result, tool_call_id=call_id or "unknown"))
                    continue

                try:
                    # Prefer dict args when available; fall back to passing raw args
                    if isinstance(args, dict):
                        tool_result = selected_tool.invoke(args)
                    else:
                        tool_result = selected_tool.invoke(args)
                except Exception as e:
                    tool_result = f"Tool execution error for '{name}': {e}"

                # Append tool result as ToolMessage for the model to consume
                try:
                    messages.append(ToolMessage(content=str(tool_result), tool_call_id=call_id or name))
                except TypeError:
                    # Older versions may have different signature; keep minimal
                    messages.append(ToolMessage(content=str(tool_result), tool_call_id=call_id or "unknown"))

            # Continue loop for the model to incorporate tool results
            continue

        # No tool calls -> final answer
        if getattr(ai_msg, "content", None):
            print(f"Final Answer: {ai_msg.content}")
        else:
            print(f"Final Answer: {ai_msg}")
        break
    else:
        print("Stopped after max tool loops without a final answer.")


if __name__ == "__main__":
    main()
