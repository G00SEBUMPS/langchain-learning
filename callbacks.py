from typing import Dict, Any, List
from langchain_classic.callbacks.base import BaseCallbackHandler
from langchain_classic.schema import LLMResult


class AgentCallbackHandler(BaseCallbackHandler):
    """Callback handler for agent execution."""

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        print(f"*** Prompt to LLM was :***\n{prompts[0]}\n********************")
        return super().on_llm_start(serialized, prompts, **kwargs)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        print(
            f"*** LLM Response was :***\n{response.generations[0][0].text}\n********************"
        )
        return super().on_llm_end(response, **kwargs)
