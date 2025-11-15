"""Simple Q/A Bot using LangChain."""

import argparse
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.prompts import PromptTemplate
from typing import Optional, List, Any
from langchain.llms.base import LLM
from pydantic.v1 import Field


class QwenLocalLLM(LLM):
    """LangChain LLM wrapper."""

    # Pydantic fields
    tokenizer: Any = Field(default=None, exclude=True)
    model: Any = Field(default=None, exclude=True)
    trust_remote_code: bool = False
    device: str = "cpu"

    class Config:
        # Allow HF model/tokenizer
        arbitrary_types_allowed = True

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B-Thinking-2507",
        device: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        # Initialize BaseModel
        super().__init__(**kwargs)

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.trust_remote_code = trust_remote_code

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=self.trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            trust_remote_code=self.trust_remote_code
        ).to(self.device)

    @property
    def _llm_type(self) -> str:
        return "qwen_local"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Generate concise answer and trim verbosity."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Only decode new generated tokens
        gen_only = output_ids[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(gen_only, skip_special_tokens=True).strip()

        if "Answer:" in text:
            text = text.split("Answer:", 1)[1].strip()

        for sep in ["\n\n", "\r\n\r\n"]:
            if sep in text:
                text = text.split(sep, 1)[0].strip()

        for end in [".", "!", "?"]:
            idx = text.find(end)
            if idx != -1:
                text = text[: idx + 1]
                break

        return text.strip()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Simple Q/A Bot using LangChain."
    )
    parser.add_argument(
        "-q",
        "--question",
        type=str,
        required=False,
        default="What is the currency of Japan?",
        help="Question to ask the model.",
    )
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        required=False,
        default="Qwen/Qwen3-4B-Thinking-2507",
        help="Hugging Face model id to load (e.g. Qwen/Qwen3-4B-Thinking-2507).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow execution of custom remote code.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    llm = QwenLocalLLM(model_name=args.model_name, trust_remote_code=args.trust_remote_code)

    prompt_template = """You are a helpful assistant.

Answer the user's question with one short sentence.

Question: {question}

Answer:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
    qa_chain = prompt | llm

    question = args.question
    answer = qa_chain.invoke({"question": question})

    print(f"Model:   {args.model_name}")
    print(f"Question: {question}")
    print(f"Answer:   {answer}")


if __name__ == "__main__":
    main()
