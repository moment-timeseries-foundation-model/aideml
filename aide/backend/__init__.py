from collections import defaultdict

import logging

from . import backend_anthropic, backend_openai, backend_openrouter, backend_gdm
from .utils import FunctionSpec, OutputType, PromptType, compile_prompt_to_md

logger = logging.getLogger("aide")

# cost per input/output token for each model
# source https://platform.openai.com/docs/pricing
MODEL_COST = {
    "gpt-4o-2024-08-06": {"input": 2.5 / 1000000, "output": 10 / 1000000},
    "o3-mini-2025-01-31": {"input": 1.1 / 1000000, "output": 4.4 / 1000000},
    "o3-2025-04-16": {"input": 10 / 1000000, "output": 40 / 1000000},
    "o4-mini-2025-04-16": {"input": 1.1 / 1000000, "output": 4.4 / 1000000},
    "gpt-4.1-2025-04-14": {"input": 2 / 1000000, "output": 8 / 1000000},
    "gpt-4.1-mini-2025-04-14": {"input": 0.4 / 1000000, "output": 1.6 / 1000000},
    "claude-opus-4-20250514": {"input": 15 / 1000000, "output": 75 / 1000000},
    "claude-sonnet-4-20250514": {"input": 3 / 1000000, "output": 15 / 1000000},
    "claude-3-7-sonnet-20250219": {"input": 3 / 1000000, "output": 15 / 1000000},
    "claude-3-7-sonnet-20250219-think": {"input": 3 / 1000000, "output": 15 / 1000000},
    "claude-3-5-sonnet-20241022": {"input": 3 / 1000000, "output": 15 / 1000000},
    "claude-3-5-sonnet-20241022-think": {"input": 3 / 1000000, "output": 15 / 1000000},
    "gemini-2.5-flash-preview-05-20": {
        "input": 0.15 / 1000000,
        "output": 3.5 / 1000000,
    },
    "gemini-2.5-pro-preview-06-05": {"input": 1.25 / 1000000, "output": 10 / 1000000},
    "deepseek-reasoner": {"input": 0.55 / 1000000, "output": 2.19 / 1000000},
    "deepseek-chat": {"input": 0.27 / 1000000, "output": 1.1 / 1000000},
    "Llama-4-Maverick-17B-128E-Instruct-FP8": {
        "input": 0 / 1000000,
        "output": 0 / 1000000,
    },
    "Llama-3.3-8B-Instruct": {"input": 0 / 1000000, "output": 0 / 1000000},
    "Llama-3.3-70B-Instruct": {"input": 0 / 1000000, "output": 0 / 1000000},
}


def determine_provider(model: str) -> str:
    if (
        model.startswith("gpt-")
        or model.startswith("o1-")
        or model.startswith("o3-")
        or model.startswith("o4-")
        or model.startswith("deepseek-")
        or model.startswith("Llama-")
    ):
        return "openai"
    elif model.startswith("claude-"):
        return "anthropic"
    elif model.startswith("gemini-"):
        return "gdm"
    # all other models are handle by openrouter
    else:
        return "openrouter"


provider_to_query_func = {
    "openai": backend_openai.query,
    "anthropic": backend_anthropic.query,
    "gdm": backend_gdm.query,
    "openrouter": backend_openrouter.query,
    # "meta": backend_meta.query,
}


class TokenCounter:
    def __init__(self, cost_limit: int | None):
        self.cost_limit = cost_limit
        self.total_input_tokens = defaultdict(int)
        self.total_output_tokens = defaultdict(int)

    def cost(self) -> float:
        """
        compute to total cost of the tokens used
        """
        total_cost = 0

        # compute cost for input tokens
        for model_name, input_tokens in self.total_input_tokens.items():
            if model_name not in MODEL_COST:
                logger.warning(
                    f"Model {model_name} not supported for token counting, skipping"
                )
                return -1
            total_cost += input_tokens * MODEL_COST[model_name]["input"]

        # compute cost for output tokens
        for model_name, output_tokens in self.total_output_tokens.items():
            if model_name not in MODEL_COST:
                logger.warning(
                    f"Model {model_name} not supported for token counting, skipping"
                )
                return -1
            total_cost += output_tokens * MODEL_COST[model_name]["output"]
        return total_cost

    def add_tokens(self, model_name: str, input_tokens=None, output_tokens=None):
        """
        update the token counts
        """

        if input_tokens is not None:
            self.total_input_tokens[model_name] += input_tokens
        if output_tokens is not None:
            self.total_output_tokens[model_name] += output_tokens

    def remaining_output_tokens(self, model_name: str) -> int:
        """
        max_budget: the maximum dollar budget for the model
        compute the remaining tokens for a model
        """
        if model_name not in MODEL_COST:
            raise ValueError(f"Model {model_name} not supported for token counting")

        current_cost = self.cost()
        remaining_budget = self.cost_limit - current_cost
        if remaining_budget <= 0:
            return 0
        else:
            output_tokens_cost = MODEL_COST[model_name]["output"]
            return int(remaining_budget / output_tokens_cost)

    def exceed_budget_limit(self) -> bool:
        """
        check if the budget limit is exceeded
        """
        # if the cost limit is None, we don't check for budget limit
        if self.cost_limit is None:
            return False

        current_cost = self.cost()
        return current_cost >= self.cost_limit


def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    func_spec: FunctionSpec | None = None,
    convert_system_to_user: bool = False,
    token_counter: TokenCounter | None = None,
    **model_kwargs,
) -> OutputType:
    """
    General LLM query for various backends with a single system and user message.
    Supports function calling for some backends.

    Args:
        system_message (PromptType | None): Uncompiled system message (will generate a message following the OpenAI/Anthropic format)
        user_message (PromptType | None): Uncompiled user message (will generate a message following the OpenAI/Anthropic format)
        model (str): string identifier for the model to use (e.g. "gpt-4-turbo")
        temperature (float | None, optional): Temperature to sample at. Defaults to the model-specific default.
        max_tokens (int | None, optional): Maximum number of tokens to generate. Defaults to the model-specific max tokens.
        func_spec (FunctionSpec | None, optional): Optional FunctionSpec object defining a function call. If given, the return value will be a dict.

    Returns:
        OutputType: A string completion if func_spec is None, otherwise a dict with the function call details.
    """

    model_kwargs = model_kwargs | {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    logger.info("---Querying model---", extra={"verbose": True})
    system_message = compile_prompt_to_md(system_message) if system_message else None
    if system_message:
        logger.info(f"system: {system_message}", extra={"verbose": True})
    user_message = compile_prompt_to_md(user_message) if user_message else None
    if user_message:
        logger.info(f"user: {user_message}", extra={"verbose": True})
    if func_spec:
        logger.info(f"function spec: {func_spec.to_dict()}", extra={"verbose": True})

    provider = determine_provider(model)
    query_func = provider_to_query_func[provider]

    output, req_time, in_tok_count, out_tok_count, info = query_func(
        system_message=system_message,
        user_message=user_message,
        func_spec=func_spec,
        convert_system_to_user=convert_system_to_user,
        **model_kwargs,
    )
    logger.info(f"response: {output}", extra={"verbose": True})
    logger.info("---Query complete---", extra={"verbose": True})
    if token_counter is not None:
        token_counter.add_tokens(
            model, input_tokens=in_tok_count, output_tokens=out_tok_count
        )

    return output
