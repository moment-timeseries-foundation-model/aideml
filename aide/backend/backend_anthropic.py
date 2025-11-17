"""Backend for Anthropic API."""

import logging
import time

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
import anthropic

logger = logging.getLogger("aide")

_client: anthropic.Anthropic = None  # type: ignore

ANTHROPIC_TIMEOUT_EXCEPTIONS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
)

ANTHROPIC_MODEL_ALIASES = {
    "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3.7-sonnet": "claude-3-7-sonnet-20250219",
}


@once
def _setup_anthropic_client():
    global _client
    _client = anthropic.Anthropic(max_retries=0)


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    convert_system_to_user: bool = False,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """
    Query Anthropic's API, optionally with tool use (Anthropic's equivalent to function calling).
    """
    _setup_anthropic_client()

    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore
    if "max_tokens" not in filtered_kwargs:
        if "claude-3-5-sonnet" in filtered_kwargs["model"]:
            filtered_kwargs["max_tokens"] = 8192
        else:
            filtered_kwargs["max_tokens"] = 16000  # default for Claude models

    model_name = filtered_kwargs.get("model", "")
    logger.debug(f"Anthropic query called with model='{model_name}'")

    if model_name in ANTHROPIC_MODEL_ALIASES:
        model_name = ANTHROPIC_MODEL_ALIASES[model_name]
        filtered_kwargs["model"] = model_name
        logger.debug(f"Using aliased model name: {model_name}")

    if func_spec is not None and func_spec.name == "submit_review":
        filtered_kwargs["tools"] = [func_spec.as_anthropic_tool_dict]
        # Force tool use
        filtered_kwargs["tool_choice"] = func_spec.anthropic_tool_choice_dict

    # Anthropic doesn't allow not having user messages
    # if we only have system msg -> use it as user msg
    if system_message is not None and user_message is None:
        system_message, user_message = user_message, system_message

    # Anthropic passes system messages as a separate argument
    if system_message is not None:
        filtered_kwargs["system"] = system_message

    messages = opt_messages_to_list(None, user_message)
    think = False
    if (
        "claude-sonnet-4" in model_name or "claude-opus-4" in model_name
    ) and func_spec is not None:
        if model_name.endswith("-think"):
            think = True
            print("interleaved thinking enabled...")
            filtered_kwargs["extra_headers"] = {
                "anthropic-beta": "interleaved-thinking-2025-05-14"
            }

    if (
        "claude-sonnet-4" in model_name
        or "claude-opus-4" in model_name
        or "claude-3-7-sonnet" in model_name
    ) and func_spec is None:
        if model_name.endswith("-think"):
            think = True
            print("extended thinking enabled...")
            filtered_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10000}
            filtered_kwargs["temperature"] = 1
    if model_name.endswith("-think"):
        # remove trailing '-think' from model name
        filtered_kwargs["model"] = model_name[:-6]
    t0 = time.time()
    message = backoff_create(
        _client.messages.create,
        retry_exceptions=ANTHROPIC_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    # Handle tool calls if present
    if (
        func_spec is not None
        and "tools" in filtered_kwargs
        and len(message.content) > 0
        and message.content[0].type == "tool_use"
    ):
        block = message.content[0]  # This is a "ToolUseBlock"
        # block has attributes: type, id, name, input
        assert (
            block.name == func_spec.name
        ), f"Function name mismatch: expected {func_spec.name}, got {block.name}"
        output = block.input  # Anthropic calls the parameters "input"

    # handle thinking if enabled
    elif think:
        for block in message.content:
            if block.type == "thinking":
                continue  #! skip thinking blocks for now
            elif block.type == "text":
                output = block.text
    else:
        # For non-tool responses, ensure we have text content
        assert len(message.content) == 1, "Expected single content item"
        assert (
            message.content[0].type == "text"
        ), f"Expected text response, got {message.content[0].type}"
        output = message.content[0].text

    in_tokens = message.usage.input_tokens
    out_tokens = message.usage.output_tokens

    info = {
        "stop_reason": message.stop_reason,
        "model": message.model,
    }

    return output, req_time, in_tokens, out_tokens, info
