"""Backend for GDM Gemini API"""

import time
import logging
import os

from google.api_core import exceptions
from google import genai
from google.genai.types import (
    HarmCategory,
    HarmBlockThreshold,
    SafetySetting,
    GenerateContentConfig,
    Tool,
)

# from google.generativeai.generative_models import generation_types

from funcy import notnone, once, select_values
from .utils import FunctionSpec, OutputType, backoff_create

logger = logging.getLogger("aide")

gdm_model = None  # type: ignore
generation_config = None  # type: ignore

GDM_TIMEOUT_EXCEPTIONS = (
    exceptions.RetryError,
    exceptions.TooManyRequests,
    exceptions.ResourceExhausted,
    exceptions.InternalServerError,
)
safety_settings = [
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
]


@once
def _setup_gdm_client():
    global _client
    global generation_config
    _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    convert_system_to_user: bool = False,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    _setup_gdm_client()
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore
    if "max_tokens" not in filtered_kwargs:
        filtered_kwargs["max_tokens"] = 65536  # default for Claude models
    model = filtered_kwargs.get("model", "")
    temperature = filtered_kwargs.get("temperature", None)
    max_output_tokens = filtered_kwargs.get("max_output_tokens", None)

    if func_spec is not None:
        tools = [Tool(function_declarations=[func_spec.as_gdm_tool_dict])]
        generation_config = GenerateContentConfig(
            tools=tools,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            safety_settings=safety_settings,
        )
    else:
        generation_config = GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            safety_settings=safety_settings,
        )
    # GDM gemini api doesnt support system messages outside of the beta
    parts = []
    if system_message:
        parts.append({"text": system_message})
    if user_message:
        parts.append({"text": user_message})
    messages = [{"role": "user", "parts": parts}] if parts else []

    t0 = time.time()
    response = backoff_create(
        _client.models.generate_content,
        retry_exceptions=GDM_TIMEOUT_EXCEPTIONS,
        model=model,
        contents=messages,
        config=generation_config,
    )
    req_time = time.time() - t0
    # Check if the model responded with a function call
    # iterate over parts of the response
    function_call = None
    for part in response.candidates[0].content.parts:
        if part.function_call:
            function_call = part.function_call
            break
    if func_spec is not None:
        assert function_call is not None, "Function call not found in response"
        func_name = function_call.name
        assert (
            func_name == func_spec.name
        ), f"Function name mismatch: expected {func_spec.name}, got {func_name}"
        func_args = {key: value for key, value in function_call.args.items()}
        func_args["function_name"] = func_name
        output = func_args
    else:
        # if response.prompt_feedback and response.prompt_feedback.block_reason:
        #     output = response.prompt_feedback
        #     print(output)
        # else:
        #     # Otherwise, return the text content
        output = response.text
    in_tokens = response.usage_metadata.prompt_token_count
    out_tokens = response.usage_metadata.candidates_token_count
    info = {}  # this isnt used anywhere, but is an expected return value

    # only `output` is actually used by scaffolding
    return output, req_time, in_tokens, out_tokens, info
