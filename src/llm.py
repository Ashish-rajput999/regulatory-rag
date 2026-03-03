import os
import litellm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from litellm.exceptions import RateLimitError, APIConnectionError

@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    reraise=True,
)
def call_groq(
    prompt: str,
    model: str,
    system: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 2000,
    json_mode: bool = False,
) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "api_key": os.environ["GROQ_API_KEY"],
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = litellm.completion(**kwargs)
    return resp.choices[0].message.content
