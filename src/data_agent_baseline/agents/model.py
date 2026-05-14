from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from openai import APIError, OpenAI


@dataclass(frozen=True, slots=True)
class ModelMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ModelStep:
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str
    plan: str = ""  # Plan-and-Solve: 다음 단계들의 계획 (선택사항)


class ModelAdapter(Protocol):
    def complete(self, messages: list[ModelMessage]) -> str:
        raise NotImplementedError


class OpenAIModelAdapter:
    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float,
        request_timeout_seconds: float = 60.0,
        max_retries: int = 1,
        provider_only: tuple[str, ...] = (),
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.request_timeout_seconds = request_timeout_seconds
        self.max_retries = max_retries
        self.provider_only = tuple(provider_only)
        self.seed = seed

    def complete(self, messages: list[ModelMessage] | list[dict[str, str]]) -> str:
        if not self.api_key:
            raise RuntimeError(
                "Missing model API key. Set MODEL_API_KEY env var (preferred) or agent.api_key in config."
            )
        if not self.api_base:
            raise RuntimeError(
                "Missing model API base URL. Set MODEL_API_URL env var (preferred) or agent.api_base in config."
            )

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=self.request_timeout_seconds,
            max_retries=self.max_retries,
        )

        # ModelMessage 또는 dict 형식 모두 수용
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                formatted_messages.append(msg)
            else:
                formatted_messages.append({"role": msg.role, "content": msg.content})

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": self.temperature,
        }
        if self.seed is not None:
            request_kwargs["seed"] = self.seed
        if self.provider_only:
            # OpenRouter-specific routing knob — pinned providers only, no fallback,
            # so dev-side variance from cross-provider routing is removed. Other
            # OpenAI-compatible servers ignore unknown extra_body fields.
            request_kwargs["extra_body"] = {
                "provider": {
                    "only": list(self.provider_only),
                    "allow_fallbacks": False,
                }
            }

        try:
            response = client.chat.completions.create(**request_kwargs)
        except APIError as exc:
            raise RuntimeError(f"Model request failed: {exc}") from exc
        """
        choices = response.choices or []
        if not choices:
            raise RuntimeError("Model response missing choices.")
        content = choices[0].message.content
        if not isinstance(content, str):
            raise RuntimeError("Model response missing text content.")
        """
        choices = response.choices or []
        if not choices:
            raise RuntimeError("Model response missing choices.")
        
        #content = choices[0].message.content
        #if not isinstance(content, str):
        #    raise RuntimeError("Model response missing text content.")

        message = choices[0].message
        content = message.content
        
        if not isinstance(content, str) or not content.strip():
            reasoning = getattr(message, "reasoning_content", None)
            if isinstance(reasoning, str) and reasoning.strip():
                content = reasoning
            else:
                raw = getattr(message, "content", "") or ""
                import re as _re
                think_match = _re.search(r"<think>(.*?)</think>", raw, _re.DOTALL)
                if think_match:
                    content = think_match.group(1).strip()
 
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Model response missing text content.")

        return content


class ModelAdapterLLMWrapper:
    """
    ACON의 HistoryOptimizer는 llm.generate(prompt, temperature=...) 호출을 기대함
    에이전트의 ModelAdapter는 complete(messages) 메서드를 사용함
    이 클래스는 둘을 연결해서 ACON 프레임워크가 에이전트의 모델 사용할 수 있게 함.
    """
    
    def __init__(self, model_adapter: ModelAdapter):
        self.adapter = model_adapter
    
    def generate(self, prompt: str, temperature: float = None) -> str:
        """ACON의 HistoryOptimizer에서 호출하는 메서드
        
        None이나 비정상 응답 시 exception을 raise하여 ACON이 압축을 스킵하도록 유도
        """
        # ACON의 프롬프트는 사용자 입력이니까 user role로 전달
        messages = [{"role": "user", "content": prompt}]
        
        # temperature 파라미터 전달
        # OpenAIModelAdapter는 complete()에서 temperature를 지원하지 않기 때문에 adapter.temperature 대신 사용
        response = self.adapter.complete(messages)
        
        # None이나 비정상 응답 시 exception을 raise, exception을 catch하고 압축 스킵
        if response is None:
            raise RuntimeError("[ACON LLM] API returned None response - likely API error, rate limit, or service disruption")
        
        if not isinstance(response, str):
            raise RuntimeError(f"[ACON LLM] API returned non-string type {type(response).__name__} instead of str")
        
        return response



class ScriptedModelAdapter:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def complete(self, messages: list[ModelMessage]) -> str:
        del messages
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        return self._responses.pop(0)
