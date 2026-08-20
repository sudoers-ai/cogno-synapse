"""
cogno_synapse.openai_backend — OpenAI-compatible LLM backend.

OpenAI Chat Completions API (gpt-4o, gpt-4o-mini, o-series, …). Implements
``LLMBackend`` + ``ToolCallingBackend`` (native function calling).

Adapted from the parent Cogno backend: stdlib logging (no infra logger), no
tenant contextvar (the host owns key rotation), and — per cogno-anima's
errors-propagate contract — it **raises** on transport/auth failure instead of
returning ``("", 0, 0)`` (a ``FallbackBackend`` catches and tries the next).

Optional dependency: ``pip install "cogno-anima[openai]"`` (or ``openai``).
"""

from __future__ import annotations

import os
import time
import json
import logging

from cogno_synapse.errors import InvalidAPIKeyError
from cogno_synapse.tool_parsing import parse_tool_calls_from_text
from cogno_synapse._obs import log_done, log_request, warn_if_retryable

logger = logging.getLogger("cogno_synapse.openai")


def _warn_if_truncated(resp: object, model: str) -> bool:
    """Log when the PROVIDER says it cut the response. Returns whether it did.

    ``finish_reason`` is "length" when the answer hit the token ceiling and "content_filter"
    when it was cut for policy. Both arrive as an ordinary successful response with a shorter
    string — no exception, no type difference — so a truncation is only ever discovered later,
    by whatever chokes on the fragment, and reported as that thing instead.
    """
    try:
        reason = str(getattr(resp.choices[0], "finish_reason", "") or "")  # type: ignore[attr-defined]
    except (AttributeError, IndexError):
        return False
    if reason in ("length", "content_filter"):
        logger.warning("event=truncated_response provider=openai model=%s finish_reason=%s",
                       model, reason)
        return True
    return False


def _wants_json(system: str) -> bool:
    """O CHAMADOR pediu JSON — lendo só o system prompt, que é nosso.

    A primeira versão lia também o `prompt`, e ali mora texto do contato: o prompt da voz
    do SUPEREGO embute `# User request\n"{ctx.user_input}"` literalmente. Com isso um cliente
    que escrevesse "me manda em json" — ou só citasse `config.json` — flipava uma chamada de
    PROSA para `json_object` e recebia um objeto JSON como resposta. Entrada do usuário
    decidindo parâmetro de API é o defeito, não a redação.

    A regra da OpenAI (a palavra "json" tem de aparecer nas mensagens) continua satisfeita: os
    prompts de NOUMENO e NER a trazem no system, que é onde deve estar — quem pede o formato é
    o estágio, não quem conversa.
    """
    return "json" in (system or "").lower()


def _is_auth_error(exc: Exception) -> bool:
    if type(exc).__name__ in ("AuthenticationError", "PermissionDeniedError"):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    return status in (401, 403)


class OpenAIBackend:
    """Backend for OpenAI's Chat Completions API."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 4096,
        timeout: int = 120,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        # Point at any OpenAI-compatible endpoint (DeepSeek, Moonshot/Kimi, xAI,
        # OpenRouter, Together, Fireworks, …). None → OpenAI's default base URL.
        self.base_url = base_url
        # Learned once per instance: an affected model fails EVERY tool call, and the EGO
        # reuses one backend across 5–8 steps plus correction retries. Rediscovering the
        # conflict each time buys nothing and costs a full failed round-trip per step.
        self._tools_need_effort_none = False
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set — OpenAI calls will fail")

    @property
    def _is_o_series(self) -> bool:
        m = self.model.lower()
        return m.startswith(("o1", "o3", "o4", "gpt-5"))

    def _supports_json_mode(self) -> bool:
        """`response_format={"type":"json_object"}` só vale para a API da OpenAI.

        A classe serve TAMBÉM os compatíveis (DeepSeek, Moonshot, xAI, OpenRouter, Together,
        Fireworks) via `base_url`, e nem todos aceitam o parâmetro — mandar às cegas trocaria um
        JSON malformado ocasional por um 400 em todo turno, que é bem pior. Sem `base_url` =
        OpenAI de verdade.

        Fora também a série de raciocínio (o1/o3/o4/gpt-5): ela já tem tratamento próprio de
        parâmetros aqui (`max_completion_tokens` em vez de `max_tokens`, sem `temperature`), e
        mandar `response_format` a um modelo que o recusa troca um JSON malformado ocasional
        por um 400 em TODO turno — a mesma troca ruim que o guard de `base_url` evita.
        """
        return not self.base_url and not self._is_o_series

    def _token_limit_kwargs(self) -> dict:
        key = "max_completion_tokens" if self._is_o_series else "max_tokens"
        return {key: self.max_tokens}

    def _client(self):
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                'openai not installed. Run: pip install "cogno-anima[openai]"'
            ) from exc
        kwargs: dict = {"api_key": self.api_key, "timeout": self.timeout}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return openai.AsyncOpenAI(**kwargs)

    async def generate(self, system: str, prompt: str) -> tuple[str, int, int]:
        client = self._client()
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            **self._token_limit_kwargs(),
        }
        if self.temperature is not None and not self._is_o_series:
            kwargs["temperature"] = self.temperature
        # JSON MODE quando o prompt pede JSON. Sem isso o modelo devolve JSON "quase válido" e
        # o estágio derruba o turno do cliente: medido 2026-08-19, `StageParseError` no NOUMENO
        # em 12,5% dos cenários do closer_bench com gpt-4o-mini, intermitente, sempre no
        # `context_turn`, sempre por volta do caractere 211. O provedor devolveu `finish_reason`
        # normal — não era corte de stream nem teto de tokens, era JSON malformado mesmo, e o
        # retry de truncamento (que existe) gastava uma segunda chamada para falhar igual.
        #
        # O gatilho NÃO é heurística frouxa: a própria OpenAI RECUSA `json_object` se a palavra
        # "json" não aparecer nas mensagens, então a condição que checamos é a mesma que a API
        # impõe. Um prompt que não pede JSON não entra no modo, e nada muda para ele.
        if _wants_json(system) and self._supports_json_mode():
            kwargs["response_format"] = {"type": "json_object"}
        log_request(logger, "openai", self.model, system, prompt)
        try:
            t0 = time.perf_counter()
            resp = await client.chat.completions.create(**kwargs)
            usage = resp.usage
            tokens_in = usage.prompt_tokens if usage else 0
            tokens_out = usage.completion_tokens if usage else 0
            log_done(logger, "openai", self.model, t0, tokens_in, tokens_out)
            # A cut response is indistinguishable from a complete one at this layer — same
            # shape, same type, no exception — so it travels on and fails much later, where
            # the damage is diagnosed as something else. Measured 2026-08-04: a NOUMENO
            # payload arrived ending mid-string ("…about the volume of c"), raised
            # StageParseError, and killed the user's turn; the report said "bad JSON", which
            # is what the parser saw and not what happened. `finish_reason` is the provider
            # telling us plainly, and nothing read it.
            #
            # A WARNING, not an exception: the caller may still salvage a truncated answer
            # (a prose reply loses its tail, not its meaning), and raising here would turn a
            # degraded turn into a dead one. What it buys is a log line naming the cause, at
            # the only layer that can see it.
            _warn_if_truncated(resp, self.model)
            return (resp.choices[0].message.content or "", tokens_in, tokens_out)
        except Exception as exc:
            if _is_auth_error(exc):
                raise InvalidAPIKeyError(
                    f"OPENAI_API_KEY invalid/rejected (model={self.model}): {exc}"
                ) from exc
            warn_if_retryable(logger, "openai", self.model, exc)
            raise
        finally:
            await _safe_close(client)

    async def chat_with_tools(
        self, messages: list[dict], tools: list[dict], tool_choice=None,
    ) -> tuple[dict, int, int]:
        client = self._client()
        kwargs: dict = {"model": self.model, "messages": messages, **self._token_limit_kwargs()}
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None and tools:
            kwargs["tool_choice"] = tool_choice
        if self.temperature is not None and not self._is_o_series:
            kwargs["temperature"] = self.temperature
        if self._tools_need_effort_none and tools:
            kwargs["reasoning_effort"] = "none"
        try:
            try:
                resp = await client.chat.completions.create(**kwargs)
            except Exception as exc:
                # Some reasoning models refuse function tools while reasoning is on, and say so
                # precisely: "Function tools with reasoning_effort are not supported for
                # <model> in /v1/chat/completions. To use function tools, use /v1/responses or
                # set reasoning_effort to 'none'." Measured 2026-08-20 on gpt-5.6-luna/terra/sol
                # — every EGO turn on those models died with a 400 while gpt-5/-mini/-nano and
                # the 5.4 family were unaffected.
                #
                # RETRY on the provider's own instruction rather than hardcoding a model list:
                # the constraint belongs to the model, not to a name we can enumerate, and a
                # future model with the same rule works without another release. Applied ONLY
                # here, so ``generate`` (NOUMENO/NER/voice) keeps full reasoning — the trade is
                # made where tools are required, not everywhere.
                if not _is_reasoning_tools_conflict(exc):
                    raise
                logger.warning(
                    "stage=LLM event=reasoning_tools_conflict model=%s — retrying with "
                    "reasoning_effort='none' (the provider's stated remedy)", self.model)
                kwargs["reasoning_effort"] = "none"
                resp = await client.chat.completions.create(**kwargs)
                self._tools_need_effort_none = True      # don't pay the 400 again
            msg = resp.choices[0].message
            usage = resp.usage
            tokens_in = usage.prompt_tokens if usage else 0
            tokens_out = usage.completion_tokens if usage else 0
            _warn_if_truncated(resp, self.model)
            result: dict = {"content": msg.content or ""}
            if msg.tool_calls:
                result["tool_calls"] = [_openai_tool_call(tc) for tc in msg.tool_calls]
            elif result["content"]:
                rescued = parse_tool_calls_from_text(result["content"], tools)
                if rescued:
                    result["tool_calls"] = rescued
            return result, tokens_in, tokens_out
        except Exception as exc:
            if _is_auth_error(exc):
                raise InvalidAPIKeyError(
                    f"OPENAI_API_KEY invalid/rejected (model={self.model}): {exc}"
                ) from exc
            raise
        finally:
            await _safe_close(client)

    def supports_native_tools(self) -> bool:
        return True


def _is_reasoning_tools_conflict(exc: Exception) -> bool:
    """The provider refusing function tools because reasoning is enabled.

    Matched on the error's own ``param``/message rather than on a model-name list: the rule is
    the model's, and enumerating names means the next model with it fails in production until
    someone notices. Both signals must point at reasoning AND tools, so an unrelated 400 that
    happens to mention one of the words does not trigger a pointless retry.
    """
    low = str(exc).lower()
    # The TOOLS half is what separates "reasoning conflicts with tools" (retry helps) from
    # "this endpoint does not accept reasoning_effort at all" (retry re-sends the very
    # parameter just rejected — a guaranteed second 400, double latency, and the caller ends
    # up seeing the retry's error instead of the original). An OpenAI-COMPATIBLE endpoint
    # reached through ``base_url`` returns exactly that second shape, and the SDK sets
    # ``param='reasoning_effort'`` on BOTH — so the param alone cannot be the trigger.
    if "function tools" not in low:
        return False
    return ("reasoning_effort" in low
            or str(getattr(exc, "param", "") or "") == "reasoning_effort")


def _openai_tool_call(tc) -> dict:
    args = tc.function.arguments
    return {
        "id": tc.id,
        "type": "function",
        "function": {
            "name": tc.function.name,
            "arguments": args if isinstance(args, str) else json.dumps(args),
        },
    }


async def _safe_close(client) -> None:
    try:
        await client.close()
    except (TypeError, AttributeError):
        pass  # MagicMock in tests / no-op clients
