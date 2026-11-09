"""OpenAI-compatible structured output with a client-side required-tool gate."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from jsonschema import validate as validate_json_schema
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class RequiredToolError(ValueError):
    """No valid submission was obtained within the formatting-repair budget."""


class CompatibleChatOpenAI(ChatOpenAI):
    structured_output_mode: Literal["native", "prompt"] = "native"
    structured_max_retries: int = Field(default=2, ge=0, le=5)
    schema_call_log: list[dict[str, Any]] = Field(default_factory=list, exclude=True, repr=False)
    request_log: list[dict[str, Any]] = Field(default_factory=list, exclude=True, repr=False)
    payload_log: list[dict[str, Any]] = Field(default_factory=list, exclude=True, repr=False)
    usage_log_path: str | None = Field(default=None, exclude=True, repr=False)
    runtime_budget_path: str | None = Field(default=None, exclude=True, repr=False)
    request_scene_dir: str | None = Field(default=None, exclude=True, repr=False)
    request_scene_role: str = Field(default="model", exclude=True, repr=False)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from popper.runtime_budget import RuntimeBudget, journal
        budget = RuntimeBudget(Path(self.runtime_budget_path)) if self.runtime_budget_path else None
        events = (budget.events if budget else
                  Path(self.usage_log_path).with_suffix('.timing.jsonl') if self.usage_log_path else None)
        started = time.monotonic()
        if budget:
            request_id, timeout = budget.begin_request()
            kwargs['timeout'] = timeout
        else:
            request_id = uuid4().hex
            if events:
                journal(events, 'request_start', request_id=request_id, model=self.model_name)
        try:
            from popper.request_scene import recording_scope
            with recording_scope(self, request_id):
                result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except BaseException as exc:
            if events:
                journal(events, 'request_error', request_id=request_id,
                        duration_seconds=time.monotonic()-started, error_type=type(exc).__name__)
            raise
        if events:
            journal(events, 'request_end', request_id=request_id,
                    duration_seconds=time.monotonic()-started, model=self.model_name,
                    role=Path(self.usage_log_path).stem if self.usage_log_path else None)
        return result

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        from popper.request_scene import recording_scope
        with recording_scope(self):
            return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _stream(self, *args, **kwargs):
        from popper.request_scene import recording_scope
        with recording_scope(self):
            yield from super()._stream(*args, **kwargs)

    async def _astream(self, *args, **kwargs):
        from popper.request_scene import recording_scope
        with recording_scope(self):
            async for chunk in super()._astream(*args, **kwargs):
                yield chunk

    def _create_chat_result(self, response: Any, generation_info: dict | None = None) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        data = response if isinstance(response, dict) else response.model_dump()
        for generation, choice in zip(result.generations, data.get("choices", [])):
            message = choice.get("message", {})
            if "reasoning_content" in message:
                generation.message.additional_kwargs["reasoning_content"] = message["reasoning_content"]
        # This usage journal stores counts only. Opt-in request scenes are
        # captured separately at the HTTP boundary.
        record = {"usage": data.get("usage"), "model": data.get("model", self.model_name),
                  "received_at_utc": datetime.now(timezone.utc).isoformat()}
        self.request_log.append(record)
        if self.usage_log_path is not None:
            path = Path(self.usage_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Append completed-response usage immediately, so process timeouts
            # do not erase earlier usage. No prompts, reasoning, or keys.
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
        return result

    def _get_request_payload(self, input_: LanguageModelInput, *, stop: list[str] | None = None,
                             **kwargs: Any) -> dict:
        messages = self._convert_input(input_).to_messages()
        payload = super()._get_request_payload(messages, stop=stop, **kwargs)
        for message, outgoing in zip(messages, payload["messages"]):
            if isinstance(message, AIMessage) and "reasoning_content" in message.additional_kwargs:
                outgoing["reasoning_content"] = message.additional_kwargs["reasoning_content"]
        # Size-only telemetry; full opt-in scenes are handled at the HTTP boundary.
        self.payload_log.append({
            "message_chars": [len(str(message.get("content") or "")) for message in payload["messages"]],
            "message_roles": [message.get("role") for message in payload["messages"]],
            "reasoning_history_chars": sum(len(str(message.get("reasoning_content") or "")) for message in payload["messages"]),
            "tools_schema_chars": len(json.dumps(payload.get("tools", []))),
        })
        return payload

    @staticmethod
    def _parse_submission(message: AIMessage, schema: Any, tool: dict) -> Any:
        calls = message.additional_kwargs.get("tool_calls", [])
        if len(calls) != 1:
            raise RequiredToolError("Submit exactly one call to the target tool.")
        function = calls[0].get("function", {})
        expected = tool["function"]["name"]
        if function.get("name") != expected:
            raise RequiredToolError(f"Only the target tool {expected!r} is accepted.")
        arguments = json.loads(function.get("arguments", ""))
        if not isinstance(arguments, dict):
            raise RequiredToolError("Tool arguments must be a JSON object.")
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return schema.model_validate(arguments)
        validate_json_schema(arguments, tool["function"]["parameters"])
        return arguments

    @staticmethod
    def _repair(history: list[BaseMessage], message: AIMessage, error: Exception,
                name: str) -> None:
        history.append(message)  # includes reasoning_content, even without a tool call
        calls = message.additional_kwargs.get("tool_calls", [])
        feedback = (
            f"Submission rejected: {type(error).__name__}: {error}. "
            f"Submit exactly one {name} tool call with valid arguments. "
            "This is an output-format repair only: keep the same analysis, test, "
            "data and statistical result. Do not collect new evidence or seek a "
            "different significance result."
        )
        # Never dispatch any proposed function. Close every tool-call ID so the
        # next model request has a well-formed assistant/tool conversation.
        for call in calls:
            call_id = call.get("id")
            if not isinstance(call_id, str) or not call_id:
                raise RequiredToolError("Cannot repair a tool call without a tool_call_id.")
            history.append(ToolMessage(content=feedback, tool_call_id=call_id))
        history.append(HumanMessage(content=feedback))

    def with_structured_output(self, schema: Any = None, *,
                               method: Literal["function_calling", "json_mode", "json_schema"] = "function_calling",
                               include_raw: bool = False, strict: bool | None = None,
                               **kwargs: Any) -> Runnable[LanguageModelInput, Any]:
        native = super().with_structured_output(schema, method=method, include_raw=include_raw,
                                                 strict=strict, **kwargs)
        if method != "function_calling":
            return native
        tool = convert_to_openai_tool(schema, strict=strict)
        name = tool["function"]["name"]
        # ``auto`` is supported with DeepSeek thinking; no required/named choice.
        automatic = self.bind_tools([schema], tool_choice="auto", strict=strict,
                                    parallel_tool_calls=False)

        def prepare(value: LanguageModelInput) -> tuple[list[BaseMessage], dict]:
            instruction = (
                f"You must submit your result using exactly one call to the {name} tool. "
                "The tool packages your result; it does not execute an analysis. "
                "Do not use another tool or return only prose as your final submission. "
                "Fill the required parameters according to the supplied schema. "
                "If a submission is rejected, repair only its tool name/format/schema; "
                "preserve the analysis and statistical result."
            )
            history = [SystemMessage(content=instruction), *self._convert_input(value).to_messages()]
            audit = {"call_id": uuid4().hex, "tool": name, "attempts": 0, "retries": 0,
                     "accepted": False, "errors": [], "tool_names": []}
            self.schema_call_log.append(audit)
            # The installed LC version drops reasoning deltas when streaming.
            # Use complete responses in this mode so all reasoning can be replayed.
            self.disable_streaming = True
            self.streaming = False
            return history, audit

        def accept(message: BaseMessage, audit: dict, attempt: int) -> tuple[Any, Exception | None]:
            audit["attempts"], audit["retries"] = attempt + 1, attempt
            if not isinstance(message, AIMessage):
                return None, RequiredToolError("Expected an assistant message.")
            calls = message.additional_kwargs.get("tool_calls", [])
            audit["tool_names"].append([call.get("function", {}).get("name") for call in calls])
            try:
                parsed = self._parse_submission(message, schema, tool)
            except (ValueError, TypeError, KeyError) as error:
                audit["errors"].append(f"{type(error).__name__}: {error}")
                return None, error
            except Exception as error:
                # JSON Schema's ValidationError is not a ValueError.
                from jsonschema.exceptions import ValidationError
                if not isinstance(error, ValidationError):
                    raise
                audit["errors"].append(f"{type(error).__name__}: {error}")
                return None, error
            audit["accepted"] = True
            return parsed, None

        def finish(raw: BaseMessage | None, parsed: Any, error: Exception | None) -> Any:
            if include_raw:
                return {"raw": raw, "parsed": parsed, "parsing_error": error}
            if error is not None:
                raise RequiredToolError(
                    f"Could not submit {name} after {self.structured_max_retries + 1} attempts: {error}"
                ) from error
            return parsed

        def invoke(value: LanguageModelInput, config: RunnableConfig) -> Any:
            if self.structured_output_mode == "native":
                return native.invoke(value, config=config)
            history, audit = prepare(value)
            raw = None
            parsed = None
            error = None
            for attempt in range(self.structured_max_retries + 1):
                raw = automatic.invoke(history, config=config)
                parsed, error = accept(raw, audit, attempt)
                if error is None:
                    break
                if attempt < self.structured_max_retries:
                    if not isinstance(raw, AIMessage):
                        raise RequiredToolError("Cannot repair a non-assistant response.")
                    self._repair(history, raw, error, name)
            return finish(raw, parsed, error)

        async def ainvoke(value: LanguageModelInput, config: RunnableConfig) -> Any:
            if self.structured_output_mode == "native":
                return await native.ainvoke(value, config=config)
            history, audit = prepare(value)
            raw = None
            parsed = None
            error = None
            for attempt in range(self.structured_max_retries + 1):
                raw = await automatic.ainvoke(history, config=config)
                parsed, error = accept(raw, audit, attempt)
                if error is None:
                    break
                if attempt < self.structured_max_retries:
                    if not isinstance(raw, AIMessage):
                        raise RequiredToolError("Cannot repair a non-assistant response.")
                    self._repair(history, raw, error, name)
            return finish(raw, parsed, error)

        return RunnableLambda(invoke, afunc=ainvoke)
