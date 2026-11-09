"""Contract tests for prompt-required tools and thinking history replay."""

import asyncio
import json
from pathlib import Path
import tempfile
import unittest

import httpx
from pydantic import BaseModel, SecretStr

from popper.llm.compatible_model import CompatibleChatOpenAI, RequiredToolError


class Verdict(BaseModel):
    valid: bool


def response(kind: str, index: int) -> dict:
    message = {"role": "assistant", "content": None, "reasoning_content": f"opaque-state-{index}"}
    if kind == "no_tool":
        message["content"] = "A prose answer instead of a submission."
    else:
        arguments = '{"valid": true}'
        if kind == "bad_arguments":
            arguments = '{"valid": "not-a-boolean"}'
        if kind == "bad_json":
            arguments = '{invalid JSON'
        message["tool_calls"] = [{"id": f"call-{index}", "type": "function", "function": {
            "name": "unwanted_tool" if kind == "wrong_tool" else "Verdict",
            "arguments": arguments,
        }}]
        if kind == "multiple_tools":
            message["tool_calls"].append({"id": f"extra-{index}", "type": "function", "function": {
                "name": "unwanted_tool", "arguments": '{}',
            }})
    return {"id": f"offline-{index}", "object": "chat.completion", "created": 0,
            "model": "deepseek-flash", "choices": [{"index": 0, "message": message,
                                                      "finish_reason": "tool_calls" if kind != "no_tool" else "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


class PromptRequiredTests(unittest.TestCase):
    def test_usage_is_journaled_immediately_without_prompt_reasoning_or_credentials(self):
        def handler(request):
            body = json.loads(request.content)
            self.assertEqual(body['reasoning_effort'], 'high')
            return httpx.Response(200, json=response('valid', 1))

        with tempfile.TemporaryDirectory() as directory, httpx.Client(transport=httpx.MockTransport(handler)) as transport:
            path = Path(directory) / 'usage' / 'executor.jsonl'
            client = self.client(transport)
            client.extra_body = dict(client.extra_body or {}, reasoning_effort='high')
            client.usage_log_path = str(path)
            client.invoke('private-prompt-text')
            saved = path.read_text(encoding='utf-8')
            record = json.loads(saved)
            self.assertEqual(record['usage']['total_tokens'], 2)
            self.assertIn('received_at_utc', record)
            self.assertNotIn('opaque-state', saved)
            self.assertNotIn('private-prompt-text', saved)
            self.assertNotIn('offline-test-key', saved)

    def client(self, transport, **kwargs):
        return CompatibleChatOpenAI(model="deepseek-flash", api_key=SecretStr("offline-test-key"),
                                    base_url="https://offline.example/v1", http_client=transport,
                                    extra_body={"thinking": {"type": "enabled"}}, **kwargs)

    def test_dynamic_mode_keeps_thinking_and_auto_choice(self):
        requests = []

        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            return httpx.Response(200, json=response("valid", len(requests)))

        with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
            client = self.client(transport)
            bound = client.with_structured_output(Verdict)
            # POPPER constructs bindings before the runner selects the policy.
            client.structured_output_mode = "prompt"
            result = bound.invoke("Return validity")
        self.assertIsInstance(result, Verdict)
        self.assertTrue(result.valid)
        self.assertEqual(requests[0]["tool_choice"], "auto")
        self.assertEqual(requests[0]["thinking"], {"type": "enabled"})
        self.assertEqual([t["function"]["name"] for t in requests[0]["tools"]], ["Verdict"])
        self.assertEqual(client.schema_call_log[0]["retries"], 0)

    def test_repair_wrong_name_no_call_malformed_json_and_invalid_schema(self):
        for kind in ("no_tool", "wrong_tool", "bad_json", "bad_arguments", "multiple_tools"):
            with self.subTest(kind=kind):
                requests = []

                def handler(request):
                    requests.append(json.loads(request.content))
                    return httpx.Response(200, json=response(kind if len(requests) == 1 else "valid", len(requests)))

                with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
                    client = self.client(transport, structured_output_mode="prompt")
                    result = client.with_structured_output(Verdict).invoke("Keep the analysis fixed")
                self.assertIsInstance(result, Verdict)
                self.assertTrue(result.valid)
                self.assertEqual(len(requests), 2)
                assistants = [m for m in requests[1]["messages"] if m["role"] == "assistant"]
                self.assertEqual(assistants[0]["reasoning_content"], "opaque-state-1")
                tool_results = [m for m in requests[1]["messages"] if m["role"] == "tool"]
                self.assertEqual(len(tool_results), 0 if kind == "no_tool" else 2 if kind == "multiple_tools" else 1)
                self.assertTrue(all("rejected" in m["content"] for m in tool_results))
                self.assertIn("keep the same analysis", requests[1]["messages"][-1]["content"])
                self.assertEqual(client.schema_call_log[0]["retries"], 1)
                self.assertTrue(client.schema_call_log[0]["accepted"])

    def test_exhaustion_is_failure_and_never_a_default_answer(self):
        requests = []

        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json=response("wrong_tool", len(requests)))

        with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
            client = self.client(transport, structured_output_mode="prompt", structured_max_retries=1)
            with self.assertRaises(RequiredToolError):
                client.with_structured_output(Verdict).invoke("Return validity")
        self.assertEqual(len(requests), 2)
        self.assertFalse(client.schema_call_log[0]["accepted"])

    def test_include_raw_preserves_success_and_failure_contract(self):
        for kind in ("valid", "no_tool"):
            with self.subTest(kind=kind):
                with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response(kind, 1)))) as transport:
                    client = self.client(transport, structured_output_mode="prompt", structured_max_retries=0)
                    result = client.with_structured_output(Verdict, include_raw=True).invoke("Return validity")
                self.assertEqual(set(result), {"raw", "parsed", "parsing_error"})
                self.assertEqual(result["raw"].additional_kwargs["reasoning_content"], "opaque-state-1")
                if kind == "valid":
                    self.assertIsInstance(result["parsed"], Verdict)
                    self.assertIsNone(result["parsing_error"])
                else:
                    self.assertIsNone(result["parsed"])
                    self.assertIsInstance(result["parsing_error"], RequiredToolError)

    def test_dict_schema_is_locally_validated(self):
        schema = {"name": "Verdict", "parameters": {"type": "object", "required": ["valid"],
                                                   "properties": {"valid": {"type": "boolean"}}}}
        requests = []

        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json=response("bad_arguments" if len(requests) == 1 else "valid", len(requests)))

        with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
            client = self.client(transport, structured_output_mode="prompt")
            result = client.with_structured_output(schema).invoke("Return validity")
        self.assertEqual(result, {"valid": True})
        self.assertEqual(len(requests), 2)

    def test_async_repairs_also_replay_reasoning(self):
        requests = []

        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json=response("no_tool" if len(requests) == 1 else "valid", len(requests)))

        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
                client = CompatibleChatOpenAI(model="deepseek-flash", api_key=SecretStr("offline-test-key"),
                                              base_url="https://offline.example/v1", http_async_client=transport,
                                              structured_output_mode="prompt",
                                              extra_body={"thinking": {"type": "enabled"}})
                return await client.with_structured_output(Verdict).ainvoke("Return validity")

        result = asyncio.run(run())
        self.assertIsInstance(result, Verdict)
        self.assertEqual(requests[1]["messages"][-2]["reasoning_content"], "opaque-state-1")


if __name__ == "__main__":
    unittest.main()
