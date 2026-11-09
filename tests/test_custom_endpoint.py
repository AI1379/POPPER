"""Offline regression tests for custom endpoints and existing LLM routing."""

import contextlib
import io
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from popper import Popper
from popper.agent import SequentialFalsificationTest
from popper.react_agent import ReactAgent
from popper.utils import get_llm


BASE_URL = "https://model-service.example/custom/v1"
MODEL = "custom/research-model"
KEY = "explicit-offline-test-key"


class Verdict(BaseModel):
    valid: bool


class CustomEndpointTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {
            "OPENAI_API_KEY": "environment-offline-test-key",
            "ANTHROPIC_API_KEY": "anthropic-offline-test-key",
        })
        environment.start()
        self.addCleanup(environment.stop)
        # Any accidental real HTTP request fails locally. MockTransport never
        # opens a socket, so the request/structured-output test still runs.
        network = patch("socket.socket.connect", side_effect=AssertionError("Real network is forbidden in these tests"))
        network.start()
        self.addCleanup(network.stop)

    def assert_endpoint(self, llm):
        self.assertIsInstance(llm, ChatOpenAI)
        self.assertEqual(llm.model_name, MODEL)
        self.assertEqual(llm.openai_api_base, BASE_URL)
        self.assertEqual(str(llm.root_client.base_url), BASE_URL + "/")
        self.assertEqual(str(llm.root_async_client.base_url), BASE_URL + "/")
        self.assertEqual(llm.openai_api_key.get_secret_value(), KEY)

    def test_explicit_url_overrides_all_model_name_prefixes_and_port(self):
        for model in (MODEL, "gpt-custom", "claude-custom", "o1-custom", "o3-custom"):
            with self.subTest(model=model):
                llm = get_llm(model, base_url=BASE_URL, port=1234, api_key=KEY)
                self.assertIsInstance(llm, ChatOpenAI)
                self.assertEqual(llm.model_name, model)
                self.assertEqual(llm.openai_api_base, BASE_URL)
                self.assertEqual(llm.openai_api_key.get_secret_value(), KEY)

    def test_endpoint_key_falls_back_to_environment(self):
        for key in (None, "", "EMPTY"):
            with self.subTest(key=key):
                llm = get_llm(MODEL, base_url=BASE_URL, api_key=key)
                self.assertEqual(llm.openai_api_key.get_secret_value(), "environment-offline-test-key")

    def test_unauthenticated_endpoint_uses_placeholder(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            llm = get_llm(MODEL, base_url=BASE_URL)
        self.assertEqual(llm.openai_api_key.get_secret_value(), "EMPTY")

    def test_invalid_explicit_base_url_fails_without_provider_fallback(self):
        for url in ("", "  ", 123):
            with self.subTest(url=url), self.assertRaisesRegex(ValueError, "base_url"):
                get_llm("claude-custom", base_url=url)

    def test_actual_sdk_request_and_structured_output_use_custom_endpoint(self):
        requests = []

        def respond(request):
            requests.append(request)
            body = json.loads(request.content)
            return httpx.Response(200, json={
                "id": "chatcmpl-offline", "object": "chat.completion", "created": 0,
                "model": MODEL,
                "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                    "role": "assistant", "content": None,
                    "tool_calls": [{"id": "call-offline", "type": "function", "function": {
                        "name": body["tools"][0]["function"]["name"],
                        "arguments": '{"valid": true}',
                    }}],
                }}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            llm = get_llm(MODEL, base_url=BASE_URL, api_key=KEY, http_client=client)
            result = llm.with_structured_output(Verdict).invoke("Return a validity verdict.")
        self.assertTrue(result.valid)
        self.assertEqual(len(requests), 1)
        self.assertEqual(str(requests[0].url), BASE_URL + "/chat/completions")
        self.assertEqual(requests[0].headers["Authorization"], "Bearer " + KEY)
        self.assertEqual(json.loads(requests[0].content)["model"], MODEL)

    def test_public_popper_configuration_reaches_both_executor_types(self):
        for react in (True, False):
            with self.subTest(react=react), contextlib.redirect_stdout(io.StringIO()):
                agent = Popper(llm=MODEL, base_url=BASE_URL, api_key=KEY, domain="statistics")
                agent.data_loader = SimpleNamespace(table_dict={}, data_desc="Observations: X, Z, Y")
                agent.configure(use_react_agent=react, max_num_of_tests=1, max_retry=1)
                engine = agent.agent
                self.assert_endpoint(engine.llm)  # summarizer, relevance checker, final parser
                self.assert_endpoint(engine.test_proposal_agent.llm)
                self.assert_endpoint(engine.test_coding_agent.llm)
                if react:
                    self.assert_endpoint(engine.test_coding_agent.agent.llm)
                    self.assertEqual(engine.test_coding_agent.agent.api, "openai")

    def test_likelihood_estimator_receives_endpoint(self):
        with contextlib.redirect_stdout(io.StringIO()):
            agent = Popper(llm=MODEL, base_url=BASE_URL, api_key=KEY,
                           domain="statistics", llm_approx=True)
            agent.data_loader = SimpleNamespace(table_dict={}, data_desc="Observations: X, Z, Y")
            agent.configure(use_react_agent=False, max_num_of_tests=1, max_retry=1)
        self.assert_endpoint(agent.agent.likelihood_estimation_agent.llm)

    def test_explicit_url_removes_local_port_requirement(self):
        engine = SequentialFalsificationTest(llm=MODEL, is_local=True, base_url=BASE_URL, api_key=KEY)
        self.assert_endpoint(engine.llm)
        with self.assertRaisesRegex(AssertionError, "server port"):
            SequentialFalsificationTest(llm=MODEL, is_local=True)

    def test_legacy_cloud_factories_keep_their_arguments(self):
        with patch("popper.utils.ChatOpenAI") as factory:
            get_llm("gpt-existing", api_key=KEY)
            factory.assert_called_once_with(model="gpt-existing", temperature=0.7)
        with patch("popper.utils.ChatOpenAI") as factory:
            get_llm("o1-existing")
            factory.assert_called_once_with(model="o1-existing", temperature=-1)
        with patch("popper.utils.ChatAnthropic") as factory:
            get_llm("claude-existing")
            factory.assert_called_once_with(model="claude-existing", temperature=0.7, max_tokens=4096)

    def test_legacy_local_endpoint_remains_local(self):
        with patch("popper.utils.CustomChatModel") as factory, patch("popper.utils.openai.Client") as client:
            llm = get_llm("local-existing", port=8765, api_key=KEY)
            factory.assert_called_once_with(model="local-existing", model_type="Local", temperature=0.7)
            client.assert_called_once_with(base_url="http://127.0.0.1:8765/v1", api_key=KEY)
            self.assertIs(llm.client, client.return_value.chat.completions)

    def test_legacy_react_cloud_and_local_factories_are_preserved(self):
        agent = ReactAgent.__new__(ReactAgent)
        with patch("popper.react_agent.ChatOpenAI") as factory:
            agent.get_model("openai", "gpt-existing")
            factory.assert_called_once_with(model="gpt-existing", api_key="environment-offline-test-key")
        with patch("popper.react_agent.ChatAnthropic") as factory:
            agent.get_model("anthropic", "claude-existing")
            factory.assert_called_once_with(model="claude-existing", api_key="anthropic-offline-test-key")
        with patch("popper.react_agent.CustomChatModel") as factory, patch("popper.react_agent.openai.Client") as client:
            agent.get_model("local", "local-existing", port=8765, api_key=KEY)
            factory.assert_called_once_with(model="local-existing", model_type="custom")
            client.assert_called_once_with(base_url="http://127.0.0.1:8765/v1", api_key=KEY)


if __name__ == "__main__":
    unittest.main()
