"""Offline end-to-end checks of request recording, replay and context capture."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from popper import Popper
from popper.llm.compatible_model import CompatibleChatOpenAI
from popper.request_scene import configured_clients, export_scene, inspect_scene, replay_scene, save_run_context

KEY = "offline-private-key-123456"
BASE = "https://offline.example/v1"


def answer(text="ok"):
    return dict(id="offline", object="chat.completion", created=0, model="test-model",
                choices=[dict(index=0, message=dict(role="assistant", content=text,
                                                   reasoning_content="opaque-provider-state"), finish_reason="stop")],
                usage=dict(prompt_tokens=1, completion_tokens=1, total_tokens=2))


def model(root, transport, role="designer", **kwargs):
    return CompatibleChatOpenAI(model="test-model", base_url=BASE, api_key=KEY,
                                http_client=transport, request_scene_dir=str(root),
                                request_scene_role=role, max_retries=kwargs.pop("max_retries", 0), **kwargs)


def scene_paths(root):
    return sorted(path.parent for path in root.glob("*/*/metadata.json"))


def test_actual_merged_request_and_response_are_preserved(tmp_path):
    captured = []
    def handle(request):
        captured.append(request.content)
        assert request.headers["Authorization"] == "Bearer " + KEY
        return httpx.Response(200, json=answer())
    with httpx.Client(transport=httpx.MockTransport(handle)) as transport:
        client = model(tmp_path, transport, extra_body={"thinking": {"type": "enabled"}, "reasoning_effort": "high"})
        assert client.invoke("a private research prompt").content == "ok"
    [folder] = scene_paths(tmp_path)
    metadata, payload = inspect_scene(folder)
    assert (folder / "request_body.bin").read_bytes() == captured[0]
    assert payload["thinking"] == {"type": "enabled"} and payload["reasoning_effort"] == "high"
    assert metadata["request_body_exact"] and metadata["response_state"] == "complete"
    assert json.loads((folder / "response_body.bin").read_bytes())["choices"][0]["message"]["reasoning_content"] == "opaque-provider-state"
    assert all(KEY.encode() not in path.read_bytes() for path in tmp_path.rglob("*") if path.is_file())


def test_repair_snapshot_contains_tool_result_and_continuation(tmp_path):
    from pydantic import BaseModel
    class Verdict(BaseModel):
        valid: bool
    calls = []
    def handle(request):
        calls.append(json.loads(request.content))
        response = answer("not submitted")
        if len(calls) == 2:
            response["choices"][0]["message"]["tool_calls"] = [dict(id="call-2", type="function", function=dict(name="Verdict", arguments='{"valid":true}'))]
        return httpx.Response(200, json=response)
    with httpx.Client(transport=httpx.MockTransport(handle)) as transport:
        result = model(tmp_path, transport, structured_output_mode="prompt", structured_max_retries=1).with_structured_output(Verdict).invoke("Check validity")
        assert result.valid
    folders = scene_paths(tmp_path)
    assert len(folders) == 2
    second = next(folder for folder in folders if len(inspect_scene(folder)[1]["messages"]) > 2)
    exported = tmp_path / "exported"
    export_scene(second, exported)
    assert "opaque-provider-state" in (exported / "request.json").read_text()
    assert "opaque-provider-state" not in (exported / "request_public.json").read_text()
    inspect_scene(exported)


def test_async_concurrent_roles_use_their_own_context(tmp_path):
    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield json.dumps(answer()).encode()
    def handle(request):
        return httpx.Response(200, stream=Stream(), headers={"content-type": "application/json"})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as transport:
            clients = [CompatibleChatOpenAI(model="test-model", base_url=BASE, api_key=KEY,
                       http_async_client=transport, request_scene_dir=str(tmp_path), request_scene_role=role, max_retries=0)
                       for role in ["designer", "engine"]]
            return await asyncio.gather(*(client.ainvoke(role) for client, role in zip(clients, ["designer", "engine"])))
    assert len(asyncio.run(run())) == 2
    folders = scene_paths(tmp_path)
    assert len(folders) == 2
    for folder in folders:
        metadata, payload = inspect_scene(folder)
        assert payload["messages"][0]["content"] == metadata["role"]
        assert metadata["response_state"] == "complete"


def sse():
    item = dict(id="stream", object="chat.completion.chunk", created=0, model="test-model",
                choices=[dict(index=0, delta=dict(role="assistant", content="hello"), finish_reason=None)])
    end = dict(id="stream", object="chat.completion.chunk", created=0, model="test-model",
               choices=[dict(index=0, delta={}, finish_reason="stop")])
    return ("data: " + json.dumps(item) + "\n\ndata: " + json.dumps(end) + "\n\ndata: [DONE]\n\n").encode()


def test_streaming_capture_preserves_output_and_records_sse(tmp_path):
    body = sse()
    def handle(request):
        return httpx.Response(200, stream=httpx.ByteStream(body), headers={"content-type": "text/event-stream"})
    with httpx.Client(transport=httpx.MockTransport(handle)) as transport:
        text = "".join(chunk.content for chunk in model(tmp_path, transport).stream("stream please"))
    assert text == "hello"
    [folder] = scene_paths(tmp_path)
    metadata, payload = inspect_scene(folder)
    assert payload["stream"] is True and metadata["response_state"] == "complete"
    assert (folder / "response_body.bin").read_bytes() == body


def test_network_failure_retains_pre_send_request(tmp_path):
    def handle(request):
        raise httpx.ReadTimeout("offline timeout")
    with httpx.Client(transport=httpx.MockTransport(handle)) as transport:
        with pytest.raises(Exception):
            model(tmp_path, transport).invoke("failed attempt")
    [folder] = scene_paths(tmp_path)
    metadata, payload = inspect_scene(folder)
    assert payload["messages"][0]["content"] == "failed attempt"
    assert metadata["response_state"] == "not_recorded"
    assert metadata["invocation_error_type"] == "APITimeoutError"


def test_partial_stream_is_marked_partial(tmp_path):
    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            yield sse().split(b"\n\n")[0] + b"\n\n"
            raise httpx.ReadError("stream interrupted")
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=Stream(), headers={"content-type": "text/event-stream"}))) as transport:
        with pytest.raises(httpx.ReadError):
            list(model(tmp_path, transport).stream("stream"))
    [folder] = scene_paths(tmp_path)
    metadata, _ = inspect_scene(folder)
    assert metadata["response_state"] == "partial"
    assert metadata["response_error_type"] == "ReadError"
    assert (folder / "response_body.bin").read_bytes()


def test_command_line_replay_defaults_to_no_network(tmp_path, capsys):
    import sys
    from popper.request_scene import main
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=answer()))) as transport:
        model(tmp_path, transport).invoke("capture")
    [folder] = scene_paths(tmp_path)
    with patch.object(sys, "argv", ["scene", "replay", str(folder)]), patch.object(httpx.Client, "send", side_effect=AssertionError("Unexpected HTTP request")):
        main()
    assert json.loads(capsys.readouterr().out)["network_requests"] == 0


def test_sdk_retry_gets_separate_scenes_with_one_invocation(tmp_path):
    count = 0
    def handle(request):
        nonlocal count
        count += 1
        return httpx.Response(500, json={"error": "temporary"}) if count == 1 else httpx.Response(200, json=answer())
    with httpx.Client(transport=httpx.MockTransport(handle)) as transport:
        model(tmp_path, transport, max_retries=1).invoke("retry")
    records = [inspect_scene(folder)[0] for folder in scene_paths(tmp_path)]
    assert len(records) == 2 and {record["http_status"] for record in records} == {200, 500}
    assert len({record["invocation_id"] for record in records}) == 1


def test_credential_echo_is_redacted_and_blocks_exact_replay(tmp_path):
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=answer(KEY)))) as transport:
        model(tmp_path, transport).invoke("Echo " + KEY)
    [folder] = scene_paths(tmp_path)
    metadata, _ = inspect_scene(folder)
    assert metadata["credentials_redacted"] and not metadata["request_body_exact"]
    assert all(KEY.encode() not in path.read_bytes() for path in tmp_path.rglob("*") if path.is_file())
    with pytest.raises(ValueError, match="redacted"):
        replay_scene(folder, tmp_path / "blocked", KEY, endpoint=BASE + "/chat/completions")


def test_replay_sends_exact_body_once_to_explicit_endpoint(tmp_path):
    original = []
    def handle(request):
        original.append(request.content)
        return httpx.Response(200, json=answer())
    with httpx.Client(transport=httpx.MockTransport(handle)) as transport:
        model(tmp_path / "capture", transport).invoke("replay me")
    [folder] = scene_paths(tmp_path / "capture")
    with pytest.raises(ValueError, match="endpoint"):
        replay_scene(folder, tmp_path / "wrong", KEY, endpoint="https://other.example/v1/chat/completions")
    with httpx.Client(transport=httpx.MockTransport(handle)) as transport:
        assert replay_scene(folder, tmp_path / "replay", KEY, endpoint=BASE + "/chat/completions", client=transport) == 200
        with pytest.raises(FileExistsError):
            replay_scene(folder, tmp_path / "replay", KEY, endpoint=BASE + "/chat/completions", client=transport)
    assert original == [original[0], original[0]]


def test_context_export_carries_inputs_and_detects_mutation(tmp_path):
    data = tmp_path / "input.csv"
    data.write_text("X,Y\n1,2\n", encoding="utf8")
    root = tmp_path / "scenes"
    save_run_context(root, {"model": "test"}, {"observation.csv": data}, {})
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=answer()))) as transport:
        model(root, transport).invoke("context")
    [folder] = scene_paths(root)
    exported = tmp_path / "portable"
    export_scene(folder, exported)
    assert (exported / "context/inputs/observation.csv").read_bytes() == data.read_bytes()
    inspect_scene(exported)
    (exported / "context/inputs/observation.csv").write_text("changed")
    with pytest.raises(ValueError, match="Context file"):
        inspect_scene(exported)


def test_public_popper_option_reaches_all_clients(tmp_path):
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=answer()))) as transport:
        def factory(**kwargs):
            return CompatibleChatOpenAI(**kwargs, http_client=transport, max_retries=0)
        with patch("popper.utils.CompatibleChatOpenAI", side_effect=factory):
            popper = Popper(llm="test-model", base_url=BASE, api_key=KEY, domain="statistics", request_scene_dir=str(tmp_path))
            popper.data_loader = SimpleNamespace(table_dict={}, data_desc="X,Z,Y")
            popper.configure(use_react_agent=True, execution_mode="native_tools", max_num_of_tests=1)
            for role, client in configured_clients(popper).items():
                if client is not None:
                    client.invoke(role)
    records = [inspect_scene(folder)[0] for folder in scene_paths(tmp_path)]
    assert {record["role"] for record in records} == {"engine", "designer", "pvalue_parser", "react_executor"}
