"""Capture and export individual model requests at the HTTP boundary.

Request headers are never stored. Provider continuation fields are preserved in
the raw body so a later request can be replayed without inventing missing state.
Human-readable exports omit those fields. This is a model-call checkpoint, not
a serializer for an executing Python kernel.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import httpx

_ACTIVE = ContextVar("popper_request_scene", default=None)
_EXTENSION = "popper_request_scene"
_OPAQUE_FIELDS = {"reasoning_content", "thinking", "encrypted_content", "signature"}


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _json(path, value):
    _write(path, json.dumps(value, ensure_ascii=False, indent=2).encode("utf8"))


def public_view(value):
    if isinstance(value, dict):
        return {key: ("[PROVIDER_CONTINUATION_OMITTED]" if key in _OPAQUE_FIELDS else public_view(item))
                for key, item in value.items()}
    if isinstance(value, list):
        return [public_view(item) for item in value]
    return value


class RequestSceneRecorder:
    def __init__(self, root, role, secrets=()):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", role):
            raise ValueError("Scene role must be a simple name")
        self.root = Path(root)
        self.role = role
        self.secrets = tuple(secret.encode("utf8") for secret in secrets if secret and secret != "EMPTY")
        self.scenes = []

    def clean(self, data):
        for secret in self.secrets:
            data = data.replace(secret, b"[REDACTED_CREDENTIAL]")
        return data

    def start(self, request, invocation_id):
        original = request.content
        body = self.clean(original)
        parsed = json.loads(body)
        ident = uuid4().hex
        folder = self.root / self.role / ident
        folder.mkdir(parents=True, exist_ok=False, mode=0o700)
        url = urlsplit(str(request.url))
        if url.username or url.password:
            raise ValueError("Credential-bearing endpoint URL is not supported")
        safe_url = self.clean(urlunsplit(url).encode("utf8")).decode("utf8")
        versions = {name: importlib.metadata.version(name) for name in ["openai", "httpx", "langchain-openai", "langchain-core"]}
        metadata = dict(
            format="popper-http-scene-v1", scene_id=ident, invocation_id=invocation_id,
            role=self.role, started_utc=_utc(), method=request.method, url=safe_url,
            model=parsed.get("model"), request_sha256=_sha(body), request_bytes=len(body),
            transport_timeout=request.extensions.get("timeout"),
            credentials_redacted=body != original or safe_url != str(request.url),
            request_body_exact=body == original, headers_saved=False,
            request_state="saved_before_send", response_state="not_recorded", versions=versions,
        )
        context_manifest = self.root / "context" / "manifest.json"
        if context_manifest.exists():
            metadata["context_manifest_sha256"] = _sha(context_manifest.read_bytes())
        _write(folder / "request_body.bin", body)
        _json(folder / "request.json", parsed)
        _json(folder / "metadata.json", metadata)
        self.scenes.append((folder, invocation_id))
        request.extensions[_EXTENSION] = (self, folder, metadata)

    def response(self, folder, metadata, response, data, complete, error_type=None):
        safe = self.clean(data)
        _write(folder / "response_body.bin", safe)
        metadata = dict(metadata, completed_utc=_utc(), http_status=response.status_code,
                        response_content_type=response.headers.get("content-type"),
                        response_content_encoding=response.headers.get("content-encoding"),
                        response_state="complete" if complete else "partial",
                        response_sha256=_sha(safe), response_bytes=len(safe),
                        response_credentials_redacted=safe != data, response_error_type=error_type)
        _json(folder / "metadata.json", metadata)


class _SyncRecordedStream(httpx.SyncByteStream):
    def __init__(self, stream, response, record):
        self.stream, self.response, self.record = stream, response, record
        self.buffer = bytearray()
        self.saved = False

    def save(self, complete, error_type=None):
        if not self.saved:
            recorder, folder, metadata = self.record
            recorder.response(folder, metadata, self.response, bytes(self.buffer), complete, error_type)
            self.saved = True

    def __iter__(self):
        try:
            for chunk in self.stream:
                self.buffer.extend(chunk)
                yield chunk
        except BaseException as error:
            self.save(False, type(error).__name__)
            raise
        else:
            self.save(True)

    def close(self):
        try:
            self.stream.close()
        finally:
            self.save(False)


class _AsyncRecordedStream(httpx.AsyncByteStream):
    def __init__(self, stream, response, record):
        self.stream, self.response, self.record = stream, response, record
        self.buffer = bytearray()
        self.saved = False

    def save(self, complete, error_type=None):
        if not self.saved:
            recorder, folder, metadata = self.record
            recorder.response(folder, metadata, self.response, bytes(self.buffer), complete, error_type)
            self.saved = True

    async def __aiter__(self):
        try:
            async for chunk in self.stream:
                self.buffer.extend(chunk)
                yield chunk
        except BaseException as error:
            self.save(False, type(error).__name__)
            raise
        else:
            self.save(True)

    async def aclose(self):
        try:
            await self.stream.aclose()
        finally:
            self.save(False)


def _request_hook(request):
    active = _ACTIVE.get()
    if active is not None and _EXTENSION not in request.extensions:
        recorder, invocation_id = active
        recorder.start(request, invocation_id)


def _response_hook(response):
    record = response.request.extensions.get(_EXTENSION)
    if record is not None:
        if response.is_stream_consumed:
            recorder, folder, metadata = record
            recorder.response(folder, metadata, response, response.content, True)
        else:
            response.stream = _SyncRecordedStream(response.stream, response, record)


async def _async_request_hook(request):
    await request.aread()
    _request_hook(request)


async def _async_response_hook(response):
    record = response.request.extensions.get(_EXTENSION)
    if record is not None:
        if response.is_stream_consumed:
            recorder, folder, metadata = record
            recorder.response(folder, metadata, response, response.content, True)
        else:
            response.stream = _AsyncRecordedStream(response.stream, response, record)


def install_hooks(client):
    for http_client, request_hook, response_hook in [
        (client.root_client._client, _request_hook, _response_hook),
        (client.root_async_client._client, _async_request_hook, _async_response_hook),
    ]:
        if request_hook not in http_client.event_hooks["request"]:
            http_client.event_hooks["request"].append(request_hook)
        if response_hook not in http_client.event_hooks["response"]:
            http_client.event_hooks["response"].append(response_hook)


def configured_clients(agent):
    engine = getattr(agent, "agent", None)
    coder = getattr(engine, "test_coding_agent", None)
    clients = {
        "engine": getattr(engine, "llm", None),
        "designer": getattr(getattr(engine, "test_proposal_agent", None), "llm", None),
        "pvalue_parser": getattr(coder, "llm", None),
        "react_executor": getattr(getattr(coder, "agent", None), "llm", None),
    }
    likelihood = getattr(getattr(engine, "likelihood_estimation_agent", None), "llm", None)
    if likelihood is not None:
        clients["likelihood_estimator"] = likelihood
    return clients


def save_run_context(root, configuration, public_files, source_files, secrets=(), max_copy_bytes=64 * 1024 * 1024):
    """Save public inputs/source once; large files retain a hash and original path."""
    context = Path(root) / "context"
    context.mkdir(parents=True, exist_ok=False, mode=0o700)
    cleaner = RequestSceneRecorder(root, "context", secrets)
    _write(context / "configuration.json", cleaner.clean(json.dumps(configuration, ensure_ascii=False, indent=2).encode("utf8")))
    files = {}
    for kind, entries in [("inputs", public_files), ("source", source_files)]:
        for name, original in entries.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Context filenames must stay within the snapshot")
            original = Path(original)
            digest = hashlib.sha256()
            with original.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            record = dict(original_path=cleaner.clean(str(original.resolve()).encode()).decode(),
                          original_sha256=digest.hexdigest(), bytes=original.stat().st_size,
                          copied=original.stat().st_size <= max_copy_bytes)
            target_name = (Path(kind) / relative).as_posix()
            if record["copied"]:
                original_bytes = original.read_bytes()
                stored = cleaner.clean(original_bytes)
                _write(context / target_name, stored)
                record.update(stored_sha256=_sha(stored), credentials_redacted=stored != original_bytes)
            files[target_name] = record
    _json(context / "manifest.json", dict(format="popper-run-context-v1", created_utc=_utc(),
                                          configuration_sha256=_sha((context / "configuration.json").read_bytes()),
                                          files=files, max_copy_bytes=max_copy_bytes,
                                          scope="Public input files, source and configuration; Python process memory is not serialized."))


def _context_directory(folder, metadata):
    if "context_manifest_sha256" not in metadata:
        return None
    candidates = [folder / "context", folder.parent.parent / "context"]
    for context in candidates:
        manifest_path = context / "manifest.json"
        if manifest_path.is_file() and _sha(manifest_path.read_bytes()) == metadata["context_manifest_sha256"]:
            manifest = json.loads(manifest_path.read_text(encoding="utf8"))
            if _sha((context / "configuration.json").read_bytes()) != manifest["configuration_sha256"]:
                raise ValueError("Context configuration hash mismatch")
            for name, record in manifest["files"].items():
                relative = Path(name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Invalid context path")
                if record["copied"] and _sha((context / relative).read_bytes()) != record["stored_sha256"]:
                    raise ValueError("Context file hash mismatch: " + name)
            return context
    raise ValueError("Scene's run context is missing or changed")


@contextmanager
def recording_scope(client, invocation_id=None):
    recorder = None
    if client.request_scene_dir:
        install_hooks(client)
        secret = client.openai_api_key.get_secret_value() if client.openai_api_key else ""
        recorder = RequestSceneRecorder(client.request_scene_dir, client.request_scene_role, [secret])
    active = _ACTIVE.get()
    if recorder and active and active[0].root == recorder.root and active[0].role == recorder.role:
        value = active
    else:
        value = (recorder, invocation_id or uuid4().hex) if recorder else None
    token = _ACTIVE.set(value)
    try:
        yield
    except BaseException as error:
        if value is not None:
            for folder, recorded_invocation in value[0].scenes:
                if recorded_invocation == value[1]:
                    path = folder / "metadata.json"
                    metadata = json.loads(path.read_text(encoding="utf8"))
                    metadata["invocation_error_type"] = type(error).__name__
                    _json(path, metadata)
        raise
    finally:
        _ACTIVE.reset(token)


def inspect_scene(folder):
    folder = Path(folder)
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf8"))
    body = (folder / "request_body.bin").read_bytes()
    if _sha(body) != metadata["request_sha256"]:
        raise ValueError("Request scene hash mismatch")
    parsed = json.loads(body)
    if parsed != json.loads((folder / "request.json").read_text(encoding="utf8")):
        raise ValueError("Readable request differs from captured body")
    response_file = folder / "response_body.bin"
    if "response_sha256" in metadata and _sha(response_file.read_bytes()) != metadata["response_sha256"]:
        raise ValueError("Response scene hash mismatch")
    _context_directory(folder, metadata)
    return metadata, parsed


def export_scene(folder, destination):
    folder, destination = Path(folder), Path(destination)
    metadata, payload = inspect_scene(folder)
    destination.mkdir(parents=True, exist_ok=False)
    _write(destination / "request_body.bin", (folder / "request_body.bin").read_bytes())
    _json(destination / "request.json", payload)
    _json(destination / "request_public.json", public_view(payload))
    _json(destination / "metadata.json", metadata)
    if (folder / "response_body.bin").exists():
        _write(destination / "response_body.bin", (folder / "response_body.bin").read_bytes())
        try:
            response = json.loads((folder / "response_body.bin").read_bytes())
        except (ValueError, UnicodeDecodeError):
            response = {"note": "The raw response is streamed or encoded; inspect response metadata and the original body."}
        _json(destination / "response_public.json", public_view(response))
    context = _context_directory(folder, metadata)
    if context is not None:
        shutil.copytree(context, destination / "context")
    _json(destination / "manifest.json", {path.relative_to(destination).as_posix(): _sha(path.read_bytes()) for path in destination.rglob("*") if path.is_file()})


def replay_scene(folder, output, key, *, endpoint, client=None, timeout=300):
    """Send exactly one recorded HTTP body; never execute any returned tool call."""
    folder, output = Path(folder), Path(output)
    metadata, payload = inspect_scene(folder)
    if metadata["credentials_redacted"]:
        raise ValueError("A credential-redacted scene cannot be replayed as an exact request")
    if payload.get("stream"):
        raise ValueError("Streaming scenes can be exported; this replay command accepts non-streaming requests")
    url = urlsplit(metadata["url"])
    if url.scheme != "https" or url.username or url.password:
        raise ValueError("Replay requires the captured credential-free HTTPS endpoint")
    if endpoint != metadata["url"]:
        raise ValueError("Explicit replay endpoint does not match the captured endpoint")
    output.mkdir(parents=True, exist_ok=False)
    _json(output / "attempt.json", dict(started_utc=_utc(), source_scene=metadata["scene_id"], url=metadata["url"], requests=1))
    context = _context_directory(folder, metadata)
    if context is not None:
        shutil.copytree(context, output / "captured" / "context")
    owned_client = client is None
    client = client or httpx.Client(timeout=timeout, follow_redirects=False)
    recorder = RequestSceneRecorder(output / "captured", "replay", [key])
    request = client.build_request(metadata["method"], metadata["url"], content=(folder / "request_body.bin").read_bytes(),
                                   headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    recorder.start(request, uuid4().hex)
    try:
        response = client.send(request, follow_redirects=False)
        _, capture_folder, capture_metadata = request.extensions[_EXTENSION]
        recorder.response(capture_folder, capture_metadata, response, response.content, True)
        _json(output / "result.json", dict(http_status=response.status_code, completed_utc=_utc(), requests=1))
        return response.status_code
    except Exception as error:
        _json(output / "result.json", dict(error_type=type(error).__name__, requests=1, completed_utc=_utc()))
        raise
    finally:
        if owned_client:
            client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["list", "inspect", "export", "replay"])
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--send", action="store_true", help="Actually send one potentially billable model request")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--endpoint", help="Explicit credential destination, exactly matching the inspected scene URL")
    args = parser.parse_args()
    if args.command == "list":
        records = [json.loads(path.read_text(encoding="utf8")) for path in args.path.rglob("metadata.json")]
        records = [row for row in records if row.get("format") == "popper-http-scene-v1"]
        print(json.dumps(sorted(records, key=lambda row: row["started_utc"]), ensure_ascii=False, indent=2))
        return
    metadata, _ = inspect_scene(args.path)
    if args.command == "inspect" or args.command == "replay" and not args.send:
        print(json.dumps(dict(metadata, network_requests=0, replay_send_flag_required=True), ensure_ascii=False, indent=2))
    elif args.command == "export":
        if args.output is None:
            parser.error("export requires --output")
        export_scene(args.path, args.output)
    else:
        if args.output is None or not args.endpoint or not os.environ.get(args.api_key_env):
            parser.error("replay --send requires --output, --endpoint, and the named API-key environment variable")
        print(replay_scene(args.path, args.output, os.environ[args.api_key_env], endpoint=args.endpoint))


if __name__ == "__main__":
    main()
