"""Auditable text/native executors sharing an execution-backed result gate.

This verifies execution provenance, not statistical validity or resistance to
malicious Python. Generated code can still compute the wrong test or constants.
"""
from __future__ import annotations

import ast
import json
import math
import re
from typing import Any, Literal
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from popper.react_utils import CustomPromptTemplate, CustomPythonAstREPLTool, template
from popper.tool_output import bound_observation

ExecutionMode = Literal["legacy", "guarded_text", "native_tools"]

RESULT_INSTRUCTION = """
Execute Python against the supplied data before reporting a test result. Never
invent tool observations or simulate an execution transcript in your reply.
After computing the final p-value, call record_test_result(p_value, n=sample_size,
estimate=effect_estimate) INSIDE the Python tool. n and estimate are optional.
This host helper prints a result_id tied to that execution. Preserve numerical
warnings and use stable tail probabilities; p must be finite and 0 < p <= 1.
Submit the resulting ID to finish. An ID from an earlier test is invalid.
Registration proves execution only, not validity: implement the requested test
faithfully; do not substitute constants, simulated data, or a different estimand.
Exception: if no valid test is possible, or the proposed H1 is incompatible with
the result, register the conservative fallback with
record_test_result(1.0, fallback_reason="specific reason the test cannot support H1").
This is an explicit non-rejection, not an execution failure. Do not invent a small
p-value. Ordinary computed p-values, including a computed 1.0, need no reason.
On protocol rejection, repair the submission and preserve the analysis/result;
do not search for a more significant result. Keep output focused on diagnostics,
sample sizes, estimates, uncertainty and p-values. Full output is archived and
can be retrieved with read_tool_output(handle, start=0, length=4000, raw=False).
"""

# Native tools do not render the legacy ReAct template. Keep its statistical
# requirements in the actual native message, without importing text Action /
# Observation syntax or the unrelated demonstration dataset.
STATISTICAL_INTEGRITY_INSTRUCTION = (
    "\nStatistical requirements inherited from the original POPPER execution template. "
    "In this native-tool workflow, 'Final Answer' or 'return a p-value' means "
    "register the computed result and submit its ID with finish_test; keep using "
    "native tool calls rather than a text ReAct transcript.\n\n"
    + "IMPORTANT: Please make sure the Final Answer"
    + template.split("IMPORTANT: Please make sure the Final Answer", 1)[1]
      .split("Note: all datasets", 1)[0]
    + "\n**IMPORTANT**\n"
    + template.split("**IMPORTANT**\n", 1)[1].split("\n\nBegin!", 1)[0]
)


class ProtocolError(ValueError):
    pass


class ResultRegistry:
    def __init__(self, tool: CustomPythonAstREPLTool):
        self.tool = tool
        self.records: dict[str, dict[str, Any]] = {}

    def record(self, p_value: Any, *, n: Any = None, estimate: Any = None,
               fallback_reason: str | None = None) -> str:
        p = float(p_value)
        if isinstance(p_value, bool) or not math.isfinite(p) or not 0 < p <= 1:
            raise ProtocolError("p_value must be finite and 0 < p <= 1")
        if n is not None and (isinstance(n, bool) or int(n) != n or n <= 0):
            raise ProtocolError("n must be a positive integer")
        if estimate is not None and not math.isfinite(float(estimate)):
            raise ProtocolError("estimate must be finite")
        if fallback_reason is not None:
            if p != 1.0 or not isinstance(fallback_reason, str) or not fallback_reason.strip():
                raise ProtocolError("A conservative fallback requires p=1 and a nonempty reason")
            fallback_reason = fallback_reason.strip()
        result_id = "result_" + uuid4().hex
        record = dict(result_id=result_id, p_value=p, n=int(n) if n is not None else None,
                      estimate=float(estimate) if estimate is not None else None,
                      execution_index=len(self.tool.executions), fallback_reason=fallback_reason)
        self.records[result_id] = record
        print("REGISTERED_TEST_RESULT " + json.dumps(record))
        return result_id

    def resolve(self, result_id: str) -> dict[str, Any]:
        if result_id not in self.records:
            raise ProtocolError("Unknown result_id: execute and register the requested test first")
        record = self.records[result_id]
        index = record["execution_index"]
        if index >= len(self.tool.executions) or not self.tool.executions[index]["succeeded"]:
            raise ProtocolError("Result registration came from an incomplete or failed execution")
        return dict(record)


def parse_guarded_text(text: str) -> tuple[str, str]:
    """Accept one Python action OR one result-ID submission.

    Parse Python before inspecting marker strings so print('Observation:') is
    valid. Syntax mistakes reach Python and return its actual diagnostic; mixed
    action/observation transcripts remain protocol violations.
    """
    clean = text.strip()
    if clean.startswith("```") and clean.endswith("```"):
        clean = clean[3:-3].strip()
    action = re.search(r"(?m)^Action\s*:\s*([^\n]+)\nAction Input\s*:\s*", clean)
    if action:
        if action.group(1).strip() != "python_repl_ast":
            raise ProtocolError("Only python_repl_ast is allowed")
        code = clean[action.end():].strip()
        if code.startswith("```"):
            fence = re.fullmatch(r"```(?:python)?\s*\n(.*?)\n```\s*", code, re.DOTALL)
            if fence is None:
                raise ProtocolError("Submit one closed Python fence without trailing prose")
            code = fence.group(1)
        if not code.strip():
            raise ProtocolError("Empty Python action")
        try:
            ast.parse(code)
        except SyntaxError as exc:
            if re.search(r"(?:^|\n|\))\s*(?:Observation|Final Answer|Action(?: Input)?)\s*:", code):
                raise ProtocolError("Untrusted Observation or mixed action transcript") from exc
        if "Observation:" in clean[:action.start()] or "Final Answer:" in clean[:action.start()]:
            raise ProtocolError("Mixed observation/final answer and action")
        return "python_repl_ast", code
    if "Observation:" in clean:
        raise ProtocolError("Untrusted Observation transcript: only the environment supplies observations")
    final = re.search(r"(?m)^Final Answer:\s*result_id=(result_[a-f0-9]{32})\s*\Z", clean)
    if final:
        return "finish", final.group(1)
    raise ProtocolError("Use one Action/Action Input or Final Answer: result_id=<registered ID>")


NATIVE_TOOLS = [
    {"type": "function", "function": {"name": "python_repl_ast",
     "description": "Execute Python in a persistent namespace containing the supplied dataframes.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                    "required": ["query"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "finish_test",
     "description": "Submit a result_id printed by record_test_result in a successful Python execution.",
     "parameters": {"type": "object", "properties": {"result_id": {"type": "string"}},
                    "required": ["result_id"], "additionalProperties": False}}},
]


class GuardedExecutor:
    def __init__(self, llm: Any, mode: ExecutionMode, max_iterations: int = 25,
                 max_protocol_retries: int = 2):
        if mode not in ("guarded_text", "native_tools"):
            raise ValueError("Guarded executor requires guarded_text or native_tools")
        self.llm, self.mode = llm, mode
        self.max_iterations, self.max_protocol_retries = max_iterations, max_protocol_retries
        self.tools = [CustomPythonAstREPLTool()]
        self.prompt = CustomPromptTemplate(template=template, tools=[self.tools[0]],
                                           input_variables=["system_prompt", "input", "intermediate_steps"])
        self.tools[0]._context_prompt = self.prompt
        self.protocol_audit: list[dict[str, Any]] = []
        self.result_audit: list[dict[str, Any]] = []
        self.last_result: dict[str, Any] | None = None
        self.runtime_budget = None
        self._kernel = None

    def configure_budget(self, budget):
        self.runtime_budget = budget

    def invoke(self, input: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._invoke(input)
        finally:
            if self._kernel is not None:
                self._kernel.close()
                self._kernel = None

    def _invoke(self, input: dict[str, Any]) -> dict[str, Any]:
        from langchain.schema import AgentAction

        self.last_result = None
        tool = self.tools[0]
        registry = ResultRegistry(tool)
        tool.set_result_recorder(registry.record)
        session_id = uuid4().hex
        system = input["system_prompt"] + RESULT_INSTRUCTION
        if self.mode == "native_tools":
            system += STATISTICAL_INTEGRITY_INSTRUCTION
        if self.runtime_budget is not None:
            from popper.runtime_budget import BUDGET_INSTRUCTION
            from popper.budget_kernel import BudgetKernel
            system += BUDGET_INSTRUCTION
            self._kernel = BudgetKernel(tool, self.runtime_budget, session_id)
        text_system = system + ("\nRespond with one Action: python_repl_ast and Action Input: Python code, "
                                "then wait. To finish use ONLY Final Answer: result_id=<registered ID>.")
        history: list[BaseMessage] = [SystemMessage(content=system), HumanMessage(content=input["input"])]
        steps: list[tuple[Any, str]] = []
        repairs = 0
        bound = self.llm.bind_tools(NATIVE_TOOLS, tool_choice="auto", parallel_tool_calls=False) if self.mode == "native_tools" else None
        for step in range(self.max_iterations):
            if self.runtime_budget is not None:
                self.runtime_budget.check()
                history[0] = SystemMessage(content=system + '\n' + self.runtime_budget.feedback())
            if bound is None:
                prompt = self.prompt.format(system_prompt=text_system, input=input["input"], intermediate_steps=steps)
                message = self.llm.invoke(prompt, stop=["\nObservation:"])
            else:
                # Preserve every assistant message including reasoning_content;
                # only older tool-output views may be bounded deterministically.
                request_history = list(history)
                old_tools = [i for i, m in enumerate(history) if isinstance(m, ToolMessage)][:-2]
                if self.prompt.compact_context:
                    for i in old_tools:
                        m = history[i]
                        view, _ = bound_observation(str(m.content), self.prompt.history_output_max_chars)
                        request_history[i] = m.model_copy(update={"content": view})
                message = bound.invoke(request_history)
            if not isinstance(message, AIMessage) or not isinstance(message.content, str):
                raise ProtocolError("Expected a text-content assistant message")
            event: dict[str, Any] = dict(session_id=session_id, step=step, mode=self.mode,
                                        content=message.content, observation_markers=message.content.count("Observation:"),
                                        accepted=False)
            self.protocol_audit.append(event)
            calls = message.tool_calls
            if bound is not None:
                history.append(message)  # retains reasoning and native call IDs
                event["tool_calls"] = calls
            try:
                if bound is None:
                    name, argument = parse_guarded_text(message.content)
                else:
                    if len(calls) != 1 or message.invalid_tool_calls:
                        raise ProtocolError("Submit exactly one well-formed tool call")
                    call = calls[0]
                    name, args = call["name"], call["args"]
                    key = "query" if name == "python_repl_ast" else "result_id"
                    if name not in ("python_repl_ast", "finish_test") or set(args) != {key} or not isinstance(args[key], str):
                        raise ProtocolError("Unknown tool or invalid arguments")
                    argument = args[key]
                    if "Observation:" in message.content:
                        raise ProtocolError("Untrusted Observation in assistant content")
                event["action"] = name
                if name == "python_repl_ast":
                    observation = (self._kernel.invoke(argument, registry) if self._kernel is not None
                                   else str(tool.invoke(argument)))
                    event["execution_index"] = len(tool.executions) - 1
                    event["accepted"] = True
                    # The repair limit applies to consecutive protocol rejections,
                    # never Python exceptions or unrelated earlier mistakes.
                    repairs = 0
                else:
                    record = registry.resolve(argument)
                    self.last_result = record
                    self.result_audit.append(dict(record, session_id=session_id))
                    event.update(accepted=True, result_id=argument)
                    return {"output": f"Executed test result: p-value = {record['p_value']:.17g}; result_id={argument}",
                            "result": record}
            except ProtocolError as exc:
                event["error"] = str(exc)
                repairs += 1
                observation = (f"Protocol rejected: {exc}. Repair the format or reference an actual registered "
                               "result; preserve the requested test and computed result. Do not invent observations.")
                if bound is not None:
                    # Close every advertised call, including malformed/unknown calls.
                    raw_calls = message.additional_kwargs.get("tool_calls", [])
                    ids = [c.get("id") for c in raw_calls] if raw_calls else [c.get("id") for c in [*calls, *message.invalid_tool_calls]]
                    for call_id in ids:
                        if not call_id:
                            raise ProtocolError("Cannot repair missing tool_call_id") from exc
                        history.append(ToolMessage(content=observation, tool_call_id=call_id))
                    history.append(HumanMessage(content=observation))
                if repairs > self.max_protocol_retries:
                    raise ProtocolError("Protocol repair budget exhausted") from exc
                if bound is None:
                    steps.append((AgentAction(tool="protocol_feedback", tool_input="", log=message.content), observation))
                continue
            if bound is None:
                steps.append((AgentAction(tool=name, tool_input=argument, log=message.content), observation))
            else:
                history.append(ToolMessage(content=observation, tool_call_id=calls[0]["id"]))
        raise ProtocolError("Execution iteration budget exhausted without a registered final result")
