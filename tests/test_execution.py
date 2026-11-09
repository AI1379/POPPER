import unittest
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain.schema import AgentFinish

from popper.execution import GuardedExecutor, ProtocolError, ResultRegistry, parse_guarded_text
from popper.react_utils import CustomOutputParser, CustomPythonAstREPLTool


FABRICATED = """Thought: I will run the test.
Action: python_repl_ast
Action Input: print(df.shape)Observation:
(1000, 3)
Action: python_repl_ast
Action Input: print(p)Observation:
p_value=5.216606912460331e-16
Final Answer: p=5.216606912460331e-16
"""


class FakeModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.inputs: list[Any] = []

    def bind_tools(self, *args, **kwargs):
        assert kwargs['tool_choice'] == 'auto'
        return self

    def invoke(self, value, **kwargs):
        self.inputs.append(list(value) if isinstance(value, list) else value)
        reply = next(self.replies)
        return reply(value) if callable(reply) else reply


def action(code, call_id='python_1', reason='retained reasoning'):
    return AIMessage(content='', additional_kwargs={'reasoning_content': reason},
                     tool_calls=[{'name': 'python_repl_ast', 'args': {'query': code}, 'id': call_id, 'type': 'tool_call'}])


def finish_from_history(history):
    import re
    observation = next(m.content for m in reversed(history) if isinstance(m, ToolMessage))
    match = re.search(r'result_[a-f0-9]{32}', str(observation))
    assert match
    return AIMessage(content='', tool_calls=[{'name': 'finish_test', 'args': {'result_id': match.group()},
                                              'id': 'finish_1', 'type': 'tool_call'}])


class ExecutionTests(unittest.TestCase):
    def test_historical_failure_shape_is_accepted_by_legacy_and_rejected_by_guard(self):
        self.assertIsInstance(CustomOutputParser().parse(FABRICATED), AgentFinish)
        with self.assertRaisesRegex(ProtocolError, 'Observation'):
            parse_guarded_text(FABRICATED)

    def test_markers_inside_valid_python_strings_are_not_transcripts(self):
        self.assertEqual(parse_guarded_text('Action: python_repl_ast\nAction Input: print("Observation:")'),
                         ('python_repl_ast', 'print("Observation:")'))
        with self.assertRaises(ProtocolError):
            parse_guarded_text('I think p is 0.01')

    def test_registry_rejects_missing_stale_and_failed_execution(self):
        tool = CustomPythonAstREPLTool()
        registry = ResultRegistry(tool)
        tool.set_result_recorder(registry.record)
        tool.invoke("result = record_test_result(0.2)\nraise ValueError('after registration')")
        result_id = next(iter(registry.records))
        with self.assertRaisesRegex(ProtocolError, 'failed execution'):
            registry.resolve(result_id)
        with self.assertRaisesRegex(ProtocolError, 'Unknown'):
            ResultRegistry(tool).resolve(result_id)
        for p in [0, -1, 2, float('nan'), float('inf'), True]:
            with self.assertRaises(ProtocolError):
                registry.record(p)

    def test_native_tools_execute_and_preserve_reasoning_between_rounds(self):
        model = FakeModel([action('record_test_result(0.125, n=400)'), finish_from_history])
        executor = GuardedExecutor(model, 'native_tools')
        output = executor.invoke({'system_prompt': 'test', 'input': 'test'})
        self.assertEqual(output['result']['p_value'], 0.125)
        self.assertEqual(len(executor.tools[0].executions), 1)
        prior = next(m for m in model.inputs[1] if isinstance(m, AIMessage))
        self.assertEqual(prior.additional_kwargs['reasoning_content'], 'retained reasoning')
        self.assertEqual(next(m for m in model.inputs[1] if isinstance(m, ToolMessage)).tool_call_id, 'python_1')

    def test_native_fake_transcript_is_rejected_then_repaired(self):
        model = FakeModel([AIMessage(content=FABRICATED, additional_kwargs={'reasoning_content': 'keep me'}),
                           action('record_test_result(0.4)'), finish_from_history])
        executor = GuardedExecutor(model, 'native_tools')
        self.assertEqual(executor.invoke({'system_prompt': '', 'input': ''})['result']['p_value'], 0.4)
        self.assertFalse(executor.protocol_audit[0]['accepted'])
        self.assertEqual(len(executor.tools[0].executions), 1)
        prior = next(m for m in model.inputs[1] if isinstance(m, AIMessage))
        self.assertEqual(prior.additional_kwargs['reasoning_content'], 'keep me')

    def test_multiple_calls_are_closed_and_not_executed(self):
        first = action('raise AssertionError("must never execute")')
        first.tool_calls.append({'name': 'unknown', 'args': {}, 'id': 'bad_2', 'type': 'tool_call'})
        model = FakeModel([first, action('record_test_result(0.8)'), finish_from_history])
        executor = GuardedExecutor(model, 'native_tools')
        executor.invoke({'system_prompt': '', 'input': ''})
        closed = [m.tool_call_id for m in model.inputs[1] if isinstance(m, ToolMessage)]
        self.assertEqual(closed, ['python_1', 'bad_2'])
        self.assertEqual(len(executor.tools[0].executions), 1)

    def test_text_gate_executes_and_resolves_host_result(self):
        def final(prompt):
            import re
            ids = re.findall(r'result_[a-f0-9]{32}', prompt)
            return AIMessage(content='Final Answer: result_id=' + ids[-1])
        model = FakeModel([AIMessage(content=FABRICATED),
                           AIMessage(content='Action: python_repl_ast\nAction Input: record_test_result(0.6)'), final])
        executor = GuardedExecutor(model, 'guarded_text')
        executor.prompt.compact_context = True
        self.assertEqual(executor.invoke({'system_prompt': '', 'input': ''})['result']['p_value'], 0.6)
        self.assertEqual(len(executor.tools[0].executions), 1)
        self.assertFalse(executor.protocol_audit[0]['accepted'])

    def test_repair_budget_never_defaults_to_a_result(self):
        model = FakeModel([AIMessage(content='Final Answer: p=0.001')] * 3)
        executor = GuardedExecutor(model, 'guarded_text')
        with self.assertRaisesRegex(ProtocolError, 'budget exhausted'):
            executor.invoke({'system_prompt': '', 'input': ''})
        self.assertIsNone(executor.last_result)
        self.assertEqual(executor.tools[0].executions, [])

    def test_scattered_python_typos_are_observations_not_protocol_rejections(self):
        def final(prompt):
            import re
            return AIMessage(content='Final Answer: result_id=' + re.findall(r'result_[a-f0-9]{32}', prompt)[-1])
        replies = []
        for index in range(3):
            replies.extend([AIMessage(content='Action: python_repl_ast\nAction Input: x ='),
                            AIMessage(content=f'Action: python_repl_ast\nAction Input: x = {index}')])
        replies.extend([AIMessage(content='Action: python_repl_ast\nAction Input: record_test_result(0.4)'), final])
        model = FakeModel(replies)
        executor = GuardedExecutor(model, 'guarded_text', max_protocol_retries=0)
        result = executor.invoke(dict(system_prompt='', input=''))
        self.assertEqual(result['result']['p_value'], .4)
        self.assertIn('SyntaxError', model.inputs[1])
        self.assertEqual(sum(not e['succeeded'] for e in executor.tools[0].executions), 3)
        self.assertTrue(all(e['accepted'] for e in executor.protocol_audit))

    def test_protocol_repair_limit_is_consecutive(self):
        replies = []
        for _ in range(3):
            replies.extend([AIMessage(content='invented answer'), action('x = 1')])
        replies.extend([action('record_test_result(.5)'), finish_from_history])
        executor = GuardedExecutor(FakeModel(replies), 'native_tools', max_protocol_retries=1)
        self.assertEqual(executor.invoke(dict(system_prompt='', input=''))['result']['p_value'], .5)

    def test_explicit_p1_fallback_is_registered_with_reason(self):
        executor = GuardedExecutor(FakeModel([
            action('record_test_result(1.0, fallback_reason="No valid test for this estimand")'),
            finish_from_history]), 'native_tools')
        result = executor.invoke(dict(system_prompt='', input=''))['result']
        self.assertEqual(result['p_value'], 1)
        self.assertEqual(result['fallback_reason'], 'No valid test for this estimand')
        registry = ResultRegistry(CustomPythonAstREPLTool())
        for p, reason in [(.01, 'not feasible'), (1, ''), (1, False)]:
            with self.subTest(p=p, reason=reason), self.assertRaises(ProtocolError):
                registry.record(p, fallback_reason=reason)

    def test_trailing_prose_remains_a_protocol_violation(self):
        with self.assertRaises(ProtocolError):
            parse_guarded_text('Action: python_repl_ast\nAction Input: ```python\nprint(1)\n```\nDone.')


if __name__ == '__main__':
    unittest.main()
