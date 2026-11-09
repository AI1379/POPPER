import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage
from popper.execution import GuardedExecutor
from popper.runtime_budget import BudgetExceeded, RuntimeBudget


class Model:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.inputs = []

    def bind_tools(self, *args, **kwargs):
        return self

    def invoke(self, history):
        self.inputs.append(history)
        reply = next(self.replies)
        return reply(history) if callable(reply) else reply


def action(code):
    return AIMessage(content='', additional_kwargs={'reasoning_content': 'keep thinking'},
        tool_calls=[dict(name='python_repl_ast', args={'query': code}, id='test_call', type='tool_call')])


def finish(history):
    import re
    last = next(str(m.content) for m in reversed(history) if isinstance(m, ToolMessage))
    result_id = re.search(r'result_[a-f0-9]{32}', last).group()
    return AIMessage(content='', tool_calls=[dict(name='finish_test', args={'result_id': result_id},
                                                  id='finish', type='tool_call')])


class BudgetTests(unittest.TestCase):
    def test_worker_without_compaction_preserves_long_output_and_p1_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            budget = RuntimeBudget.create(Path(directory), initial=100, hard=300, tool_seconds=20)
            model = Model([action("print('x' * 20000)\nrecord_test_result(1., fallback_reason='not identifiable')"), finish])
            executor = GuardedExecutor(model, 'native_tools')
            executor.configure_budget(budget)
            result = executor.invoke(dict(system_prompt='', input=''))
            output = next(m.content for m in model.inputs[1] if isinstance(m, ToolMessage))
            self.assertGreater(len(output), 20000)
            self.assertNotIn('characters omitted', output)
            self.assertEqual(result['result']['fallback_reason'], 'not identifiable')

    def test_runtime_extension_and_hard_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            b = RuntimeBudget.create(Path(directory), initial=200, hard=400)
            self.assertTrue(b.reserve(100, evidence={'measured_seconds': 1}))
            self.assertGreater(b.remaining(), 260)
            self.assertFalse(b.reserve(400, evidence={'measured_seconds': 4}))
            self.assertLessEqual(b.state()['deadline'], b.state()['hard_deadline'])

    def test_checkpoints_resume_same_draws_and_reject_changed_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            b = RuntimeBudget.create(Path(directory), initial=200, hard=400)
            run = b.resampler('test')
            seen = []
            def draw(i):
                seen.append(i)
                return i*i
            with patch.object(b, 'reserve', return_value=False):
                with self.assertRaises(BudgetExceeded):
                    run('boot', 6, draw, pilot_size=2, plan='fixed')
            self.assertEqual(seen, [0, 1])
            values = run('boot', 6, draw, pilot_size=2, plan='fixed', resume=True)
            self.assertEqual(values, [0, 1, 4, 9, 16, 25])
            self.assertEqual(seen, list(range(6)))
            with self.assertRaises(ValueError):
                run('boot', 7, draw, pilot_size=2, plan='fixed', resume=True)

    def test_process_keeps_namespace_and_registration_and_reasoning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            b = RuntimeBudget.create(root, initial=100, hard=300, tool_seconds=20)
            m = Model([action('x=4'), action('record_test_result(1/x, n=10)'), finish])
            e = GuardedExecutor(m, 'native_tools')
            e.tools[0].configure_output(root)
            e.configure_budget(b)
            result = e.invoke(dict(system_prompt='', input='test'))
            self.assertEqual(result['result']['p_value'], .25)
            self.assertEqual(len(e.tools[0].executions), 2)
            self.assertTrue(any(isinstance(x, AIMessage) and x.additional_kwargs.get('reasoning_content')
                                == 'keep thinking' for x in m.inputs[-1]))
            events = [json.loads(line) for line in b.events.read_text().splitlines()]
            self.assertEqual(sum(x['event']=='tool_start' for x in events), 2)
            self.assertEqual(sum(x['event']=='tool_end' for x in events), 2)

    def test_stalled_tool_is_killed_and_cannot_register_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            b = RuntimeBudget.create(root, initial=100, hard=300, tool_seconds=5)
            m = Model([action('import time\nrecord_test_result(.01)\ntime.sleep(60)'),
                       action('record_test_result(.5)'), finish])
            e = GuardedExecutor(m, 'native_tools')
            e.tools[0].configure_output(root)
            e.configure_budget(b)
            result = e.invoke(dict(system_prompt='', input='test'))
            self.assertEqual(result['result']['p_value'], .5)
            self.assertFalse(e.tools[0].executions[0]['succeeded'])
            self.assertIn('tool_interrupted', b.events.read_text())
            self.assertIn('time.sleep(60)', b.events.read_text())

    def test_request_start_finish_error_and_limits_without_network(self):
        from popper.llm.compatible_model import CompatibleChatOpenAI
        from langchain_openai import ChatOpenAI
        from langchain_core.outputs import ChatResult, ChatGeneration
        with tempfile.TemporaryDirectory() as directory:
            b = RuntimeBudget.create(Path(directory), max_requests=2)
            m = CompatibleChatOpenAI(api_key='offline-test', model='unused', runtime_budget_path=directory)
            result = ChatResult(generations=[ChatGeneration(message=AIMessage(content='ok'))])
            with patch.object(ChatOpenAI, '_generate', return_value=result) as api:
                m._generate([])
                self.assertLessEqual(api.call_args.kwargs['timeout'], 240)
            with patch.object(ChatOpenAI, '_generate', side_effect=TimeoutError):
                with self.assertRaises(TimeoutError):
                    m._generate([])
            with self.assertRaises(BudgetExceeded):
                m._generate([])
            events = b.events.read_text()
            self.assertIn('request_start', events)
            self.assertIn('request_end', events)
            self.assertIn('request_error', events)

    def test_chunks_can_exceed_single_tool_limit_with_durable_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            b = RuntimeBudget.create(root, initial=100, hard=300, tool_seconds=5)
            code = '''import time
def draw(i):
    time.sleep(.2)
    return i
values = run_resamples('chunked', 30, draw, pilot_size=1, chunk_size=2,
                       plan='fixed 30 draws, deterministic smoke check')
assert len(values)==30
record_test_result(.5)
'''
            e = GuardedExecutor(Model([action(code), finish]), 'native_tools')
            e.tools[0].configure_output(root)
            e.configure_budget(b)
            result = e.invoke(dict(system_prompt='',input=''))
            self.assertEqual(result['result']['p_value'], .5)
            events = [json.loads(line) for line in b.events.read_text().splitlines()]
            self.assertGreater(next(x['duration_seconds'] for x in events if x['event']=='tool_end'), 5)
            self.assertGreater(sum(x['event']=='chunk_end' for x in events), 1)


if __name__ == '__main__':
    unittest.main()
