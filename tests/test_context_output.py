import gzip
import io
import json
import logging
from pathlib import Path
import tempfile
import unittest

from langchain.schema import AgentAction

from popper.react_utils import CustomPromptTemplate, CustomPythonAstREPLTool, template
from popper.tool_output import bound_observation, deduplicate_warnings


class ContextOutputTests(unittest.TestCase):
    def test_repeated_warnings_keep_counts_and_exact_statistical_results(self):
        with tempfile.TemporaryDirectory() as directory:
            tool = CustomPythonAstREPLTool()
            tool.configure_output(Path(directory), 12000)
            output = tool._run("""import warnings
warnings.simplefilter('always')
for i in range(5017):
    warnings.warn('solver setting is deprecated', FutureWarning)
for i in range(65):
    warnings.warn('Maximum iterations reached; fit may not have converged', RuntimeWarning)
print('B=200, p_value=4.975124378109453e-03')
print('T_obs=0.02013325968718207')
""")
            self.assertLess(len(output), 2000)
            self.assertIn('5017 times', output)
            self.assertIn('65 times', output)
            self.assertIn('fit may not have converged', output)
            self.assertIn('p_value=4.975124378109453e-03', output)
            self.assertIn('T_obs=0.02013325968718207', output)
            audit = tool.output_audit[0]
            self.assertEqual(sum(w['count'] for w in audit['warnings']), 5082)
            with gzip.open(audit['raw_path'], 'rt', encoding='utf-8') as stream:
                raw = stream.read()
            self.assertEqual(raw.count('FutureWarning:'), 5017)
            self.assertEqual(raw.count('RuntimeWarning:'), 65)
            self.assertIn('p_value=4.975124378109453e-03', raw)

    def test_multiline_diagnostics_are_retained_verbatim(self):
        with tempfile.TemporaryDirectory() as directory:
            tool = CustomPythonAstREPLTool()
            tool.configure_output(Path(directory))
            output = tool._run("""import warnings
warnings.simplefilter('always')
for i in range(3):
    warnings.warn('Optimizer failed.\\nstatus=1\\nIncrease max_iter or review scaling.', RuntimeWarning)
print('p_value=2.37e-08')
""")
            self.assertIn('Optimizer failed.\nstatus=1\nIncrease max_iter or review scaling.', output)
            self.assertIn('3 times', output)
            self.assertIn('p_value=2.37e-08', output)

    def test_large_output_can_be_read_after_namespace_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            tool = CustomPythonAstREPLTool()
            tool.configure_output(Path(directory), 1000)
            output = tool._run("""for i in range(1000):
    print(f'row {i}: ordinary data')
    if i == 500:
        print('p_value=9.876543210987654e-03')
print('n_test=200')
""")
            self.assertIn('characters omitted', output)
            self.assertIn('p_value=9.876543210987654e-03', output)
            self.assertIn('n_test=200', output)
            tool._set_globals({})
            reader = tool._exec_globals['read_tool_output']
            raw = reader('tool_0001', start=0, length=12000, raw=True)
            self.assertIn('row 500', raw)
            with self.assertRaises(ValueError):
                reader('../.env')
            with self.assertRaises(ValueError):
                reader('tool_0001', length=100000)

    def test_compaction_does_not_change_computations_or_persistent_variables(self):
        code = """import numpy as np
rng = np.random.default_rng(727621)
values = rng.normal(size=400)
result = float(values.mean())
print('statistic:', result)
result
"""
        plain = CustomPythonAstREPLTool()
        original = plain._run(code)
        with tempfile.TemporaryDirectory() as directory:
            compact = CustomPythonAstREPLTool()
            compact.configure_output(Path(directory))
            compressed = compact._run(code)
            self.assertEqual(original, compressed)
            self.assertEqual(plain._exec_globals['result'], compact._exec_globals['result'])
            compact._run('last_assigned = 42')
            self.assertEqual(compact._run('last_assigned').strip(), '42')

    def test_errors_are_visible_and_logger_stream_is_restored(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        logging.getLogger().addHandler(handler)
        try:
            with tempfile.TemporaryDirectory() as directory:
                tool = CustomPythonAstREPLTool()
                tool.configure_output(Path(directory))
                output = tool._run("raise ValueError('bad input data')")
                self.assertIn('Traceback', output)
                self.assertIn('ValueError: bad input data', output)
                self.assertIs(handler.stream, stream)
        finally:
            logging.getLogger().removeHandler(handler)

    def test_context_retains_actions_recent_output_and_numerical_diagnostics(self):
        prompt = CustomPromptTemplate(template=template, tools=[], compact_context=True,
                                      history_output_max_chars=1000,
                                      input_variables=['system_prompt', 'input', 'intermediate_steps'])
        observations = []
        for index in range(4):
            text = ('ordinary row\n' * 1500 + '\np_value=0.004975124378109453\n'
                    'IterationLimitWarning: failed fits count=65\n' + 'ordinary row\n' * 1500
                    + f'[Full output: tool_{index + 1:04d}]')
            observations.append((AgentAction(tool='python_repl_ast', tool_input=f'code_{index}',
                                              log=f'Action Input: exact_code_{index}'), text))
        formatted = prompt.format(system_prompt='Analyze carefully', input='target hypothesis',
                                  intermediate_steps=observations)
        for index in range(4):
            self.assertIn(f'exact_code_{index}', formatted)
            self.assertIn(f'tool_{index + 1:04d}', formatted)
        self.assertIn('p_value=0.004975124378109453', formatted)
        self.assertIn('failed fits count=65', formatted)
        self.assertNotIn('Body Length Evolution', formatted)
        self.assertIn('DO NOT perform p-hacking', formatted)
        self.assertIn(observations[-1][1], formatted)
        self.assertLess(prompt.context_audit[-1]['delivered_observation_chars'],
                        prompt.context_audit[-1]['observation_chars'])

    def test_large_single_line_and_historical_warning_replay(self):
        view, truncated = bound_observation('x' * 30000, 1000)
        self.assertTrue(truncated)
        self.assertLess(len(view), 2000)
        # A large first line must not move the real tail outside the selected view.
        trailing = '\n'.join(f'tail row {index}' for index in range(100))
        with_tail, _ = bound_observation('x' * 30000 + '\n' + trailing, 1000)
        self.assertIn('tail row 99', with_tail)
        self.assertLess(len(with_tail), 2000)
        block = 'file.py:10: FutureWarning: deprecated setting\n  warnings.warn(...)\n'
        replay, warnings = deduplicate_warnings(block * 5015 + 'p_value=1.996008e-03\n')
        self.assertLess(len(replay), 500)
        self.assertIn('5015 times', replay)
        self.assertIn('p_value=1.996008e-03', replay)
        self.assertEqual(warnings[0]['count'], 5015)


if __name__ == '__main__':
    unittest.main()
