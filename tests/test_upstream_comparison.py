"""Differential checks against unmodified excerpts of pinned upstream source.

The fixture records the original commit and full-source hashes. No network,
checkout, hand-written reference algorithm or live model is used by these tests.
The known executor differences are intentional, not an upstream-equivalence claim.
"""
import contextlib
import io
import json
import logging
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pytest

from popper.agent import e_value_kappa_calibrator
from popper.react_utils import CustomPromptTemplate, CustomPythonAstREPLTool, template


REFERENCE = json.loads((Path(__file__).parent / 'fixtures/upstream_5961186.json').read_text())
namespace = dict(np=np, re=re, io=io, contextlib=contextlib, logging=logging)
for file in REFERENCE['files'].values():
    for name, source in file['extracted'].items():
        if name != 'template':
            exec(compile(source, 'upstream@' + REFERENCE['commit'], 'exec'), namespace)


def upstream_run(state, code):
    root = logging.getLogger()
    handlers = list(root.handlers)
    root.handlers = [logging.StreamHandler(io.StringIO())]
    try:
        return namespace['_run'](state, code)
    finally:
        root.handlers = handlers


@pytest.mark.parametrize('pvalues', [[1.], [.04], [1., .0004], [.0004, 1.], [.02, .03, .01]])
def test_calibrator_matches_pinned_upstream(pvalues):
    assert e_value_kappa_calibrator(pvalues, alpha=.05) == namespace['e_value_kappa_calibrator'](pvalues, alpha=.05)


def test_uncompacted_empty_history_prompt_matches_upstream():
    original = REFERENCE['files']['popper/react_utils.py']['extracted']['template']
    prompt = CustomPromptTemplate(template=template, tools=[], compact_context=False,
                                  input_variables=['system_prompt', 'input', 'intermediate_steps'])
    expected = original.format(system_prompt='system', input='test', tools='', tool_names='', agent_scratchpad='')
    assert prompt.format(system_prompt='system', input='test', intermediate_steps=[]) == expected


@pytest.mark.parametrize('code,old,new', [
    ('print(42)', '42\n', '42\n'),
    ('for i in range(3):\n    v = i * 10\n    print(v)', '20\n', '0\n10\n20\n'),
    ("print('a',\n 'b')", "'(' was never closed (<string>, line 1)", 'a b\n'),
])
def test_fixed_snippets_compare_both_executors(code, old, new):
    reference = upstream_run(SimpleNamespace(_exec_globals={}), code)
    actual = CustomPythonAstREPLTool()._run(code)
    assert reference == old
    assert actual == new


def test_last_assignment_and_exception_are_documented_differences():
    old, new = SimpleNamespace(_exec_globals={}), CustomPythonAstREPLTool()
    upstream_run(old, 'value = 42')
    new._run('value = 42')
    assert 'value' not in old._exec_globals
    assert new._run('value').strip() == '42'
    code = "print('before')\nraise KeyError('missing')\npass"
    assert upstream_run(old, code) == "'missing'"
    output = new._run(code)
    assert 'before' in output and 'Traceback' in output and 'KeyError' in output


def test_single_statement_loop_is_an_indentation_error_upstream():
    code = 'for v in [0, 10, 20]:\n    print(v)'
    assert 'expected an indented block' in upstream_run(SimpleNamespace(_exec_globals={}), code)
    assert CustomPythonAstREPLTool()._run(code) == '0\n10\n20\n'
