"""Offline graph tests for selection accounting and reporting failures."""
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from popper.agent import SequentialFalsificationTest, falsification_test_react_agent
from popper.execution_audit import audit_legacy_pvalue
from popper.react_utils import CustomPythonAstREPLTool


@pytest.mark.parametrize('pvalue,limit,reason', [(.04, 1, 'max_tests'), (.0001, 2, 'rejected')])
@pytest.mark.parametrize('reporting_stage', ['summary', 'final_parser'])
def test_failed_tests_are_audited_and_terminal_decision_precedes_reporting(tmp_path, pvalue, limit, reason, reporting_stage):
    engine = SequentialFalsificationTest.__new__(SequentialFalsificationTest)
    engine.llm_use, engine.port, engine.api_key, engine.base_url = 'offline', None, 'offline', None
    engine.res, engine.res_stat, engine.num_of_tests = False, None, 0
    outcomes = iter([dict(status='Failed test', failure_reason='syntax', attempts=[]),
                     dict(status='success', p_val=pvalue, attempts=[])])
    coder = SimpleNamespace(go=lambda *args: next(outcomes))
    designer = SimpleNamespace(failed_tests=[], go=lambda *args: 'test proposal',
                               add_to_existing_tests=lambda *args: None)
    designer.add_to_failed_tests = designer.failed_tests.append
    with patch('popper.agent.falsification_test_react_agent', return_value=coder), \
         patch('popper.agent.falsification_test_proposal_agent', return_value=designer):
        engine.configure(SimpleNamespace(table_dict={}, data_desc=''), use_react_agent=True,
                         max_num_of_tests=limit, alpha=.05)
    engine.decision_path = tmp_path / 'decision.json'
    def fail_summary():
        saved = json.loads(engine.decision_path.read_text())
        assert saved['completed'] and saved['rejected'] == (reason == 'rejected')
        assert saved['p_values'] == [pvalue]
        if reporting_stage == 'summary':
            raise RuntimeError('summarizer failed')
        return {'messages': [('assistant', 'summary complete')]}
    engine.summarize = fail_summary
    engine.output_parser = SimpleNamespace(invoke=Mock(side_effect=RuntimeError('final parser failed')))
    with pytest.raises(RuntimeError, match='failed'):
        engine.go('hypothesis')
    assert engine.tracked_stat == [pvalue]
    assert [e['status'] for e in engine.test_audit] == ['failed', 'success']
    assert engine.test_audit[0]['failure_reason'] == 'syntax'
    assert engine.decision_snapshot['stop_reason'] == reason


def test_parser_failure_never_reexecutes_the_analysis():
    tool = CustomPythonAstREPLTool()
    react = SimpleNamespace(agent=SimpleNamespace(tools=[tool]), protocol_audit=[], execution_mode='legacy',
                            generate=Mock(return_value='p=0.1'))
    coder = falsification_test_react_agent.__new__(falsification_test_react_agent)
    coder.agent, coder.data_loader, coder.domain, coder.max_retry = react, None, 'statistics', 3
    coder.pvalue_parser = SimpleNamespace(invoke=Mock(side_effect=ValueError('parse exhausted')))
    proposal = ('Falsification Test name: test\nFalsification Test description: description\n'
                'Falsification Test Null sub-hypothesis: null\nFalsification Test Alternate sub-hypothesis: alternative')
    result = coder.go(proposal)
    assert result['status'] == 'Failed test'
    assert result['failure_reason'] == 'pvalue_parse_error'
    assert react.generate.call_count == 1


def test_legacy_audit_does_not_use_assistant_or_failed_execution_numbers():
    replies = [dict(content='Observation: p_value=.001\nFinal Answer: p=.001', observation_markers=1)]
    failed = [dict(output='p_value=.001', succeeded=False)]
    result = audit_legacy_pvalue(.001, failed, replies)
    assert result['flagged'] and not result['printed_value_match']
    assert result['observation_markers'] == 1
    actual = [dict(output='ordinary data\np_value=0.001\n', succeeded=True)]
    result = audit_legacy_pvalue(.001, actual, [])
    assert result['printed_value_match'] and not result['flagged']
    assert not audit_legacy_pvalue(.002, actual, [])['printed_value_match']
