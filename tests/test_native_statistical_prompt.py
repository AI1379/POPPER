"""Regression checks for policies that were lost in the native message path."""
import pytest
from popper.execution import GuardedExecutor
from popper.prompt_utils import get_react_coding_agent_system_prompt

class CaptureStop(Exception):
    pass

class CaptureModel:
    def __init__(self):
        self.request = None
    def bind_tools(self, *args, **kwargs):
        return self
    def invoke(self, request, *args, **kwargs):
        self.request = request
        raise CaptureStop()

@pytest.mark.parametrize('mode', ['native_tools', 'guarded_text'])
def test_actual_execution_request_retains_statistical_requirements(mode):
    model = CaptureModel()
    executor = GuardedExecutor(model, mode, max_iterations=1)
    with pytest.raises(CaptureStop):
        executor.invoke({'system_prompt': get_react_coding_agent_system_prompt(domain='statistics'),
                         'input': 'Evaluate the specified null; audit sentinel.'})
    text = model.request if isinstance(model.request, str) else '\n'.join(str(m.content) for m in model.request)
    for requirement in [
        'return a p-value of 1.00e+00',
        'make sure the sample sizes and the statistical test is valid',
        'aggregate them in a meaningful and rigorous way',
        'H1 is also incorrect',
        'DO NOT perform p-hacking',
        'report a p-value EXACTLY AS IT IS',
    ]:
        assert requirement in text
    if mode == 'native_tools':
        assert 'submit its ID with finish_test' in text
        assert 'Body Length Evolution' not in text
    assert executor.tools[0].executions == []
