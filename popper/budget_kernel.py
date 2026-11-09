"""Persistent, terminable Python worker for the optional budgeted executor."""
from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import time
from uuid import uuid4

from popper.runtime_budget import BudgetExceeded, RuntimeBudget, journal


def _serve(connection, tables, root, session_id, executions, audit, output_options):
    from threadpoolctl import threadpool_limits
    from popper.execution import ResultRegistry
    from popper.react_utils import CustomPythonAstREPLTool

    budget = RuntimeBudget(Path(root))
    with threadpool_limits(limits=budget.state()['threads']):
        tool = CustomPythonAstREPLTool()
        tool._set_globals(tables)
        tool.configure_output(Path(root), **output_options)
        tool._executions = executions
        assert tool._output_store is not None
        tool._output_store.audit = audit
        registry = ResultRegistry(tool)
        tool.set_result_recorder(registry.record)
        tool._exec_globals['run_resamples'] = budget.resampler(session_id)
        while True:
            query = connection.recv()
            if query is None:
                return
            observation = tool.invoke(query)
            connection.send(dict(observation=observation, execution=tool.executions[-1],
                                 audit=tool.output_audit, records=registry.records))


class BudgetKernel:
    def __init__(self, tool, budget: RuntimeBudget, session_id: str):
        self.tool, self.budget, self.session_id = tool, budget, session_id
        # Capture input tables before adding result recorders. Only dataframe
        # tables are public inputs; never send API clients or registry closures.
        import pandas as pd
        self.tables = {k: v for k, v in tool._exec_globals.items() if isinstance(v, pd.DataFrame)}
        self.process = None
        self.connection = None

    def start(self):
        import json
        context = mp.get_context('spawn')
        parent, child = context.Pipe()
        store = self.tool._output_store
        if store is not None:
            # A crash after archive publication but before the pipe reply must
            # never cause a subsequent kernel to overwrite that raw archive.
            store.audit = [json.loads(p.read_text(encoding='utf-8'))
                           for p in sorted(self.budget.root.glob('tool_*.json'))]
        options = dict(max_chars=store.max_chars, compact_output=store.compact) if store else {}
        self.process = context.Process(target=_serve, args=(child, self.tables, str(self.budget.root),
            self.session_id, self.tool.executions, self.tool.output_audit, options), daemon=True)
        self.process.start()
        child.close()
        self.connection = parent
        journal(self.budget.events, 'kernel_start', pid=self.process.pid, session=self.session_id)

    def close(self):
        if self.process is not None and self.process.is_alive():
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(self.process.pid), '/T', '/F'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
                               creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                self.process.terminate()
            self.process.join(timeout=5)
        if self.connection is not None:
            self.connection.close()
        self.process = self.connection = None

    def invoke(self, query: str, registry) -> str:
        self.budget.check()
        if self.process is None:
            self.start()
        process, connection = self.process, self.connection
        assert process is not None and connection is not None
        ident = uuid4().hex
        journal(self.budget.events, 'tool_start', tool_id=ident, session=self.session_id,
                execution_index=len(self.tool.executions), code=query, pid=process.pid)
        started = last_progress = time.monotonic()
        offset = self.budget.events.stat().st_size
        connection.send(query)
        try:
            while not connection.poll(.2):
                self.budget.check()
                # Only a completed, checkpointed chunk resets the tool watchdog.
                # Arbitrary stdout or model claims cannot prolong a computation.
                with self.budget.events.open('rb') as stream:
                    stream.seek(offset)
                    lines = stream.readlines()
                    offset = stream.tell()
                import json
                for line in lines:
                    try:
                        event = json.loads(line)
                    except (ValueError, UnicodeError):
                        continue
                    if event.get('event') == 'chunk_end' and event.get('session') == self.session_id:
                        last_progress = time.monotonic()
                if time.monotonic()-last_progress >= self.budget.state()['tool_seconds']:
                    raise TimeoutError('No completed checkpoint within tool time limit')
                if not process.is_alive():
                    raise RuntimeError('Python kernel exited without a result')
            reply = connection.recv()
            self.tool._executions.append(reply['execution'])
            if self.tool._output_store is not None:
                self.tool._output_store.audit = reply['audit']
            registry.records.update(reply['records'])
            journal(self.budget.events, 'tool_end', tool_id=ident,
                    duration_seconds=time.monotonic()-started, succeeded=reply['execution']['succeeded'])
            return reply['observation']
        except (TimeoutError, BudgetExceeded, EOFError, OSError, RuntimeError) as exc:
            self.close()
            self.tool._executions.append(dict(code=query, output=str(exc), succeeded=False))
            journal(self.budget.events, 'tool_interrupted', tool_id=ident,
                    duration_seconds=time.monotonic()-started, error_type=type(exc).__name__, error=str(exc))
            if isinstance(exc, BudgetExceeded):
                raise
            return (f'Tool interrupted: {exc}. Python variables have been reset; original data remain. '
                    'Completed resampling checkpoints remain on disk. Recreate the same function and '
                    'use resume=True with the unchanged plan. No result from this execution is valid.')
