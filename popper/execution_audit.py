"""Offline diagnostics; a numeric match is evidence, not proof of provenance."""
import math
import re


NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
LABELLED_P = re.compile(r"\bp(?:[ _-]?(?:value|val))?\s*[:=]\s*(" + NUMBER + r")", re.I)
BARE_P = re.compile(r"^\s*(" + NUMBER + r")\s*$")


def audit_legacy_pvalue(p_value, executions, replies):
    """Inspect only this attempt's successful outputs; never assistant prose."""
    candidates = []
    for index, execution in enumerate(executions):
        if not execution.get("succeeded"):
            continue
        for line in execution.get("output", "").splitlines():
            numbers = LABELLED_P.findall(line)
            bare = BARE_P.fullmatch(line)
            if bare:
                numbers.append(bare.group(1))
            for token in numbers:
                value = float(token)
                if math.isfinite(value) and 0 < value <= 1:
                    candidates.append(dict(execution_index=index, value=value, line=line))
    matches = [item for item in candidates
               if math.isclose(float(p_value), item["value"], rel_tol=1e-6, abs_tol=0)]
    markers = sum(item.get("observation_markers", 0) for item in replies)
    return dict(p_value=float(p_value), observation_markers=markers,
                flagged=bool(markers or not matches),
                printed_value_match=bool(matches), candidates=candidates,
                status="matched" if matches else "unverified",
                note="Review flag only; matching text does not establish statistical validity or provenance.")
