"""
sentinel.shell.kernel_checks
===============================

Runs the mechanical safety checks in KERNEL/GPU mode, where the frames
live in a separate process and never come back to the shell.

Why (audit finding, 2026-08-06)
---------------------------------
``_tool_run_python``'s exec branch fires six checks after every cell
(data-comparability, leakage, output-grain, numeric-stability,
ambiguous-column, memory). ``_run_python_kernel`` fired only two
(mutation-delta, memory) because every other check reads
``self._ns`` -- which is empty in kernel mode by construction. Since
kernel mode became the DEFAULT whenever a GPU is present, the more
successfully GPU mode was adopted, the LESS of the pilot-feedback safety
net actually ran. The ambiguous-column check is the only one that
mechanically drives ``ask_user``, so "ask instead of assuming" was
weakest exactly on the GPU path.

How it works
--------------
Rather than pull frames back to the shell (which would defeat the point
of GPU mode and cost a full round trip at crore scale), a small snippet
runs INSIDE the kernel and returns a compact JSON summary -- column
names, per-column inf counts, date bounds, label-duplicate flags, and
(only when the cell looks like an export) duplicate entity-key counts.
The existing check logic then runs against that summary in-process,
reusing the SAME hint lists and guidance strings as the exec-mode
modules so the two paths cannot drift into giving different advice.

Cost posture: every aggregate in the snippet is a single vectorised pass
the GPU does cheaply, and the expensive one (``nunique``-style duplicate
counting) is gated behind an export, exactly as in exec mode. Only
scalars cross the process boundary, never data.
"""

from __future__ import annotations

import json

from sentinel.shell.ambiguity_checks import _guidance as _ambiguity_guidance
from sentinel.shell.ambiguity_checks import _operational_hits
from sentinel.shell.data_checks import _LABEL_HINTS
from sentinel.shell.grain_checks import _guidance as _grain_guidance
from sentinel.shell.grain_checks import (_ENTITY_KEY_HINTS, _looks_like_export,
                                         exported_frame_names)
from sentinel.shell.leakage_checks import _lexicon_hits
from sentinel.shell.numeric_checks import _guidance as _numeric_guidance

_PAYLOAD_START = "__SENTINEL_CHECKS__"
_PAYLOAD_END = "__END_CHECKS__"

# Executed inside the kernel. `_sentinel_want_dupes` (True only when the
# cell looked like an export) and `_sentinel_cell_code` (the cell's source)
# are set by the caller immediately before this runs.
#
# J-01: the expensive per-column scans (inf, column==label) used to run on
# EVERY frame after EVERY cell -- many full passes over an 8-crore frame per
# cell. `_sentinel_check_fp` remembers each frame's (rows, columns) and the
# scan results; a frame is re-scanned only when its shape changed or the
# cell assigned into it. The cheap parts (column names) always refresh.
# J-03: label detection uses the same hints as exec mode (data_checks), so
# a `flag`/`is_mule` label gets the derived-column check on the GPU path.
CHECK_SNIPPET = r'''
try:
    import json as _json, re as _re
    _out = {}
    try:
        _sentinel_check_fp
    except NameError:
        _sentinel_check_fp = {}
    try:
        _sentinel_export_names
    except NameError:
        _sentinel_export_names = set()
    def _scalar(_x):
        # dask returns lazy scalars; int() on one raises. Materialise only
        # when we must, and give up quietly rather than breaking the check.
        try:
            if hasattr(_x, 'compute'):
                _x = _x.compute()
            return int(_x)
        except Exception:
            return None
    def _assigned(_name):
        _c = globals().get('_sentinel_cell_code') or ''
        try:
            _n = _re.escape(_name)
            return _re.search(
                r"(?:\b%s\s*(?:\[[^\n]*?\]\s*)?=(?!=))|(?:\b%s\.(?:loc|iloc|at|iat)\b)"
                r"|(?:\b%s\.\w+\([^\n]*inplace\s*=\s*True)" % (_n, _n, _n), _c) is not None
        except Exception:
            return True
    _ent_hints = ''' + repr(list(_ENTITY_KEY_HINTS)) + r'''
    _lab_hints = ''' + repr(list(_LABEL_HINTS)) + r'''
    for _name in list(dir()):
        if _name.startswith('_'):
            continue
        try:
            _v = eval(_name)
        except Exception:
            continue
        if not (hasattr(_v, 'columns') and hasattr(_v, 'dtypes')
                and hasattr(_v, '__len__')):
            continue
        try:
            _n = _scalar(len(_v))
        except Exception:
            continue
        if _n == 0:
            continue
        _cols = [str(_c) for _c in _v.columns]
        _fp = (_n, tuple(_cols))
        _prev = _sentinel_check_fp.get(_name)
        if _prev is not None and _prev[0] == _fp and not _assigned(_name):
            _info = dict(_prev[1])
            _info['dup_keys'] = {}
        else:
            _info = {'n_rows': _n, 'columns': _cols, 'inf_cols': {},
                     'label_dupes': [], 'dup_keys': {}}
            # -- inf / -inf per numeric column (unguarded divide-by-zero) --
            for _c in _v.columns:
                try:
                    _s = _v[_c]
                    if _s.dtype.kind not in 'fc':
                        continue
                    _ni = _scalar(((_s == float('inf')) | (_s == float('-inf'))).sum())
                    if _ni:
                        _info['inf_cols'][str(_c)] = _ni
                except Exception:
                    continue
            # -- a column whose values are identical to the label column --
            # Whole-token match only: substring matching on hints like "y"
            # hit payer_account_number, city, currency, ... and paid a
            # crore-scale nunique() on each before reaching the label.
            # Exact names first, float columns never, at most 3 probes.
            _cands = []
            for _c in _cols:
                _lc = _c.lower()
                _toks = [t for t in _re.split(r'[^a-z0-9]+', _lc) if t]
                _exact = _lc in _lab_hints
                if not (_exact or any(t in _lab_hints for t in _toks)
                        or any('_' in _h and _h in _lc for _h in _lab_hints)):
                    continue
                try:
                    if _v[_c].dtype.kind in 'fc':
                        continue
                except Exception:
                    continue
                _cands.append((0 if _exact else 1, _c))
            _lab = None
            for _, _c in sorted(_cands)[:3]:
                try:
                    _nu = _scalar(_v[_c].nunique())
                    if _nu is not None and 1 < _nu <= 2:
                        _lab = _c
                        break
                except Exception:
                    continue
            if _lab is not None:
                for _c in _cols:
                    if _c == _lab:
                        continue
                    try:
                        if bool((_v[_c] == _v[_lab]).all()):
                            _info['label_dupes'].append(_c)
                    except Exception:
                        continue
            _sentinel_check_fp[_name] = (_fp, {'n_rows': _n, 'columns': _cols,
                                               'inf_cols': dict(_info['inf_cols']),
                                               'label_dupes': list(_info['label_dupes'])})
        # -- duplicate entity keys (ONLY for frames this cell exports) --
        if _sentinel_want_dupes and _name in _sentinel_export_names:
            for _c in _cols:
                if any(_h in _c.lower() for _h in _ent_hints):
                    try:
                        _d = _scalar(_n - _v[_c].nunique())
                        if _d:
                            _info['dup_keys'][_c] = _d
                    except Exception:
                        pass
                    break
        _out[_name] = _info
    print("''' + _PAYLOAD_START + r'''" + _json.dumps(_out, default=str) + "''' + _PAYLOAD_END + r'''")
except Exception:
    pass
'''


def collect_payload(kernel, code: str) -> dict:
    """Run the summary snippet in `kernel` and return the parsed payload.

    Best-effort: any failure returns {} so a check problem can never break
    a cell that otherwise ran fine (same posture as every sibling check).
    """
    try:
        want = "True" if _looks_like_export(code) else "False"
        names = sorted(exported_frame_names(code))
        kernel._exec_raw(f"_sentinel_want_dupes = {want}\n"
                         f"_sentinel_export_names = set({names!r})\n"
                         f"_sentinel_cell_code = {code!r}", timeout=30)
        res = kernel._exec_raw(CHECK_SNIPPET, timeout=180)
        out = getattr(res, "output", "") or ""
        s = out.find(_PAYLOAD_START)
        if s == -1:
            return {}
        e = out.find(_PAYLOAD_END, s)
        if e == -1:
            return {}
        data = json.loads(out[s + len(_PAYLOAD_START):e])
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 -- advisory, never fatal
        return {}


def checks_from_payload(payload: dict, code: str, seen: dict) -> list[tuple[str, str]]:
    """Run every summary-derivable check over `payload`.

    Returns [(issue_kind, block_text), ...]. `seen` maps issue-kind ->
    set(), preserving the "fire once per issue, not once per cell"
    contract each exec-mode check already has.
    """
    leakage_msgs: list[str] = []
    numeric_msgs: list[str] = []
    ambiguity_msgs: list[str] = []
    grain_msgs: list[str] = []

    for fname, info in (payload or {}).items():
        try:
            columns = [str(c) for c in info.get("columns", [])]
            n_rows = int(info.get("n_rows") or 0)

            # -- leakage: name lexicon + identical-to-label ----------------
            for col, reason in _lexicon_hits(columns):
                key = f"lexicon:{fname}:{col}"
                if key in seen["leakage"]:
                    continue
                seen["leakage"].add(key)
                leakage_msgs.append(f"'{col}' in '{fname}': {reason}")
            for col in info.get("label_dupes", []) or []:
                key = f"derived:{fname}:{col}"
                if key in seen["leakage"]:
                    continue
                seen["leakage"].add(key)
                leakage_msgs.append(
                    f"'{col}' in '{fname}' has values identical to the label "
                    "column -- built from the target."
                )

            # -- numeric stability (inf/-inf) ------------------------------
            for col, n_inf in (info.get("inf_cols") or {}).items():
                key = f"inf:{fname}:{col}"
                if key in seen["numeric"]:
                    continue
                seen["numeric"].add(key)
                numeric_msgs.append(_numeric_guidance(str(col), fname, int(n_inf)))

            # -- ambiguous / operational columns ---------------------------
            for col in _operational_hits(columns):
                key = f"ambiguous:{fname}:{col}"
                if key in seen["ambiguity"]:
                    continue
                seen["ambiguity"].add(key)
                ambiguity_msgs.append(_ambiguity_guidance(str(col), fname))

            # -- output grain (already gated on export by the snippet) -----
            for col, dupes in (info.get("dup_keys") or {}).items():
                key = f"grain:{fname}:{col}"
                if key in seen["grain"]:
                    continue
                seen["grain"].add(key)
                grain_msgs.append(_grain_guidance(str(col), int(dupes), n_rows))
        except Exception:  # noqa: BLE001 -- one bad frame must not kill the rest
            continue

    blocks: list[tuple[str, str]] = []
    if leakage_msgs:
        blocks.append(("target_leakage", "[LEAKAGE CHECK] Possible target leakage:\n"
                       + "\n".join(f"  - {m}" for m in leakage_msgs)))
    if numeric_msgs:
        blocks.append(("numeric_stability", "[NUMERIC CHECK] Non-finite values found:\n"
                       + "\n".join(f"  - {m}" for m in numeric_msgs)))
    if ambiguity_msgs:
        blocks.append(("ambiguous_column", "[AMBIGUOUS COLUMN] Operational/workflow-looking "
                       "column(s):\n" + "\n".join(f"  - {m}" for m in ambiguity_msgs)))
    if grain_msgs:
        blocks.append(("output_grain", "[OUTPUT GRAIN] Duplicate entity keys in an exported "
                       "frame:\n" + "\n".join(f"  - {m}" for m in grain_msgs)))
    return blocks
