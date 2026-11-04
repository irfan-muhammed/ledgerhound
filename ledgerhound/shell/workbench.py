"""
ledgerhound.shell.workbench
===========================

Two kernel-side helpers the 2026-08-06 production run needed and did not
have: a memoised aggregation cache, and a principled non-fraud sampler.

Why -- measured, from that run's own event log
-----------------------------------------------
* **272 redundant recomputations.** `groupby('payer_account_number')` was
  recomputed **79** times, `payee_account_number` 38, `payer_device_id`
  19. Nothing cached a computed aggregate, and 58 context compressions
  meant the model kept forgetting it had already built one. `cached_agg`
  makes recomputation opt-out instead of automatic: the same
  (frame, keys, spec) returns the stored result instantly, and says so.

* **No sample frame.** The team's own answer said non-fraud rows should
  be drawn by SPE-style matched sampling at a declared ratio; the run
  instead trained on all 56M rows with no declared sampling frame at all.
  `sample_frame` builds a fraud/non-fraud sample at a stated ratio,
  optionally stratified so the non-fraud side matches the fraud side's
  distribution rather than being an arbitrary head-slice, and returns the
  frame together with the metadata needed to report it honestly (actual
  ratio achieved, population prevalence, stratification keys).

Both are injected into the kernel namespace alongside the other analyst
helpers, so they work on cuDF and pandas frames alike.
"""

from __future__ import annotations

WORKBENCH_SNIPPET = r'''
_LEDGERHOUND_AGG_CACHE = {}


def _ledgerhound_cache_key(df, keys, spec):
    try:
        _shape = (int(len(df)), int(len(df.columns)))
    except Exception:
        _shape = ("?", "?")
    return (str(sorted(map(str, df.columns)))[:400], _shape,
            str(keys), str(spec))


def cached_agg(df, keys, spec, name=None, force=False):
    """Memoised groupby-aggregate.

    cached_agg(df, 'payer_account_number', {'amount': ['mean', 'sum']})

    Returns the SAME object on a repeat call instead of recomputing. In the
    last production run the identical account-level groupby was recomputed
    79 times; this makes that free. Pass force=True to recompute
    deliberately (e.g. after mutating df).
    """
    _k = _ledgerhound_cache_key(df, keys, spec)
    if not force and _k in _LEDGERHOUND_AGG_CACHE:
        print("[cached_agg] HIT -- returning the aggregate already computed "
              "for these keys (no recomputation).")
        return _LEDGERHOUND_AGG_CACHE[_k]
    _out = df.groupby(keys).agg(spec)
    # dask/dask_cudf return a LAZY graph; caching that would store an
    # unevaluated object and make every later len()/print() recompute it.
    if hasattr(_out, 'npartitions') and hasattr(_out, 'compute'):
        _out = _out.compute()
    try:
        _out.columns = ['_'.join(map(str, c)).strip('_')
                        if isinstance(c, tuple) else str(c)
                        for c in _out.columns]
    except Exception:
        pass
    _out = _out.reset_index()
    _LEDGERHOUND_AGG_CACHE[_k] = _out
    print("[cached_agg] computed and cached -- rows=%d, name=%s"
          % (len(_out), name or '(unnamed)'))
    return _out


def cached_clear():
    _LEDGERHOUND_AGG_CACHE.clear()
    print("[cached_agg] cache cleared")


def _ledgerhound_concat(_parts):
    """Concatenate frames with the engine they live in (cuDF or pandas)."""
    if hasattr(_parts[0], 'to_pandas'):
        try:
            import cudf as _c
            return _c.concat(_parts)
        except Exception:
            pass
    import pandas as _pd
    return _pd.concat(_parts)


def sample_frame(df, label_col, ratio=10, strata=None, seed=42,
                 max_fraud=None):
    """Build a declared fraud / non-fraud sample frame.

    ratio  -- non-fraud rows per fraud row (10 => 1:10)
    strata -- optional list of columns; the non-fraud side is then drawn to
              match the fraud side's distribution over those columns, rather
              than an arbitrary slice
    Returns (sample_df, meta) where meta records population prevalence, the
    ratio actually achieved, and the strata used -- so the sampling frame
    can be reported honestly instead of inferred later.
    """
    _pos = df[df[label_col] == 1]
    _neg = df[df[label_col] != 1]
    _n_pop = int(len(df))
    _n_pos_pop = int(len(_pos))
    if max_fraud is not None and len(_pos) > max_fraud:
        _pos = _pos.sample(n=int(max_fraud), random_state=seed)
    _want = int(len(_pos) * ratio)

    if strata:
        _keys = list(strata)
        # target counts per stratum, proportional to the fraud side
        _mix = _pos.groupby(_keys).size().reset_index()
        _mix.columns = _keys + ['_n_pos']
        _tot = float(_mix['_n_pos'].sum()) or 1.0
        _parts = []
        _mix_p = _mix.to_pandas() if hasattr(_mix, 'to_pandas') else _mix
        for _, _row in _mix_p.iterrows():
            _sel = _neg
            for _k in _keys:
                _sel = _sel[_sel[_k] == _row[_k]]
            _take = int(round(_want * (float(_row['_n_pos']) / _tot)))
            if _take <= 0 or len(_sel) == 0:
                continue
            _parts.append(_sel.sample(n=min(_take, len(_sel)),
                                      random_state=seed))
        if _parts:
            # J-04: DataFrame.append is gone in pandas 2 and there is no
            # DataFrame.concat classmethod -- use the module-level concat
            # of whichever engine the parts live in.
            _neg_s = _ledgerhound_concat(_parts)
        else:
            _neg_s = _neg.sample(n=min(_want, len(_neg)), random_state=seed)
    else:
        _neg_s = _neg.sample(n=min(_want, len(_neg)), random_state=seed)

    _out = _ledgerhound_concat([_pos, _neg_s])

    _meta = {
        'population_rows': _n_pop,
        'population_fraud_rows': _n_pos_pop,
        'population_fraud_rate': (_n_pos_pop / _n_pop) if _n_pop else None,
        'sampled_fraud_rows': int(len(_pos)),
        'sampled_nonfraud_rows': int(len(_neg_s)),
        'requested_ratio': ratio,
        'achieved_ratio': (len(_neg_s) / len(_pos)) if len(_pos) else None,
        'strata': list(strata) if strata else [],
        'seed': seed,
        'note': ('Rates computed on this frame are SAMPLE rates at the '
                 'stated ratio, NOT population rates. Tag any finding from '
                 'it sample_frame="within_stratum" and convert with the '
                 'population prevalence above before reporting.'),
    }
    print('[sample_frame] fraud=%d nonfraud=%d achieved_ratio=1:%.1f '
          'population_fraud_rate=%s'
          % (_meta['sampled_fraud_rows'], _meta['sampled_nonfraud_rows'],
             _meta['achieved_ratio'] or 0,
             ('%.4f%%' % (100 * _meta['population_fraud_rate']))
             if _meta['population_fraud_rate'] is not None else 'unknown'))
    return _out, _meta
'''
