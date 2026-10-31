"""
sentinel.adversarial.shell_wiring
====================================

Builds the real, tool-capable agents sentinel.adversarial.engine debates
with. Kept separate from engine.py so the debate MECHANICS (round
alternation, transcript assembly, judge reading) stay testable with plain
duck-typed fakes -- this is the only module that actually constructs a
ConversationalAnalyst.

Each role/side combination (payer-analyst, payer-adversary, payee-
analyst, payee-adversary) gets its OWN fresh agent -- never reused across
sides, so context from the payer round can't leak into the payee round's
supposedly-independent read of the evidence.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Callable, Optional

from sentinel.adversarial.engine import (
    UNBOUNDED_MAX_STEPS, UNBOUNDED_STEP_BUDGET, UNBOUNDED_TURN_BUDGET_S,
    _ADVERSARY_PERSONA, _ANALYST_PERSONA,
)


def build_agent_factory(*, analyst_llm, adversary_llm, out_root,
                        use_kernel: bool = False,
                        renderer: Callable[[str], None] = lambda s: None,
                        analyst_counter=None, adversary_counter=None):
    """Returns build_agent(role, data_path) -> ConversationalAnalyst,
    matching sentinel.adversarial.engine.run_case's expected signature.

    analyst_llm: the regular/base model client (the session's own `llm`).
    adversary_llm: the judge model client
    (sentinel.eval.judge_config.resolve_judge(...).build()).

    use_kernel defaults OFF: a single case's data slice (one entity's
    transaction neighbourhood, not the whole crore-scale file) is small
    enough that the plain pandas sandbox is sufficient, and skipping a
    kernel start x4 per case matters at 25 cases. Pass the parent
    session's own use_kernel through if GPU-scale case slices are
    expected.
    """
    from sentinel.shell.conversation import ConversationalAnalyst

    def build_agent(role: str, data_path: str):
        llm = analyst_llm if role == "analyst" else adversary_llm
        persona = _ANALYST_PERSONA if role == "analyst" else _ADVERSARY_PERSONA
        # _ts() (conversation.py's run-folder timestamp) is second-
        # resolution -- four agents built within the same second (payer x2,
        # payee x2) would otherwise collide on run_dir. A short unique
        # subdir per role/side call sidesteps that without touching _ts().
        agent_root = Path(out_root) / f"{role}_{uuid.uuid4().hex[:8]}"
        agent = ConversationalAnalyst(
            llm, project=None, out_root=agent_root, renderer=renderer,
            autonomous=True,     # unattended -- non-risk questions self-
                                 # answer; governance refusals (grain/
                                 # phase/HITL-high) are UNCHANGED.
            max_steps=UNBOUNDED_MAX_STEPS,
            step_call_budget=UNBOUNDED_STEP_BUDGET,
            turn_budget_seconds=UNBOUNDED_TURN_BUDGET_S,
            use_kernel=use_kernel,
        )
        agent.messages[0]["content"] += persona
        # Exact token counts from the first request: each role's OWN model
        # tokenizer (the analyst runs on the actor model, the adversary on
        # the judge). Without one, calibration from the gateway still takes
        # over after the first reply.
        counter = analyst_counter if role == "analyst" else adversary_counter
        comp = getattr(agent, "_compressor", None)
        if comp is not None and counter is not None:
            comp.set_counter(counter)
        # I-17: a debate agent is not running a phased EDA methodology --
        # in autonomous mode load_data would otherwise adopt
        # topdown_7phase and every round would carry phase checklists and
        # "no report saved" refusals that mean nothing here. Mark the
        # choice as already made (none) before the data loads.
        agent._methodology_asked = True
        agent._run_tool("load_data", {"path": str(data_path)})
        return agent

    return build_agent


def build_judge_llm(project=None, *, actor_base_url: str, actor_model: str,
                    actor_key: str, actor_verify_ssl: bool = True,
                    actor_models_url: str = ""):
    """The judge model client for the adversary role AND the final
    verdict pass -- same resolution /model-judge already uses
    (sentinel.eval.judge_config.resolve_judge), so setting a judge once
    in the shell covers both eval scoring and this."""
    from sentinel.eval.judge_config import resolve_judge
    spec = resolve_judge(project, actor_base_url=actor_base_url,
                         actor_model=actor_model, actor_key=actor_key,
                         actor_verify_ssl=actor_verify_ssl,
                         actor_models_url=actor_models_url)
    return spec.build(), spec
