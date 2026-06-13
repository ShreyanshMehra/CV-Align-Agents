"""CV-Align-Agents: a multi-agent resume screening pipeline.

Agents (parser, jd_parser, matcher, critic) are orchestrated with LangGraph
around a shared :class:`cv_align_agents.state.PipelineState`. Scoring is a
deterministic function so the final number is always reproducible and auditable.
"""

__version__ = "0.1.0"
