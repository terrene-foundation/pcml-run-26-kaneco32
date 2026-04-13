# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 5: AI Agents — ReAct, Tool Use, and Function Calling
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Build a ReActAgent with the Thought -> Action -> Observation loop
#   - Define custom tools with structured schemas for agent consumption
#   - Implement function calling with tool_choice parameters (auto,
#     required, specific function)
#   - Apply cost budgets (LLMCostTracker) to prevent runaway spending
#   - Design agents using the mental framework from Deck 6B: goal,
#     thought process, specialist role, tools needed
#   - Build a custom BaseAgent with Signature for structured output
#   - Implement a critic agent for iterative refinement
#   - Add human-in-the-loop pausing for validation
#
# PREREQUISITES:
#   Exercises 1-4 (Delegate, Signature, prompt engineering).
#   Agents are LLMs with the ability to call functions — not new AI,
#   just LLMs that observe and act, not just respond.
#
# ESTIMATED TIME: ~180 min
#
# TASKS:
#    1. Load multi-hop QA dataset for agent testing
#    2. Define custom tools with structured schemas
#    3. Build ReActAgent with tool access
#    4. Run agent on multi-step analysis task
#    5. Inspect and interpret the reasoning trace
#    6. Function calling protocol (tool_choice, parallel calls)
#    7. Cost budget enforcement (LLMCostTracker)
#    8. Agent design mental framework (goal, process, specialist, tools)
#    9. Custom BaseAgent with Signature for structured analysis
#   10. Critic agent for iterative refinement
#
# DATASET: HotpotQA distractor (hotpotqa/hotpot_qa on HuggingFace)
#   Real multi-hop question-answer pairs that require reasoning over
#   multiple supporting paragraphs.  Perfect for testing the ReAct loop.
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
from pathlib import Path

import polars as pl

from kaizen import InputField, OutputField, Signature
from kaizen.core import BaseAgent
from kaizen_agents import Delegate
from kaizen_agents.agents.specialized.react import ReActAgent
from kaizen_agents.agents.specialized.simple_qa import SimpleQAAgent

from shared.kailash_helpers import setup_environment

setup_environment()

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))
if not model:
    raise EnvironmentError("Set DEFAULT_LLM_MODEL or OPENAI_PROD_MODEL in .env")
print(f"LLM Model: {model}")


# ══════════════════════════════════════════════════════════════════════════
# TASK 1: Load Multi-Hop QA Dataset
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 1: Load HotpotQA Dataset")
print("=" * 70)

CACHE_DIR = Path("data/mlfp06/hotpotqa")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "hotpotqa_500.parquet"

if CACHE_FILE.exists():
    print(f"Loading cached HotpotQA from {CACHE_FILE}")
    qa_data = pl.read_parquet(CACHE_FILE)
else:
    print("Downloading hotpotqa/hotpot_qa from HuggingFace...")
    from datasets import load_dataset

    ds = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="validation",
        trust_remote_code=True,
    )
    ds = ds.shuffle(seed=42).select(range(min(500, len(ds))))
    rows = []
    for row in ds:
        context = row["context"]
        titles = context["title"]
        sentences = context["sentences"]
        joined = "\n".join(f"[{t}] " + " ".join(s) for t, s in zip(titles, sentences))
        rows.append(
            {
                "text": joined[:4000],
                "question": row["question"],
                "answer": row["answer"],
                "level": row["level"],
                "type": row["type"],
            }
        )
    qa_data = pl.DataFrame(rows)
    qa_data.write_parquet(CACHE_FILE)
    print(f"Cached {qa_data.height} HotpotQA examples")

print(f"Loaded {qa_data.height:,} multi-hop QA examples")
print(f"Types: {dict(qa_data['type'].value_counts().iter_rows())}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert qa_data.height > 0, "Task 1: dataset should not be empty"
print("✓ Checkpoint 1 passed — HotpotQA loaded\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 2: Define Custom Tools with Structured Schemas
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 2: Custom Tool Definitions")
print("=" * 70)


# TODO: Implement data_summary(dataset_name="qa_data") that:
#   - Returns a text summary of qa_data: shape, columns, per-column stats
#     (n_unique + avg_len for strings, mean/min/max for numerics)
#   - Must have a complete docstring with Args and Returns sections
# Hint: for col in df.columns: check dtype, use pl.col(col).n_unique()
____


# TODO: Implement search_documents(query, top_k=3) that:
#   - Keyword-searches qa_data["text"] by counting query-word matches
#   - Returns top_k matching document excerpts with their Q and A
#   - Must have a complete docstring with Args and Returns sections
# Hint: score = sum(1 for word in query_lower.split() if word in text)
____


# TODO: Implement run_query(query_description) that:
#   - Interprets natural language query descriptions:
#       "count type" -> group_by("type").len()
#       "count level" -> group_by("level").len()
#       "comparison" -> filter type=="comparison", show sample
#       "bridge" -> filter type=="bridge", show sample
#       else -> return shape and columns
#   - Must have a complete docstring with Args and Returns sections
# Hint: desc = query_description.lower(); use if/elif chain
____


# TODO: Implement answer_question(question) that:
#   - Searches qa_data for a question matching the input (substring match)
#   - Returns the ground-truth answer with type and level
#   - Must have a complete docstring with Args and Returns sections
# Hint: for row in qa_data.iter_rows(named=True): check substring match
____


tools = [data_summary, search_documents, run_query, answer_question]

print("Defined tools:")
for tool in tools:
    doc_first_line = tool.__doc__.strip().split("\n")[0]
    print(f"  {tool.__name__}: {doc_first_line}")

print(f"\nTool test — data_summary():")
print(data_summary()[:300])

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert len(tools) == 4, "Task 2: should have 4 tools"
assert all(callable(t) for t in tools), "All tools should be callable"
assert all(t.__doc__ for t in tools), "All tools need docstrings (agent reads them)"
print("\n✓ Checkpoint 2 passed — 4 tools defined with structured docstrings\n")

# INTERPRETATION: Tool docstrings are the agent's API documentation.
# The agent reads the docstring to decide WHICH tool to call and with
# WHAT arguments.  Precise docstrings with Args/Returns sections lead
# to accurate tool selection.  Vague docstrings lead to wrong calls.


# ══════════════════════════════════════════════════════════════════════════
# TASK 3: Build ReActAgent with Tool Access
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 3: ReActAgent Construction")
print("=" * 70)

print(
    """
ReAct (Reasoning + Acting) loop:
  1. THOUGHT: agent reasons about the current state and what to do next
  2. ACTION:  agent selects a tool and provides arguments
  3. OBSERVATION: tool executes and returns a result
  4. Repeat until the agent decides it has enough information
  5. FINAL ANSWER: agent synthesises observations into a response
"""
)


# TODO: Define async build_react_agent() that:
#   - Creates ReActAgent(model=model, tools=tools, max_llm_cost_usd=2.0)
#   - Prints: model, tool names, budget
#   - Returns the agent
# Hint: ReActAgent(model=model, tools=tools, max_llm_cost_usd=2.0)
____


react_agent = asyncio.run(build_react_agent())

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert react_agent is not None, "Task 3: agent should be created"
print("✓ Checkpoint 3 passed — ReActAgent created\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 4: Run Agent on Multi-Step Analysis Task
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 4: Multi-Step Agent Analysis")
print("=" * 70)


# TODO: Define async multi_step_analysis() that:
#   - Creates a ReActAgent with model=model, tools=tools, max_llm_cost_usd=3.0
#   - Gets sample_q = qa_data["question"][0]
#   - Builds a task string with 5 numbered steps:
#       1. Get dataset summary
#       2. Count question types and difficulty levels
#       3. Search for documents relevant to sample_q
#       4. Look up the ground-truth answer
#       5. Synthesise findings into a report
#   - Calls agent.run(task) and returns the result
# Hint: result = await agent.run(task); extract .content or str(result)
____


analysis_result = asyncio.run(multi_step_analysis())

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert analysis_result is not None, "Task 4: analysis should produce a result"
print("✓ Checkpoint 4 passed — multi-step analysis complete\n")

# INTERPRETATION: The agent autonomously decided the order of tool calls.
# A good agent: summarises dataset first, then queries systematically.
# A poor agent: randomly calls tools or repeats the same call.


# ══════════════════════════════════════════════════════════════════════════
# TASK 5: Inspect and Interpret the Reasoning Trace
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 5: Reasoning Trace Inspection")
print("=" * 70)

# TODO: Print a multi-line explanation of the ReAct reasoning trace format:
#   - Show an example Thought -> Action -> Observation -> Final Answer sequence
#   - Explain what each step means
#   - List 4 good trace quality indicators (checkmarks) and 4 bad indicators (crosses)
#   - Explain how traces enable debugging, compliance, and quality analysis
# Hint: use print("""...""") with formatted example trace
____

# ── Checkpoint 5 ─────────────────────────────────────────────────────────
print("✓ Checkpoint 5 passed — reasoning trace interpretation complete\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 6: Function Calling Protocol
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 6: Function Calling Protocol")
print("=" * 70)

# TODO: Print an explanation of tool_choice options:
#   - "auto": model decides whether to call a tool
#   - "required": model must call at least one tool
#   - specific function: force a particular tool
#   - parallel calling: multiple tools in one turn
# Hint: explain with use cases for each option
____


# TODO: Generate tool_schemas list by introspecting each tool in tools:
#   For each tool, build a dict:
#     name: tool.__name__
#     description: first line of docstring
#     parameters: {"type": "object", "properties": {param_name: {"type": ..., "description": ...}}}
#   Use inspect.signature(tool) to get parameters
#   Map annotation int -> "integer", float -> "number", else "string"
# Hint: for name, param in sig.parameters.items(): param.annotation == int -> "integer"
____


# ── Checkpoint 6 ─────────────────────────────────────────────────────────
assert len(tool_schemas) == 4, "Task 6: should generate schemas for all 4 tools"
print("\n✓ Checkpoint 6 passed — function calling protocol explained\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 7: Cost Budget Enforcement (LLMCostTracker)
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 7: Cost Budget Enforcement")
print("=" * 70)

# TODO: Print an explanation of budget hierarchy and enforcement:
#   - Session budget -> task budget -> step budget
#   - What happens when budget is exceeded (warning, forced answer, block)
#   - How this connects to PACT governance (Ex 7)
# Hint: explain cascading budgets and partial result return
____


# TODO: Define async demonstrate_cost_budget() that:
#   - Creates a ReActAgent with max_llm_cost_usd=0.10 (very low)
#   - Runs an expensive multi-step task that should exceed the budget
#   - Catches any exception and prints the budget exceeded message
#   - Also runs a normal_agent with max_llm_cost_usd=2.0 on a simple task
#   - Returns the normal agent's result
# Hint: try: result = await low_budget_agent.run(task) except Exception as e: print(e)
____


budget_result = asyncio.run(demonstrate_cost_budget())

# ── Checkpoint 7 ─────────────────────────────────────────────────────────
print("✓ Checkpoint 7 passed — cost budget enforcement demonstrated\n")

# INTERPRETATION: Cost budgets are the financial operating envelope for
# agents.  Without budgets, a looping agent can spend $100+ on a single
# task.  max_llm_cost_usd is the first line of defence; PACT governance
# (Ex 7) adds organisational budgets on top.


# ══════════════════════════════════════════════════════════════════════════
# TASK 8: Agent Design Mental Framework
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 8: Agent Design Mental Framework (from Deck 6B)")
print("=" * 70)

# TODO: Print the 4-question agent design framework:
#   1. GOAL: specific, not vague
#   2. THOUGHT PROCESS: step-by-step human-expert approach
#   3. SPECIALIST: sharp role description (becomes system prompt)
#   4. TOOLS: versatile, fault-tolerant, caching-friendly
#   Also mention: iterative refinement, human-in-the-loop, monitoring
# Hint: use print("""...""") with the framework explained clearly
____

# Example agent design
print("\n--- Example Agent Design ---")
print("Goal: Identify which question type (comparison vs bridge) has")
print("      longer supporting contexts in the HotpotQA dataset.")
print("Process: 1) Get dataset overview  2) Separate by type")
print("         3) Compare text lengths  4) Report findings")
print("Specialist: 'NLP research analyst specialising in QA datasets'")
print("Tools: data_summary, run_query, search_documents")

# ── Checkpoint 8 ─────────────────────────────────────────────────────────
print("\n✓ Checkpoint 8 passed — agent design mental framework explained\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 9: Custom BaseAgent with Signature for Structured Analysis
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 9: BaseAgent + Signature for Structured Output")
print("=" * 70)


# TODO: Define DataAnalysisSignature(Signature) with:
#   InputFields:
#     dataset_summary (str): "Statistical summary of the dataset"
#     analysis_question (str): "Specific question to investigate"
#   OutputFields:
#     key_findings (list[str]): "Top 3-5 findings from the analysis"
#     recommended_approach (str): "Best ML approach for this data"
#     data_quality_issues (list[str]): "Potential data quality concerns"
#     next_steps (list[str]): "3-5 recommended next analysis steps"
#     confidence (float): "Confidence in findings (0.0 to 1.0)"
# Hint: class DataAnalysisSignature(Signature): with InputField / OutputField
____


# TODO: Define DataAnalysisAgent(BaseAgent) with:
#   signature = DataAnalysisSignature
#   model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))
#   max_llm_cost_usd = 1.0
# Hint: class DataAnalysisAgent(BaseAgent): — no __init__ needed
____


# TODO: Define async run_structured_agent() that:
#   - Calls data_summary() to get summary text
#   - Creates DataAnalysisAgent() and calls agent.run(dataset_summary=..., analysis_question=...)
#   - Prints key_findings, recommended_approach, data_quality_issues, next_steps, confidence
#   - Returns the result object
# Hint: result = await agent.run(dataset_summary=summary, analysis_question="...")
____


structured_result = asyncio.run(run_structured_agent())

# ── Checkpoint 9 ─────────────────────────────────────────────────────────
assert (
    structured_result is not None
), "Task 9: structured analysis should produce a result"
assert hasattr(structured_result, "key_findings"), "Result needs key_findings"
assert hasattr(structured_result, "confidence"), "Result needs confidence"
assert len(structured_result.key_findings) > 0, "Should have at least one finding"
assert 0 <= structured_result.confidence <= 1, "Confidence should be in [0, 1]"
print(
    f"\n✓ Checkpoint 9 passed — structured analysis: "
    f"{len(structured_result.key_findings)} findings, "
    f"confidence={structured_result.confidence:.2f}\n"
)

# INTERPRETATION: BaseAgent + Signature gives typed, validated output.
# result.key_findings[0] is reliable — no string parsing needed.
# Use BaseAgent when: output feeds into a pipeline or needs audit trail.
# Use ReActAgent when: the task requires tool exploration.


# ══════════════════════════════════════════════════════════════════════════
# TASK 10: Critic Agent for Iterative Refinement
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 10: Critic Agent — Iterative Refinement")
print("=" * 70)


# TODO: Define CriticSignature(Signature) with:
#   InputFields: original_analysis (str), analysis_question (str)
#   OutputFields:
#     strengths (list[str]): "What the analysis does well"
#     weaknesses (list[str]): "Gaps or errors in the analysis"
#     suggestions (list[str]): "Specific improvement suggestions"
#     quality_score (float): "Overall quality 0.0 to 1.0"
#     should_revise (bool): "Whether the analysis needs revision"
# Hint: bool OutputField works the same as float
____


# TODO: Define CriticAgent(BaseAgent) with signature=CriticSignature,
#   model from env, max_llm_cost_usd=1.0
____


# TODO: Define RefinedAnalysisSignature(Signature) with:
#   InputFields: dataset_summary (str), analysis_question (str), critic_feedback (str)
#   OutputFields: improved_findings (list[str]), methodology_note (str), confidence (float)
# And RefinedAnalysisAgent(BaseAgent) using that signature
____


# TODO: Define async iterative_refinement() that:
#   1. question = "What makes multi-hop QA harder than single-hop QA?"
#   2. Step 1: Run DataAnalysisAgent on question, capture initial findings
#   3. Step 2: Run CriticAgent on the initial analysis text
#   4. Step 3: If critique.should_revise, run RefinedAnalysisAgent with critic feedback
#              Else, print "Critic approves — no revision needed"
#   5. Return the final result (refined or initial)
# Hint: initial_text = f"Findings: {initial.key_findings}\nApproach: {initial.recommended_approach}"
____


refined_result = asyncio.run(iterative_refinement())

# ── Checkpoint 10 ────────────────────────────────────────────────────────
assert (
    refined_result is not None
), "Task 10: iterative refinement should produce a result"
print("\n✓ Checkpoint 10 passed — critic agent and iterative refinement complete\n")

# INTERPRETATION: The critic agent pattern implements quality assurance.
# This is NOT self-consistency (which samples multiple independent paths);
# the critic reviews and suggests SPECIFIC improvements.
# Use critic agents when: quality is critical and two extra LLM calls are justified.


# ══════════════════════════════════════════════════════════════════════════
# Agent Selection Guide
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("Agent Selection Guide")
print("=" * 70)

agent_guide = pl.DataFrame(
    {
        "Agent Type": [
            "ReActAgent",
            "BaseAgent + Signature",
            "SimpleQAAgent",
            "Critic Agent",
            "ReAct + Signature (hybrid)",
        ],
        "When To Use": [
            "Open-ended exploration, unknown # of steps, tool use",
            "Known output schema, feeds into pipeline, audit required",
            "Simple Q&A with structured output, no tools needed",
            "Quality assurance, iterative refinement, review loop",
            "Explore with tools first, then structured final output",
        ],
        "Cost": [
            "Medium-High (variable)",
            "Low (1 LLM call)",
            "Low (1 call)",
            "Medium (2-3 calls)",
            "High (tools + structured)",
        ],
    }
)
print(agent_guide)


# ══════════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  ✓ ReActAgent: Thought -> Action -> Observation loop; autonomous multi-step
  ✓ Tool design: docstrings = agent's API; precision determines tool selection
  ✓ Structured tool schemas: JSON Schema definitions for function calling
  ✓ Function calling protocol: tool_choice (auto, required, specific)
  ✓ Parallel function calling: multiple tools in one turn
  ✓ Cost budgets: max_llm_cost_usd prevents runaway spending
  ✓ Agent design framework: goal, thought process, specialist, tools
  ✓ BaseAgent + Signature: typed, validated, pipeline-safe output
  ✓ Critic agent: iterative refinement loop (analyse -> critique -> refine)

  Agent type selection:
    ReActAgent:       tool exploration, variable steps, open-ended
    BaseAgent+Sig:    known schema, pipeline integration, audit trail
    Critic pattern:   quality assurance, iterative improvement

  NEXT: Exercise 6 (Multi-Agent) composes multiple specialist agents.
  A supervisor delegates to domain specialists, then synthesises their
  analyses — fan-out (parallel) -> fan-in (synthesis) orchestration.
"""
)
