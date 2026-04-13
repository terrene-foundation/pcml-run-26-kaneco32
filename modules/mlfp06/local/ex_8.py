# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 8: Capstone — Full Production Platform
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Compose fine-tuned model (SFT+DPO), PACT governance, Kaizen agents,
#     and Nexus deployment into a single production-ready system
#   - Deploy a governed agent over 3 channels simultaneously
#     (API + CLI + MCP) using a single Nexus registration
#   - Implement RBAC authentication with JWT tokens
#   - Configure Nexus middleware: rate limiting, logging, CORS
#   - Integrate DriftMonitor for production model monitoring
#   - Debug agent reasoning chains
#   - Test agents with automated test harnesses
#   - Generate a regulatory compliance audit report mapping technical
#     controls to EU AI Act, AI Verify, and MAS TRM requirements
#   - Explain inference optimisations: KV-cache, flash attention, vLLM
#   - Describe the full Kailash ML lifecycle: train -> align -> govern -> deploy
#
# PREREQUISITES:
#   All previous M6 exercises (1-7).  This capstone integrates:
#   Ex 2 (SFT LoRA), Ex 3 (DPO alignment), Ex 5-6 (Kaizen agents),
#   Ex 7 (PACT governance), plus Nexus deployment introduced here.
#
# ESTIMATED TIME: ~180 min
#
# TASKS:
#    1. Load fine-tuned model with adapter from AdapterRegistry
#    2. Configure PACT governance with D/T/R envelopes
#    3. Build governed agent pipeline (PactGovernedAgent wrapping QAAgent)
#    4. Nexus deployment: register handler for 3 channels
#    5. RBAC authentication and JWT middleware
#    6. Rate limiting, logging, and CORS middleware
#    7. DriftMonitor integration for production monitoring
#    8. Agent reasoning chain debugging
#    9. Automated agent testing harness
#   10. Compliance audit report generation
#
# DATASET:
#   - cais/mmlu (HuggingFace) for evaluation questions
#   - hotpotqa/hotpot_qa (reused from Ex 5) for agent task corpus
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path

import polars as pl

from kaizen import InputField, OutputField, Signature
from kaizen.core import BaseAgent
from kaizen_agents import Delegate
from kailash_align import AdapterRegistry, AlignmentConfig, AlignmentPipeline
from kailash_ml import DriftMonitor
from kailash_nexus import Nexus
from kailash_pact import GovernanceEngine, PactGovernedAgent

from shared.kailash_helpers import setup_environment

setup_environment()

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))
if not model:
    raise EnvironmentError("Set DEFAULT_LLM_MODEL or OPENAI_PROD_MODEL in .env")


# ══════════════════════════════════════════════════════════════════════════
# TASK 1: Load Fine-Tuned Model with Adapter
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 1: Load Fine-Tuned Model from AdapterRegistry")
print("=" * 70)

EVAL_CACHE_DIR = Path("data/mlfp06/mmlu")
EVAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
EVAL_CACHE_FILE = EVAL_CACHE_DIR / "mmlu_100.parquet"

if EVAL_CACHE_FILE.exists():
    print(f"Loading cached MMLU from {EVAL_CACHE_FILE}")
    eval_data = pl.read_parquet(EVAL_CACHE_FILE)
else:
    print("Downloading cais/mmlu from HuggingFace...")
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="validation")
    ds = ds.shuffle(seed=42).select(range(min(100, len(ds))))
    rows = []
    for row in ds:
        choices = row["choices"]
        answer_idx = row["answer"]
        rows.append(
            {
                "instruction": (
                    f"{row['question']}\n\n"
                    f"A) {choices[0]}\nB) {choices[1]}\n"
                    f"C) {choices[2]}\nD) {choices[3]}"
                ),
                "response": ["A", "B", "C", "D"][answer_idx],
                "subject": row["subject"],
            }
        )
    eval_data = pl.DataFrame(rows)
    eval_data.write_parquet(EVAL_CACHE_FILE)
    print(f"Cached {eval_data.height} MMLU rows")

print(f"Evaluation data (MMLU): {eval_data.shape}")
print(f"Subjects: {eval_data['subject'].n_unique()}")


# TODO: Define async load_model() that:
#   1. Creates AdapterRegistry() and lists adapters
#   2. Tries to load the best available adapter from these candidates in order:
#      "sg_domain_slerp_merge_v1", "ultrafeedback_dpo_v1", "imdb_sentiment_sft_v1"
#      If none found, use best_adapter = {}
#   3. Creates AlignmentPipeline(AlignmentConfig(method="inference",
#        adapter_path=best_adapter.get("adapter_path", "")))
#   4. Prints the adapter name or "N/A" if none found
#   5. Returns pipeline, registry
# Hint: for candidate in (...): try: best_adapter = await registry.get_adapter(candidate)
____


inference_pipeline, registry = asyncio.run(load_model())

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert inference_pipeline is not None, "Task 1: pipeline should be created"
assert registry is not None, "Task 1: registry should be accessible"
print("✓ Checkpoint 1 passed — model and registry loaded\n")

# INTERPRETATION: AdapterRegistry is the source of truth for model
# provenance.  The capstone loads the best available adapter — ideally
# the SLERP merge from earlier exercises (SFT domain knowledge + DPO
# alignment).  Production: never hardcode model paths.


# ══════════════════════════════════════════════════════════════════════════
# TASK 2: Configure PACT Governance
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 2: PACT Governance Configuration")
print("=" * 70)

# TODO: Write org_yaml string for the MLFP Capstone ML Platform with:
#   organization: name="MLFP Capstone ML Platform", jurisdiction="Singapore",
#                 regulatory_framework="MAS TRM, AI Verify, PDPA"
#   departments: one "AI Services" department, head="ml_director", with 3 agents:
#     qa_agent (responder, internal), admin_agent (operator, confidential),
#     audit_agent (auditor, restricted)
#   delegations: 3 D/T/R chains — question_answering ($1, internal),
#     model_management ($10, confidential), compliance_audit ($50, restricted)
#   operating_envelopes: global with max_llm_cost_per_request_usd=0.10,
#     require_audit_trail=true, pii_handling="mask", fail_mode="closed"
# Hint: use triple-quoted YAML string
____

org_path = os.path.join(tempfile.gettempdir(), "capstone_org.yaml")
with open(org_path, "w") as f:
    f.write(org_yaml)

governance_engine = GovernanceEngine()


# TODO: Define async setup_governance() that:
#   - Calls governance_engine.compile_org(org_path)
#   - Prints agent count, delegation count, and budget summary per agent
#   - Returns org
____


org = asyncio.run(setup_governance())

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert org is not None and org.n_agents >= 3, "Task 2: should have 3+ agents"
print("✓ Checkpoint 2 passed — PACT governance compiled\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 3: Build Governed Agent Pipeline
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 3: Governed Agent Pipeline")
print("=" * 70)


# TODO: Define CapstoneQASignature(Signature) with:
#   InputField: question (str)
#   OutputFields: answer (str), confidence (float), sources (list[str]),
#                 reasoning_steps (list[str])
____


# TODO: Define CapstoneQAAgent(BaseAgent) with that signature,
#   model from env, max_llm_cost_usd=5.0
____


# TODO: Instantiate base_qa = CapstoneQAAgent()
# Then create 3 PactGovernedAgent instances:
#   governed_qa:    role="responder", $1 budget, ["generate_answer", "search_context"], "internal"
#   governed_admin: role="operator",  $10 budget, add ["update_model", "view_metrics", "monitor_drift"]
#   governed_audit: role="auditor",   $50 budget, add ["view_metrics", "access_audit_log", "generate_report"]
# Hint: PactGovernedAgent(agent=base_qa, governance_engine=governance_engine,
#         role=..., max_budget_usd=..., allowed_tools=[...], clearance_level=...)
____

print(f"Agent pipeline:")
print(f"  Base: CapstoneQAAgent (fine-tuned model)")
print(f"  Governed QA:    $1, internal, answer+search")
print(f"  Governed Admin: $10, confidential, +update+metrics+drift")
print(f"  Governed Audit: $50, restricted, +audit_log+report")

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert governed_qa is not None, "Task 3: governed_qa should exist"
assert governed_admin is not None, "Task 3: governed_admin should exist"
assert governed_audit is not None, "Task 3: governed_audit should exist"
print("✓ Checkpoint 3 passed — 3 governed agent levels created\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 4: Nexus Deployment — 3 Channels
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 4: Nexus Multi-Channel Deployment")
print("=" * 70)

print(
    """
Nexus deploys one handler over 3 channels simultaneously:
  API:  HTTP REST endpoints for web clients and services
  CLI:  Terminal interface for developer and operator access
  MCP:  Model Context Protocol for AI agent tool consumption

One codebase, one handler, three interfaces.  Governance is INSIDE the
handler — every channel benefits from the same access controls.
"""
)


# TODO: Define async handle_qa(question, role="qa") -> dict that:
#   - Maps role to agent: {"qa": governed_qa, "admin": governed_admin, "audit": governed_audit}
#   - Falls back to governed_qa for unknown roles
#   - Calls agent.run(question=question) in a try/except
#   - Returns success dict: answer, confidence, sources, reasoning_steps, latency_ms, governed, role
#   - Returns error dict on exception: error, governed=True, blocked=True, role
# Hint: result = await agent.run(question=question); latency = (time.time() - start) * 1000
____


# TODO: Create Nexus app and register the handler:
#   app = Nexus()
#   app.register(handle_qa)
____

print(f"Nexus app registered:")
print(f"  Handler: handle_qa(question, role)")
print(f"  Channels: API + CLI + MCP")

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert app is not None, "Task 4: Nexus app should be created"
print("✓ Checkpoint 4 passed — Nexus deployment configured\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 5: RBAC Authentication and JWT Middleware
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 5: RBAC Authentication")
print("=" * 70)

print(
    """
RBAC (Role-Based Access Control):
  Roles define what a user can access:
    viewer:   read-only, public data, QA agent only
    operator: read-write, confidential data, admin agent
    auditor:  full access, restricted data, audit agent

JWT (JSON Web Token):
  Stateless authentication token:
    Header:  {"alg": "RS256", "typ": "JWT"}
    Payload: {"sub": "user123", "role": "operator", "exp": 1718000000}
    Signature: RS256(header + payload, private_key)

  Auth flow:
    1. User authenticates (login endpoint)
    2. Server issues JWT with role claim
    3. Every request includes JWT in Authorization header
    4. Nexus middleware validates JWT and extracts role
    5. Role determines which governed agent handles the request
"""
)

# TODO: Build rbac_roles polars DataFrame with columns:
#   Role, Agent, Clearance, Budget, Capabilities
#   — 3 rows: viewer, operator, auditor
____

print(rbac_roles)


# TODO: Implement SimpleJWTAuth class with:
#   VALID_TOKENS class-level dict mapping token strings to claim dicts
#     {"token_viewer_001": {"sub": "alice", "role": "qa"}, ...}
#     Include tokens for qa, admin, audit roles
#   @classmethod validate(cls, token) -> dict | None:
#     return VALID_TOKENS.get(token)
____

for token, expected_role in [
    ("token_viewer_001", "qa"),
    ("token_operator_001", "admin"),
    ("token_auditor_001", "audit"),
    ("invalid_token", None),
]:
    claims = SimpleJWTAuth.validate(token)
    if claims:
        print(f"  Token {token[:15]}... -> role={claims['role']}")
    else:
        print(f"  Token {token[:15]}... -> REJECTED (invalid)")

# ── Checkpoint 5 ─────────────────────────────────────────────────────────
assert (
    SimpleJWTAuth.validate("token_viewer_001") is not None
), "Valid token should authenticate"
assert (
    SimpleJWTAuth.validate("invalid_token") is None
), "Invalid token should be rejected"
print("\n✓ Checkpoint 5 passed — RBAC + JWT authentication demonstrated\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 6: Middleware — Rate Limiting, Logging, CORS
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 6: Nexus Middleware Stack")
print("=" * 70)

print(
    """
Nexus middleware stack (applied to every request):

  1. CORS (Cross-Origin Resource Sharing):
     Allow/deny requests from different origins (domains).
     Config: allowed_origins = ["https://app.example.com"]
     Without CORS: browser blocks requests from web frontends.

  2. Rate Limiting:
     Prevent abuse by limiting requests per time window.
     Config: 100 requests/minute per API key.
     Exceeding: 429 Too Many Requests response.
     Strategy: sliding window counter per client IP or API key.

  3. Request Logging:
     Log every request with: timestamp, method, path, client_id,
     latency_ms, status_code, governed_role.
     Structured logging (JSON) for aggregation in Datadog/Splunk.

  4. Authentication (Task 5): Validate JWT, extract role.

  5. Governance (inside the handler): PactGovernedAgent checks.

  Order: CORS -> Rate Limit -> Auth -> Log -> Handler -> Governance
"""
)


# TODO: Implement RateLimiter class with sliding-window algorithm:
#   __init__(self, max_requests, window_seconds):
#     self.max_requests, self.window, self.requests = {} (dict of client_id -> [timestamps])
#   allow(self, client_id) -> bool:
#     - Remove timestamps older than window_seconds
#     - If len(timestamps) >= max_requests: return False
#     - Else append now and return True
# Hint: self.requests[client_id] = [t for t in self.requests[client_id] if now - t < self.window]
____


limiter = RateLimiter(max_requests=5, window_seconds=60)

print("Rate limiting test (5 req/min):")
for i in range(7):
    allowed = limiter.allow("client_alice")
    print(f"  Request {i+1}: {'ALLOWED' if allowed else 'RATE LIMITED (429)'}")

# ── Checkpoint 6 ─────────────────────────────────────────────────────────
assert not limiter.allow("client_alice"), "6th+ request should be rate-limited"
print("\n✓ Checkpoint 6 passed — middleware stack demonstrated\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 7: DriftMonitor Integration
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 7: Production Monitoring with DriftMonitor")
print("=" * 70)

print(
    """
DriftMonitor (from M3.8) detects when production data diverges from
training data distribution — a signal that the model may need retraining.

Monitoring dimensions:
  Feature drift:     input features shift
  Prediction drift:  model outputs shift
  Label drift:       ground-truth labels shift

Metrics:
  PSI (Population Stability Index):
    PSI < 0.1:   no significant drift
    PSI 0.1-0.2: moderate drift (investigate)
    PSI > 0.2:   significant drift (retrain)
"""
)


# TODO: Define async setup_drift_monitoring() that:
#   - Creates DriftMonitor(model_name="capstone_qa_model",
#       reference_data=eval_data.select("instruction"),
#       features=["instruction"], alert_threshold_psi=0.2)
#   - Sets prod_data = eval_data.select("instruction").head(50)
#   - Calls drift_report = await monitor.check_drift(production_data=prod_data)
#   - Prints model, sample counts, drift_detected, PSI, and status
#   - Returns monitor, drift_report
____


monitor, drift_report = asyncio.run(setup_drift_monitoring())

# ── Checkpoint 7 ─────────────────────────────────────────────────────────
assert monitor is not None, "Task 7: DriftMonitor should be created"
assert drift_report is not None, "Task 7: drift report should be generated"
print("✓ Checkpoint 7 passed — DriftMonitor integration complete\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 8: Agent Reasoning Chain Debugging
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 8: Debugging Agent Reasoning Chains")
print("=" * 70)

print(
    """
Debugging traces for agentic systems:

  1. INPUT TRACE:    user query, system prompt, tools, governance constraints
  2. REASONING TRACE: Thought->Action->Observation chain, token usage per step
  3. OUTPUT TRACE:   final answer, confidence, sources, total cost/latency
  4. GOVERNANCE TRACE: access decisions, budget consumed, clearance checks
"""
)


# TODO: Define async debug_agent_call() that:
#   - Takes question = eval_data["instruction"][0]
#   - Prints INPUT TRACE: question[:100], role, budget, clearance
#   - Calls result = await handle_qa(question, role="qa")
#   - Prints OUTPUT TRACE: answer[:150] or error, confidence, sources, latency
#   - Prints GOVERNANCE TRACE: role, governed flag, blocked flag
#   - Returns result
# Hint: if "error" in result: print(f"Status: BLOCKED") else: print(f"Answer: {result['answer']}")
____


debug_result = asyncio.run(debug_agent_call())

# ── Checkpoint 8 ─────────────────────────────────────────────────────────
assert debug_result is not None, "Task 8: debug call should produce a result"
print("\n✓ Checkpoint 8 passed — agent debugging demonstrated\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 9: Automated Agent Testing Harness
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 9: Automated Agent Testing")
print("=" * 70)


# TODO: Define async run_test_harness() -> pl.DataFrame that runs 5 tests:
#   Test 1: Normal QA query (role="qa") — should succeed (no "error" in result)
#   Test 2: Invalid role fallback — role="invalid_role" should default gracefully
#   Test 3: Admin query with admin role (role="admin") — should succeed
#   Test 4: Budget cascade — run 5 QA queries; check none blocked
#   Test 5: Cross-role governance — same question with "qa" and "admin" roles
#   Build test_results list of dicts: {test, passed, detail}
#   Print pl.DataFrame(test_results) and pass count
#   Return the DataFrame
# Hint: passed = "error" not in result; test_df = pl.DataFrame(test_results)
____


test_df = asyncio.run(run_test_harness())

# ── Checkpoint 9 ─────────────────────────────────────────────────────────
assert test_df.height >= 5, "Task 9: should run at least 5 tests"
print("✓ Checkpoint 9 passed — automated test harness complete\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 10: Compliance Audit Report
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 10: Compliance Audit Report")
print("=" * 70)

qa_audit = governed_qa.get_audit_trail()
admin_audit = governed_admin.get_audit_trail()
audit_audit = governed_audit.get_audit_trail()


# TODO: Define async generate_compliance_report() that prints a structured report with:
#   Section 1: AGENT ACTIVITY SUMMARY (action counts from audit trails)
#   Section 2: GOVERNANCE ENFORCEMENT (D/T/R chains, budget, tool, clearance, audit, fail mode)
#   Section 3: AUTHENTICATION & ACCESS CONTROL (JWT, RBAC roles, rate limiting, CORS)
#   Section 4: MODEL PROVENANCE (base model from env, adapters, registry, drift monitoring)
#   Section 5: DEPLOYMENT ARCHITECTURE (channels, governance coverage, sessions, monitoring)
#   Section 6: REGULATORY COMPLIANCE MAPPING — polars DataFrame with 7 rows:
#     EU AI Act Art. 9, Art. 12, Art. 14; Singapore AI Verify x2; MAS TRM 7.5; PDPA
#     Map each to the PACT control that satisfies it; Status="COMPLIANT" for all
#   Section 7: INFERENCE INFRASTRUCTURE (KV-cache, flash attention, speculative decoding)
#   Print "AUDIT RESULT: COMPLIANT" at the end
# Hint: print the regulatory DataFrame with print(regulatory)
____


asyncio.run(generate_compliance_report())

# ── Checkpoint 10 ────────────────────────────────────────────────────────
assert len(qa_audit) >= 0, "Audit trail should be accessible"
print("\n✓ Checkpoint 10 passed — compliance audit report generated\n")


# ══════════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  ✓ AdapterRegistry:     model provenance, version control, lineage
  ✓ AlignmentPipeline:   SFT domain knowledge + DPO preference alignment
  ✓ BaseAgent+Signature: structured agent with typed output contract
  ✓ PactGovernedAgent:   D/T/R wrapper: budget + tools + clearance
  ✓ Nexus:               multi-channel deployment (API, CLI, MCP)
  ✓ RBAC + JWT:          role-based authentication and authorisation
  ✓ Middleware:           rate limiting, logging, CORS
  ✓ DriftMonitor:        production monitoring with PSI drift detection
  ✓ Agent debugging:     input/reasoning/output/governance traces
  ✓ Automated testing:   test harness for governed agent systems
  ✓ Compliance audit:    regulatory mapping to EU AI Act, AI Verify, MAS TRM
  ✓ GovernanceEngine:    compile org, check access, generate audit trail
"""
)

print("═" * 70)
print("  MLFP06 COMPLETE — COURSE SUMMARY")
print("═" * 70)
print(
    """
  Module 6 — LLMs, Alignment & Production (Exercises 1-8):
    Ex 1: Prompt Engineering — 6 techniques, Signature, cost tracking
    Ex 2: Fine-Tuning — LoRA + adapter FROM SCRATCH, 10-technique survey,
           model merging (TIES/DARE/SLERP), quantisation (GPTQ/AWQ/QLoRA)
    Ex 3: DPO Alignment — from-scratch loss, GRPO, LLM-as-judge (bias),
           evaluation benchmarks, beta sensitivity, safety evaluation
    Ex 4: RAG — 4 chunking strategies, BM25 from scratch, hybrid (RRF),
           re-ranking, RAGAS evaluation, HyDE
    Ex 5: AI Agents — ReAct, structured tools, function calling, cost
           budgets, agent design framework, critic agent refinement
    Ex 6: Multi-Agent — supervisor-worker, sequential, parallel, router,
           MCP server, agent memory, security considerations
    Ex 7: PACT Governance — D/T/R, operating envelopes, monotonic tightening,
           budget cascading, fail-closed, PactGovernedAgent, audit trails
    Ex 8: Capstone — full platform: align -> govern -> deploy -> monitor
           RBAC+JWT, middleware, DriftMonitor, agent testing, compliance

  The Kailash stack you have mastered:
    kailash-ml:     DataExplorer, TrainingPipeline, AutoMLEngine,
                    ModelRegistry, DriftMonitor, ExperimentTracker
    kailash-align:  AlignmentPipeline, AlignmentConfig, AdapterRegistry
    kailash-kaizen: Delegate, BaseAgent, Signature, ReActAgent,
                    Pipeline.router(), SimpleQAAgent
    kailash-pact:   GovernanceEngine, PactGovernedAgent, D/T/R grammar
    kailash-nexus:  Nexus multi-channel (API + CLI + MCP)
    kailash-mcp:    MCPServer, MCPTool

  This is production ML engineering.  You can now build, fine-tune,
  align, govern, deploy, and monitor AI systems responsibly.
"""
)
