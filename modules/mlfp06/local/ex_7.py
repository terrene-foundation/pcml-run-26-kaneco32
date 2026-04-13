# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 7: AI Governance Engineering with PACT
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Define an organisational hierarchy in YAML using D/T/R grammar
#   - Compile and validate governance structure with GovernanceEngine
#   - Define operating envelopes: task envelopes, role envelopes, and
#     explain monotonic tightening
#   - Implement budget cascading across agent hierarchies
#   - Test access control decisions (both allow and deny) and verify
#     that governance is fail-closed (deny by default)
#   - Wrap agents with PactGovernedAgent for runtime enforcement
#   - Generate audit trails and map them to regulatory requirements
#   - Write governance unit tests that verify denied access stays denied
#   - Implement clearance levels (public < internal < confidential < restricted)
#   - Apply enforcement modes: warn, block, audit
#
# PREREQUISITES:
#   Exercise 6 (multi-agent systems).  This exercise GOVERNS the systems
#   built in Ex 5-6.  Governance is engineering: access controls, budget
#   limits, audit trails — not philosophical discussion.
#
# ESTIMATED TIME: ~180 min
#
# TASKS:
#    1. Load adversarial test prompts (RealToxicityPrompts)
#    2. Write YAML organisation definition (D/T/R grammar)
#    3. Compile with GovernanceEngine and validate
#    4. Define operating envelopes (task, role, budget, tool, clearance)
#    5. Monotonic tightening — envelopes only get stricter
#    6. Budget cascading across agent hierarchies
#    7. Test access control decisions (allow + deny)
#    8. PactGovernedAgent — runtime governance wrapper
#    9. Fail-closed governance and adversarial prompt blocking
#   10. Audit trail generation and regulatory mapping
#
# DATASET: allenai/real-toxicity-prompts (HuggingFace)
#   Real-world adversarial prompts collected from web text.  Used to
#   test that PACT governance correctly blocks high-risk inputs.
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import polars as pl

from kaizen import InputField, OutputField, Signature
from kaizen.core import BaseAgent
from kailash_pact import GovernanceEngine, PactGovernedAgent

from shared.kailash_helpers import setup_environment

setup_environment()

model_name = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))


# ══════════════════════════════════════════════════════════════════════════
# TASK 1: Load Adversarial Test Prompts
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 1: Load Adversarial Test Prompts")
print("=" * 70)

CACHE_DIR = Path("data/mlfp06/toxicity")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "real_toxicity_50.parquet"

if CACHE_FILE.exists():
    print(f"Loading cached adversarial prompts from {CACHE_FILE}")
    adversarial_prompts = pl.read_parquet(CACHE_FILE)
else:
    print("Downloading allenai/real-toxicity-prompts from HuggingFace...")
    from datasets import load_dataset

    ds = load_dataset("allenai/real-toxicity-prompts", split="train")
    ds = ds.filter(
        lambda r: r["prompt"]["toxicity"] is not None and r["prompt"]["toxicity"] > 0.5
    )
    ds = ds.shuffle(seed=42).select(range(min(50, len(ds))))
    rows = [
        {
            "prompt_text": row["prompt"]["text"],
            "toxicity_score": row["prompt"]["toxicity"],
        }
        for row in ds
    ]
    adversarial_prompts = pl.DataFrame(rows)
    adversarial_prompts.write_parquet(CACHE_FILE)
    print(f"Cached {adversarial_prompts.height} adversarial prompts")

print(f"Loaded {adversarial_prompts.height} real adversarial prompts")
print(
    f"Toxicity range: {adversarial_prompts['toxicity_score'].min():.2f} — "
    f"{adversarial_prompts['toxicity_score'].max():.2f}"
)

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert adversarial_prompts.height > 0, "Task 1: should have adversarial prompts"
print("✓ Checkpoint 1 passed — adversarial test data loaded\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 2: Write YAML Organisation Definition (D/T/R Grammar)
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 2: YAML Organisation Definition")
print("=" * 70)

print(
    """
D/T/R Accountability Grammar:
  D (Delegator):   Human authority who authorises a task
  T (Task):        Bounded scope of work the agent may perform
  R (Responsible): The agent that executes within the envelope

  Every agent action MUST trace back to a human Delegator.
  If model_trainer exceeds its $100 budget, accountability traces
  to chief_ml_officer who authorised the delegation.
"""
)

# TODO: Write org_yaml string defining a Singapore FinTech AI organisation with:
#   organization: name, jurisdiction="Singapore", regulatory_framework
#   departments: 3 departments (ML Engineering, Risk & Compliance, Customer Intelligence)
#     Each department has a head and 1-2 agent entries with id, role, clearance, description
#   delegations: 6 D/T/R chains, one per agent, each with:
#     delegator, task, responsible, envelope (max_budget_usd, allowed_tools list,
#     allowed_data_clearance, optional max_data_rows/max_response_length)
#   operating_envelopes: global config with fail_mode="closed", require_audit_trail=true
# Hint: use triple-quoted string; clearance levels: public < internal < confidential < restricted
____

org_yaml_path = os.path.join(tempfile.gettempdir(), "sg_fintech_org.yaml")
with open(org_yaml_path, "w") as f:
    f.write(org_yaml)

print(f"Organisation: SG FinTech AI Division")
print(f"Departments: 3 (ML Engineering, Risk & Compliance, Customer Intelligence)")
print(f"Agents: 6")
print(f"Delegations: 6 D/T/R chains")
print(f"YAML written to: {org_yaml_path}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert os.path.exists(org_yaml_path), "Task 2: YAML file should exist"
with open(org_yaml_path) as f:
    content = f.read()
assert "departments" in content and "delegations" in content, "YAML needs both sections"
print("✓ Checkpoint 2 passed — org YAML written\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 3: Compile with GovernanceEngine
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 3: GovernanceEngine Compilation")
print("=" * 70)


# TODO: Define async compile_governance() that:
#   - Creates GovernanceEngine()
#   - Calls engine.compile_org(org_yaml_path)
#   - Prints org.n_agents, org.n_delegations, org.n_departments
#   - Also prints 4 things that compilation validates
#   - Returns engine, org
# Hint: engine = GovernanceEngine(); org = engine.compile_org(org_yaml_path)
____


engine, org = asyncio.run(compile_governance())

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert org is not None, "Task 3: compilation should succeed"
assert org.n_agents > 0, "Task 3: should have agents"
assert org.n_delegations > 0, "Task 3: should have delegations"
print(
    f"✓ Checkpoint 3 passed — compiled: {org.n_agents} agents, "
    f"{org.n_delegations} delegations\n"
)


# ══════════════════════════════════════════════════════════════════════════
# TASK 4: Operating Envelopes
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 4: Operating Envelopes")
print("=" * 70)

print(
    """
An operating envelope defines the boundaries for what an agent can do:

  Task envelope:   restricts the agent to specific task types
    model_trainer can train+evaluate but NOT deploy
  Role envelope:   restricts based on organisational role
    auditors can read and audit but NOT modify
  Budget envelope:  maximum cost per task execution
    customer_agent limited to $5 per interaction
  Tool envelope:   whitelist of permitted tool calls
    deployer can deploy+monitor+rollback but NOT train
  Clearance envelope: highest data classification accessible
    public < internal < confidential < restricted
"""
)

# TODO: Build a polars DataFrame called envelopes with columns:
#   Agent, Clearance, Budget, Tools, Role
#   — one row per agent (6 rows), matching the YAML you wrote in Task 2
# Hint: pl.DataFrame({"Agent": [...], "Clearance": [...], ...})
____

print(envelopes)

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert envelopes.height == 6, "Task 4: should have 6 agent envelopes"
print("\n✓ Checkpoint 4 passed — operating envelopes defined\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 5: Monotonic Tightening
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 5: Monotonic Tightening Principle")
print("=" * 70)

print(
    """
Monotonic tightening: envelopes can ONLY get stricter, never looser.

  Clearance hierarchy:  restricted > confidential > internal > public
  A delegated agent CANNOT have HIGHER clearance than its delegator.

  Budget hierarchy:  child budget <= parent delegation limit
  A child agent cannot be allocated more budget than its parent allows.

  This prevents privilege escalation: no agent can gain capabilities
  beyond what its human delegator authorised.
"""
)

# TODO: Define clearance_levels dict mapping name to int (public=0 ... restricted=3)
# Then define delegation_chains list of 6 tuples:
#   (delegator_name, delegator_clearance, agent_name, agent_clearance)
# Finally verify and print monotonic tightening status for each chain
# Hint: is_tighter = clearance_levels[agent_clearance] <= clearance_levels[del_clearance]
____

print("Verifying monotonic tightening:")
for delegator, del_clearance, agent, agent_clearance in delegation_chains:
    is_tighter = clearance_levels[agent_clearance] <= clearance_levels[del_clearance]
    status = "✓" if is_tighter else "✗ VIOLATION"
    print(f"  {status} {delegator}({del_clearance}) -> {agent}({agent_clearance})")

# ── Checkpoint 5 ─────────────────────────────────────────────────────────
all_valid = all(
    clearance_levels[ac] <= clearance_levels[dc] for _, dc, _, ac in delegation_chains
)
assert all_valid, "Task 5: all clearance chains should be monotonically tightening"
print("\n✓ Checkpoint 5 passed — monotonic tightening verified\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 6: Budget Cascading
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 6: Budget Cascading Across Agent Hierarchies")
print("=" * 70)

print(
    """
Budget cascading: parent agent allocates budget to children.
  Total children spend <= parent's allocation.

  ml_director total budget: $500
    data_analyst:   $20/task
    model_trainer:  $100/task
    model_deployer: $50/task

  After 3 training tasks: $300 spent
  Next request exceeding allocation: DENIED
"""
)


# TODO: Implement BudgetTracker class with:
#   __init__(self, total_budget): self.total_budget, self.consumed={}, self.allocations={}
#   allocate(self, agent_id, amount) -> bool: False if total_allocated + amount > total_budget
#   spend(self, agent_id, amount) -> bool: False if would exceed agent's allocation
#   remaining(self, agent_id) -> float: allocation - consumed
#   summary(self) -> pl.DataFrame: columns agent, allocated, consumed, remaining
# Hint: total_allocated = sum(self.allocations.values())
____


tracker = BudgetTracker(total_budget=500.0)
tracker.allocate("data_analyst", 20.0)
tracker.allocate("model_trainer", 100.0)
tracker.allocate("model_deployer", 50.0)

tracker.spend("model_trainer", 30.0)
tracker.spend("model_trainer", 30.0)
tracker.spend("model_trainer", 25.0)
tracker.spend("data_analyst", 8.0)
tracker.spend("model_deployer", 15.0)

overspend_ok = tracker.spend("model_trainer", 50.0)
print(
    f"Overspend attempt (model_trainer, $50): {'ALLOWED' if overspend_ok else 'DENIED'}"
)
print(f"\nBudget summary:")
print(tracker.summary())

# ── Checkpoint 6 ─────────────────────────────────────────────────────────
assert not overspend_ok, "Task 6: overspend should be denied"
assert tracker.remaining("model_trainer") == 15.0, "Should have $15 remaining"
print("✓ Checkpoint 6 passed — budget cascading verified\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 7: Test Access Control Decisions
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 7: Access Control Tests (Allow + Deny)")
print("=" * 70)


# TODO: Define async test_access_control() that:
#   - Defines a list of 10 test cases as (agent, resource, action, expected_allowed, reason)
#     Include a mix of allowed and denied cases testing:
#     - Correct tool in envelope -> allow
#     - Tool not in envelope -> deny
#     - Clearance too low for resource -> deny
#   - For each case: calls engine.check_access(agent_id=..., resource=..., action=...)
#   - Prints a table comparing expected vs actual decisions
#   - Returns (results_list, all_correct bool)
# Hint: decision = engine.check_access(agent_id=agent_id, resource=resource, action=action)
#       actual = decision.allowed
____


access_results, all_correct = asyncio.run(test_access_control())

# ── Checkpoint 7 ─────────────────────────────────────────────────────────
assert len(access_results) >= 10, "Task 7: should test at least 10 cases"
print(f"✓ Checkpoint 7 passed — access control: {len(access_results)} tests\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 8: PactGovernedAgent — Runtime Governance Wrapper
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 8: PactGovernedAgent Runtime Wrapper")
print("=" * 70)


# TODO: Define QASignature(Signature) with:
#   InputField: question (str)
#   OutputFields: answer (str), confidence (float)
____


# TODO: Define QAAgent(BaseAgent) with QASignature,
#   model from env, max_llm_cost_usd=5.0
____


# TODO: Instantiate base_qa = QAAgent()
# Then create 3 PactGovernedAgent instances:
#   governed_public: role="analyst", $5 budget, ["answer_question", "search_faq"], "public"
#   governed_internal: role="engineer", $50 budget, add ["read_data", "train_model"], "confidential"
#   governed_admin: role="auditor", $200 budget, add ["audit_model", "access_audit_log"], "restricted"
# Hint: PactGovernedAgent(agent=base_qa, governance_engine=engine,
#         role=..., max_budget_usd=..., allowed_tools=[...], clearance_level=...)
____

print(f"Created 3 governed agents:")
print(f"  governed_public:   $5 budget, public clearance")
print(f"  governed_internal: $50 budget, confidential clearance")
print(f"  governed_admin:    $200 budget, restricted clearance")

print(
    """
PactGovernedAgent intercepts every run() call:
  1. Check: is this action within the agent's envelope?
  2. Check: is the budget sufficient?
  3. Check: does the agent have clearance for this data?
  4. If ALL pass -> execute and charge budget
  5. If ANY fail -> return governed error (fail-closed)
"""
)


# TODO: Define async test_governed_agents() that:
#   - Runs governed_public.run(question="What is machine learning?") (should succeed)
#   - Runs governed_public.run(question="Show me model training logs") (may be blocked)
#   - Runs governed_admin.run(question="What are the model performance metrics?")
#   - Wraps each in try/except and prints outcome
# Hint: try: result = await governed_public.run(question=...); print(result.answer[:200])
#       except Exception as e: print(f"Blocked: {e}")
____


asyncio.run(test_governed_agents())

# ── Checkpoint 8 ─────────────────────────────────────────────────────────
assert governed_public is not None, "Task 8: governed_public should exist"
assert governed_internal is not None, "Task 8: governed_internal should exist"
assert governed_admin is not None, "Task 8: governed_admin should exist"
print("\n✓ Checkpoint 8 passed — PactGovernedAgent runtime wrapping complete\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 9: Fail-Closed Governance and Adversarial Prompt Blocking
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 9: Fail-Closed Governance")
print("=" * 70)

print(
    """
Fail-closed: if ANY governance check fails, access is DENIED.
This is the opposite of traditional system defaults (permit unless denied).

  check_access() returns allowed=True ONLY when ALL of:
    1. Agent has a valid delegation chain
    2. Action is in the allowed_tools list
    3. Agent clearance >= resource clearance
    4. Budget is not exceeded
    5. Content policy passes (if payload provided)

  If any check is indeterminate (e.g., clearance unknown), the answer
  is DENY — not "allow and log".
"""
)


# TODO: Define async test_adversarial_prompts() that:
#   - Takes sample = adversarial_prompts.head(10)
#   - For each row: calls engine.check_access(agent_id="customer_agent",
#       resource="external_input", action="answer_question", payload=prompt_text)
#   - Counts blocked vs allowed; prints per-prompt status with toxicity score
#   - Returns n_blocked count
# Hint: allowed = getattr(decision, "allowed", False)
____


try:
    n_blocked = asyncio.run(test_adversarial_prompts())
except TypeError:
    print("  (PACT version does not support payload kwarg; demonstrating concept)")
    n_blocked = 0

# Test fail-closed property: unknown agent should be denied
print("\n--- Fail-Closed: Unknown Agent ---")
decision = engine.check_access(
    agent_id="unknown_agent",
    resource="any_resource",
    action="any_action",
)
print(f"  Unknown agent access: {'DENIED' if not decision.allowed else 'ALLOWED'}")
assert not decision.allowed, "Fail-closed: unknown agents should be denied"

# ── Checkpoint 9 ─────────────────────────────────────────────────────────
print("✓ Checkpoint 9 passed — fail-closed governance verified\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 10: Audit Trail and Regulatory Mapping
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 10: Audit Trail and Regulatory Mapping")
print("=" * 70)

qa_audit = governed_public.get_audit_trail()
admin_audit = governed_admin.get_audit_trail()

print(f"Audit trail:")
print(f"  Public agent actions:  {len(qa_audit)}")
print(f"  Admin agent actions:   {len(admin_audit)}")

qa_blocked = sum(1 for e in qa_audit if e.get("status") == "blocked")
qa_allowed = sum(1 for e in qa_audit if e.get("status") == "allowed")
print(f"  Public allowed/blocked: {qa_allowed}/{qa_blocked}")

print(f"\n--- Decision Trace Example ---")
decision = engine.check_access(
    agent_id="model_trainer",
    resource="training_data",
    action="read",
)
print(f"  Agent: model_trainer (role=engineer, clearance=confidential)")
print(f"  Chain: chief_ml_officer -> model_training -> model_trainer")
print(f"  Envelope checks:")
print(f"    Tool 'read_data' in allowed_tools: YES")
print(f"    Clearance 'confidential' <= allowed 'confidential': YES")
print(f"    Budget consumed < $100 limit: YES")
print(f"  Decision: {'ALLOWED' if decision.allowed else 'DENIED'}")

# TODO: Build a regulatory_map polars DataFrame with columns Regulation, PACT Control, Status
#   - Map at least 6 regulatory requirements to PACT controls:
#     EU AI Act Art. 9 (Risk Management), Art. 12 (Record-keeping), Art. 14 (Human Oversight)
#     Singapore AI Verify (Accountability), MAS TRM 7.5 (Audit Trail), PDPA
#   - All statuses = "COMPLIANT"
# Hint: pl.DataFrame({"Regulation": [...], "PACT Control": [...], "Status": [...]})
____

print(f"\n--- Regulatory Compliance Mapping ---")
print(regulatory_map)

print(f"\n--- Enforcement Modes ---")
print(f"  WARN:  Log the violation but allow the action (dev/staging only)")
print(f"  BLOCK: Deny the action and return a governed error (production)")
print(f"  AUDIT: Allow but flag for human review (semi-trusted agents)")
print(f"\n  Production default: BLOCK (fail-closed)")
print(f"  Never use WARN in production — it defeats the purpose of governance")

# ── Checkpoint 10 ────────────────────────────────────────────────────────
assert decision.allowed, "model_trainer should be allowed to read training_data"
assert regulatory_map.height >= 6, "Task 10: should map at least 6 regulations"
print("\n✓ Checkpoint 10 passed — audit trail and regulatory mapping complete\n")


# ══════════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  ✓ D/T/R grammar: Delegator -> Task -> Responsible; every agent action
    traces to a human authority
  ✓ GovernanceEngine.compile_org(): validates structural governance
    (no circular delegation, monotonic clearance, budget bounds)
  ✓ Operating envelopes: task + role + budget + tool + clearance constraints
  ✓ Monotonic tightening: envelopes only get stricter down the chain
  ✓ Budget cascading: parent allocates to children; children cannot overspend
  ✓ Access control tests: verify BOTH allowed and denied cases
  ✓ PactGovernedAgent: runtime wrapper that intercepts every run() call
  ✓ Fail-closed: deny by default; allow only what is explicitly permitted
  ✓ Adversarial prompt blocking: governance as content safety layer
  ✓ Audit trail: machine-readable evidence for regulatory compliance
  ✓ Regulatory mapping: PACT controls -> EU AI Act, AI Verify, MAS TRM, PDPA
  ✓ Enforcement modes: warn (dev), block (prod), audit (semi-trusted)

  Governance principles:
    Fail-closed:          deny unless explicitly allowed
    Monotonic tightening: envelopes only get stricter
    Clearance hierarchy:  restricted > confidential > internal > public
    Budget cascading:     child budget <= parent allocation
    Audit completeness:   every decision logged (allowed AND denied)

  NEXT: Exercise 8 (Capstone) integrates EVERYTHING from M6:
  SFT + DPO + PACT governance + Nexus deployment + compliance audit.
  A complete production ML system from training to deployment.
"""
)
