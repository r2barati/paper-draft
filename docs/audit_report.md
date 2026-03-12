# Benchmark Audit Report

Full code review of the environment (`environment.py`), all four agent implementations (`oracle.py`, `dlp_agent.py`, `mssp_agent.py`, `heuristic_agent.py`), and the benchmark harness (`benchmark.py`).

---

## 1. Environment Rules Summary

The environment (`NetInvMgmtMasterEnv_New`) enforces these physics:

| Rule | Where Enforced |
|------|---------------|
| **Order clipping** — orders are `round(max(action, 0))` | [_STEP L452](file:///Users/sanamimani/Desktop/NetworkInvLog-BenchmarkTest/environment.py#L452) |
| **Raw-material nodes** supply unlimited quantities | [_STEP L455-458](file:///Users/sanamimani/Desktop/NetworkInvLog-BenchmarkTest/environment.py#L455-L458) |
| **Distributor supply cap** — shipments ≤ on-hand inventory | [_STEP L460-465](file:///Users/sanamimani/Desktop/NetworkInvLog-BenchmarkTest/environment.py#L460-L465) |
| **Factory supply cap** — `min(request, C, v × Xₜ)` | [_STEP L467-474](file:///Users/sanamimani/Desktop/NetworkInvLog-BenchmarkTest/environment.py#L467-L474) |
| **Lead-time pipeline** — orders placed at *t* arrive at *t+L* | [_STEP L483-488](file:///Users/sanamimani/Desktop/NetworkInvLog-BenchmarkTest/environment.py#L483-L488) |
| **Demand after fulfillment** — retailers satisfy demand with post-arrival inventory | [_STEP L540-548](file:///Users/sanamimani/Desktop/NetworkInvLog-BenchmarkTest/environment.py#L540-L548) |
| **Backlog accumulation** — `backlog=True` adds prior unfulfilled to current effective demand | [_STEP L534-537](file:///Users/sanamimani/Desktop/NetworkInvLog-BenchmarkTest/environment.py#L534-L537) |
| **Goodwill update** — sentiment adjusts based on *previous period* unfulfilled | [_STEP L506-508](file:///Users/sanamimani/Desktop/NetworkInvLog-BenchmarkTest/environment.py#L506-L508) |

---

## 2. Agent-by-Agent Compliance

### ✅ Oracle (`oracle.py`) — COMPLIANT (upper bound, as expected)

- Uses **perfect future demand** (known_demand_scenario) — this is the intended "Perfect Information Relaxation" upper bound.
- Correctly models the **time-step alignment** constraint: upstream can only ship from yesterday's inventory (`outgoing ≤ prev_inv`), retailers can use arrivals (`outgoing ≤ prev_inv + incoming`). This matches the environment's "ship-before-arrive" vs "arrive-then-sell" logic.
- Capacity constraints present for factories.
- Pipeline costs are tracked precisely period-by-period (L178-188).
- Initial inventory read from `I0` graph attributes — matches `_RESET`.

> [!NOTE]
> The Oracle is correctly labeled as an **upper bound**, not a fair competitor. It has perfect foresight by design.

### ✅ DLP Agent (`dlp_agent.py`) — COMPLIANT

- **Informed mode**: reads `env.demand_engine.get_current_mu(t)` — this gives the *deterministic mean* μ at each future period. This is legitimate: it uses the demand engine's public API to get the expected demand, **not** the actual realization.
- **Blind mode**: uses SMA-5 of historical realised demand (`env.D[start:current, :]`) — fair, no future information.
- Time-step alignment constraints match Oracle/environment exactly.
- Reads current on-hand inventory from `env.X[current_period, :]` at `t=0` — this is the *current* state, not future state.
- Reads historical pipeline `env.R[past_time, :]` for in-transit orders — legitimate, this is past data.

### ✅ MSSP Agent (`mssp_agent.py`) — COMPLIANT

- Same structural correctness as DLP, plus stochastic scenario branching.
- **Informed mode**: uses `env.demand_engine.get_current_mu(t)` to build the scenario tree — same legitimate access as DLP.
- **Blind mode**: uses SMA-5 of realised demand history — fair.
- Non-anticipativity constraints (NAC) correctly enforced for branching scenarios.
- Time-step alignment and capacity constraints match.

### ✅ Heuristic Agent (`heuristic_agent.py`) — COMPLIANT

- Uses Newsvendor critical-ratio formula `CR = b/(b+h)` with Poisson inverse CDF.
- **Informed mode**: reads `env.demand_engine.get_current_mu(current_period)` — same legitimate access.
- **Blind mode**: uses SMA-5 of realised demand — fair.
- Reads current on-hand `env.X[period, :]`, current pipeline `env.Y[period:, :]`, and prior backlog `env.U[period-1, :]` — all legitimate current/historical state.

### ✅ Dummy Agent (`benchmark.py:DummyAgent`) — COMPLIANT

- Samples random actions from the action space — no information advantage.

---

## 3. Benchmarking Fairness Assessment

### ✅ Consistent Environment Instantiation

All agents receive **identical** `env_kwargs` per scenario (same `scenario`, `backlog`, `demand_config`, `num_periods`). The benchmark loop in [run() L256-265](file:///Users/sanamimani/Desktop/NetworkInvLog-BenchmarkTest/benchmark.py#L256-L265) constructs these once per scenario and passes them to every evaluator.

### ✅ Seed Control

All agents use `seed = 42 + episode` per episode ([L279](file:///Users/sanamimani/Desktop/NetworkInvLog-BenchmarkTest/benchmark.py#L279)). Each evaluator calls `env.reset(seed=seed)`, ensuring **identical demand realizations** for the same episode across agents.

### ✅ Metrics Consistency

All evaluators compute the same 4 metrics: `profit`, `avg_inv`, `unfulfilled`, and `fill_rate` using the exact same formulas.

---

## 4. Issues Found

### 🔴 Issue 1: Oracle Demand Generation Uses a Different RNG Object

In [evaluate_oracle L48](file:///Users/sanamimani/Desktop/NetworkInvLog-BenchmarkTest/benchmark.py#L48), the demand trace is generated with:
```python
rng = np.random.default_rng(seed=seed)
```
This is a **separate** RNG from what `env.reset(seed=seed)` creates inside the environment. While both use the same seed, the Oracle's warm-up loop calls `dist.rvs(mu=mu, random_state=rng)` in a simple for-loop, whereas the environment's `_STEP` method calls `dist.rvs(mu=mu, random_state=self.np_random)`. Since both start from the **same seed value**, they should produce the same sequence — **but only if the RNG state advances identically**. If the environment's RNG is consumed elsewhere (e.g., `action_space.seed()` or `observation_space.seed()` in `reset()`), the sequences would **diverge**, meaning the Oracle would plan against a different demand path than what actually occurs.

> [!WARNING]
> **Impact**: The Oracle may be optimizing against a demand trace that doesn't match the actual simulation demand. This would make it either over- or under-perform vs the true optimum. Review the results CSV — if `Oracle_FillRate < 1.0` in scenarios without goodwill, this bug is confirmed (the Oracle should achieve 100% fill with perfect foresight and sufficient capacity).

**Evidence from results**: In **seasonal** scenarios without goodwill, the Oracle shows `fill_rate = 0.659` (base, backlog=True) and `0.866` (base, backlog=False). This confirms the demand-path mismatch — the Oracle is not seeing the actual realised demand.

> [!IMPORTANT]
> **Fix**: Instead of generating a separate demand trace, the benchmark should run the environment once with a dummy agent (or zero actions) to extract the actual `env.D` array, then feed that to the Oracle.

### 🟡 Issue 2: Oracle Gets Fixed Demand While Other Agents Face Stochastic Demand

The Oracle receives `user_D = demand_trace` which hard-codes the demand into the environment, while all other agents face **stochastic** Poisson-sampled demand. Even if the traces matched perfectly (see Issue 1), the Oracle's environment literally bypasses sampling — it uses deterministic demand. This is acceptable for an *upper bound* but means:
- The Oracle's profit is the **deterministic optimum** for one specific demand realization
- Other agents face the **stochastic** version of that same seed

This is methodologically standard for a "Value of Perfect Information" (VPI) bound, but should be clearly documented.

### 🟡 Issue 3: Goodwill Coupling Creates a Subtle Oracle Bias

When `use_goodwill=True`, the Oracle generates its demand trace assuming `sentiment = 1.0` throughout (the warm-up loop doesn't simulate fulfillment, so goodwill never changes). But during the actual simulation with the Oracle's actions, fulfillment quality affects sentiment, which then modifies demand via the demand engine. However, since `user_D` is hard-coded, the environment ignores the demand engine and uses the fixed trace anyway ([_STEP L519-520](file:///Users/sanamimani/Desktop/NetworkInvLog-BenchmarkTest/environment.py#L519-L520)).

This means: in goodwill scenarios, the Oracle's environment doesn't reflect goodwill dynamics at all. The Oracle's profit in goodwill vs non-goodwill scenarios will be **identical** (since `user_D` overrides everything).

**Evidence from results**: `Oracle_Profit` for `(base, stationary, goodwill=True)` = `$846.57`, exactly equal to `(base, stationary, goodwill=False)`. This confirms the Oracle effectively bypasses goodwill.

> [!NOTE]
> This is acceptable if the Oracle is labelled as a "relaxed upper bound." It means VPI in goodwill scenarios is measured against a relaxed bound, not the true-goodwill optimum.

### 🟡 Issue 4: DLP and MSSP Planning Horizons Differ

| Agent | Planning Horizon |
|-------|-----------------|
| Oracle | 30 (full episode) |
| DLP | **10** (rolling) |
| MSSP | **10** (rolling, 3-depth branching) |
| Heuristic | N/A (myopic) |

DLP and MSSP both use `planning_horizon=10` while the episode is 30 periods. This is standard for rolling-horizon methods, but it means their performance is partly limited by **planning myopia**. This is fair as long as it's documented — both OR agents use the same horizon.

### 🟢 Issue 5: Informed vs Blind Distinction Is Clean

All three adaptive agents (DLP, MSSP, Heuristic) have consistent `is_blind` semantics:
- **Informed (is_blind=False)**: calls `env.demand_engine.get_current_mu(t)` to get the expected demand mean at any future time. This is **not** cheating — it exposes the *parametric model* (trend/seasonal structure), not the actual realization.
- **Blind (is_blind=True)**: estimates μ from a 5-period SMA of historical demand. No future information.

### 🟢 Summary — Is it Fair?

| Criterion | Verdict |
|-----------|---------|
| Same environment instance per scenario | ✅ Fair |
| Same seed → same demand per episode | ⚠️ Mostly (Oracle RNG divergence risk) |
| Same metrics computed identically | ✅ Fair |
| Same planning horizon for comparable agents | ✅ Fair (DLP=MSSP=10) |
| Informed agents access same info level | ✅ Fair |
| Blind agents access same info level | ✅ Fair |
| Oracle is valid upper bound | ⚠️ Partially (demand mismatch in seasonal/goodwill) |
| No agent reads `env.D` *future* entries | ✅ Fair |

---

## 5. Recommended Fixes

1. **Fix Oracle demand trace generation** — Run the environment once with the actual seed and a do-nothing policy, extract `env.D`, and use that as the Oracle's `known_demand_scenario`. This eliminates RNG divergence.
2. **Document the goodwill relaxation** — Clearly note that the Oracle upper bound is a "Perfect Information Relaxation" that ignores goodwill dynamics.
3. **Consider increasing max_episodes** — Currently set to 2 (`benchmark.py L406`). For publication-quality results, 30-50 episodes would give tighter confidence intervals.
4. **Add variance/std columns** — Currently only means are reported. Standard deviations across episodes would help assess statistical significance.
