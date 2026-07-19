"""
Phase 2: Diagnose Structural Model
Focus: Determine if compartment structure and error model are adequate
"""

from typing import Dict, List, Optional


class Phase2Diagnose:
    """Phase 2 prompts for diagnosing structural model adequacy"""

    @staticmethod
    def generate_prompt(
        iteration: int,
        current_code: str,
        nonmem_output: str,
        parsed_results: Dict,
        warnings: List[str],
        n_subjects: int,
        plausibility_bounds: Optional[Dict] = None
    ) -> str:
        """
        Generate Phase 2 prompt focused on structural diagnosis

        Goals:
        - Check if compartment structure is adequate
        - Diagnose residual patterns
        - Assess error model appropriateness
        - Decide if structural changes needed
        """

        ofv = parsed_results.get('objective_function', 'N/A')
        minimization_ok = parsed_results.get('minimization_successful', False)
        warnings_text = "\n".join([f"- {w}" for w in warnings]) if warnings else "None"

        # Extract current ADVAN from code
        current_advan = "Unknown"
        if "ADVAN1" in current_code:
            current_advan = "ADVAN1 (IV, 1-compartment)"
        elif "ADVAN2" in current_code:
            current_advan = "ADVAN2 (Oral, 1-compartment)"
        elif "ADVAN3" in current_code:
            current_advan = "ADVAN3 (IV, 2-compartment)"
        elif "ADVAN4" in current_code:
            current_advan = "ADVAN4 (Oral, 2-compartment)"

        # Literature/LLM-derived typical values from the very start of the run
        # (see optimizer.py _build_plausibility_bounds). Used below to stop a
        # specific failure mode: re-seeding a boundary-collapsed final estimate
        # as the next iteration's initial value, which drags the optimizer
        # further into the same degenerate corner instead of resetting to a
        # sane starting point (observed on a sparse 2-compartment IV dataset,
        # where Q drifted to its upper bound and V2 to its lower bound after
        # several rounds of inheriting near-boundary final estimates).
        typical_values_section = ""
        if plausibility_bounds and plausibility_bounds.get("parameters"):
            lines = []
            for pname, pvals in plausibility_bounds["parameters"].items():
                typical = pvals.get("typical")
                if typical is None:
                    continue
                unit = pvals.get("unit", "")
                lines.append(f"  - {pname}: typical={typical} {unit} (plausible range {pvals.get('min')}-{pvals.get('max')} {unit})")
            if lines:
                typical_values_section = f"""
ORIGINAL PHYSIOLOGICAL REFERENCE VALUES (from the start of this run):
{chr(10).join(lines)}
"""

        prompt = f"""You are diagnosing structural model adequacy.

═══════════════════════════════════════════════════════════════════
PHASE 2: DIAGNOSE STRUCTURAL MODEL
═══════════════════════════════════════════════════════════════════

ITERATION: {iteration}
DATASET: N={n_subjects} subjects
CURRENT STRUCTURE: {current_advan}
MINIMIZATION: {'✓ Successful' if minimization_ok else '✗ Failed'}
OFV: {ofv}
{typical_values_section}
CURRENT MODEL:
```
{current_code}
```

NONMEM WARNINGS:
{warnings_text}

═══════════════════════════════════════════════════════════════════
DIAGNOSTIC STRATEGY: Is the Structure Adequate?
═══════════════════════════════════════════════════════════════════

**Check 1: Residual Pattern Analysis**

Look for these patterns in NONMEM warnings/output:

1. **U-shaped or Systematic Residual Trends**
   - Indicates: Wrong number of compartments
   - U-shape in CWRES vs TIME → Need more compartments
   - Action: Consider ADVAN2→ADVAN4 or ADVAN1→ADVAN3

2. **Bi-phasic or Multi-phasic Decline**
   - Indicates: Distribution phase present
   - 1-compartment insufficient for drug with distribution
   - Action: Upgrade to 2-compartment model

3. **Funnel-shaped Residuals (CWRES vs PRED)**
   - Indicates: Heteroscedastic errors
   - Narrow at low concentrations, wide at high
   - Action: Ensure proportional error component in error model

**Check 2: OFV Magnitude**

- OFV > 0 and reasonable: Likely OK
- OFV extremely negative (<-50): Possible structural issue or overfitting
- OFV extremely high (>5000 for N<50): Poor fit, structure may be wrong

**Check 3: Warnings Analysis**

Critical warnings and their interpretation:
- "PARAMETER ESTIMATE IS NEAR ITS BOUNDARY" → First suspect: over-parameterized IIV
  (a parameter hitting its boundary often means the data cannot support that random effect,
   NOT that the compartment structure is wrong)
- "PRED.LE.0" or "NUMERICAL HESSIAN" → Structural misspecification
- Systematic mentions of residual patterns → Structure problem

**Check 4: Error Model Adequacy**

Current error model type (check $ERROR block):
- Proportional only: Y = IPRED * (1 + EPS(1))
- Combined (better): W = SQRT(THETA^2 + (THETA*IPRED)^2), Y = IPRED + W*EPS(1)
- Additive only: Y = IPRED + EPS(1)

If proportional-only and low concentrations exist → Consider combined model

═══════════════════════════════════════════════════════════════════
DECISION CRITERIA — PRIORITY-ORDERED ACTIONS
═══════════════════════════════════════════════════════════════════

When estimation fails (boundary hit, covariance failure, minimization failure),
apply fixes in order of LEAST DESTRUCTIVE first. Never skip to a more
destructive action before exhausting less destructive ones.

**Priority 1 — Adjust initial estimates and bounds (least destructive):**
- Widen or shift $THETA bounds away from boundary
- Improve initial estimates using prior iteration's final values
- Switch BLOCK OMEGA → DIAGONAL (reduce correlation parameters)
→ TRY THIS FIRST for any boundary/convergence issue

⚠️ EXCEPTION — do NOT reuse a boundary-collapsed final estimate as the next
initial estimate:
  If the PRIOR iteration's final value for a parameter was itself at/near
  that parameter's bound (the boundary/covariance failure you are now fixing),
  that value is not a better starting point — it is the failure. Carrying it
  forward (even "nudged" slightly away from the bound) seeds the next fit from
  inside the same degenerate region and typically drags the optimizer further
  into it over successive iterations, rather than escaping it.
  Instead, RESET that parameter's initial estimate to its ORIGINAL physiological
  reference value (see "ORIGINAL PHYSIOLOGICAL REFERENCE VALUES" above if
  present, otherwise the base model's original initial estimate before any
  boundary issue occurred) — not the collapsed value, and not merely a small
  step away from it.
  This applies to EVERY parameter whose final estimate was at/near a bound,
  not just the one currently flagged.

**Priority 2 — Simplify IIV to stabilize estimation (diagnostic):**
- If boundary/covariance failure persists after Priority 1
- Remove ETA from the LEAST identifiable parameter first
  (peripheral volume V2 and inter-compartmental clearance Q are typically
   least identifiable with sparse data; clearance CL is most identifiable)
- Keep at minimum 1 ETA (on CL) — never remove the last random effect
- This is a DIAGNOSTIC action: simplifying IIV to assess whether the
  structural model itself is sound. It does NOT change the compartment structure.
→ PRESERVE the current ADVAN/compartment count while simplifying IIV

**Priority 3 — Improve error model (moderate):**
- Funnel-shaped residuals → switch to combined error model
- Proportional-only failing at low concentrations → add additive component
- Combined model with additive component near zero → simplify to proportional
→ Change error model structure

**Priority 4 — Change compartment structure (most destructive — last resort):**
ONLY consider this when ALL of the following are true:
  (a) IIV is already minimal (1 ETA on CL only)
  (b) Error model has been assessed
  (c) Clear pharmacokinetic evidence for structural change exists
      (U-shaped residuals, bi-phasic decline, systematic misfit)
→ ADVAN2→ADVAN4 or ADVAN1→ADVAN3 (upgrade)
→ ADVAN3→ADVAN1 or ADVAN4→ADVAN2 (downgrade) ONLY if identifiability
  is impossible even with minimal IIV and adequate data

**Keep Current Structure IF:**
✓ Minimization successful
✓ No systematic residual trends mentioned
✓ OFV reasonable (not extremely negative)
✓ No warnings about structural misspecification
→ ACTION: Make minor refinements only (bounds, error model)

**DO NOT Change Compartment Count IF:**
- IIV has not yet been simplified (try removing ETAs first)
- Only boundary/covariance issues exist (these are usually IIV problems)
- N < 20 (insufficient data for complex models)
- Current structure works after IIV simplification

═══════════════════════════════════════════════════════════════════
IIV SIMPLIFICATION TEMPLATES (Priority 2 — try before structural changes)
═══════════════════════════════════════════════════════════════════

**Remove least identifiable ETA while keeping structure:**

Step 1: BLOCK → DIAGONAL (if currently BLOCK)
```
; Before: $OMEGA BLOCK(3)
; After:
$OMEGA DIAGONAL(3)
0.09  ; IIV on CL
0.09  ; IIV on V1
0.09  ; IIV on V2
```

Step 2: Remove ETA from least identifiable parameter
Removal order (least → most identifiable): V2 → Q → V1 → CL
```
; Before (3 ETAs):
CL = TVCL * EXP(ETA(1))
V1 = TVV1 * EXP(ETA(2))
V2 = TVV2 * EXP(ETA(3))
$OMEGA DIAGONAL(3)

; After (removed V2 ETA):
CL = TVCL * EXP(ETA(1))
V1 = TVV1 * EXP(ETA(2))
V2 = TVV2
$OMEGA DIAGONAL(2)
```

IMPORTANT: When removing an ETA, also remove its $OMEGA entry and
update $TABLE to match the remaining ETAs.

═══════════════════════════════════════════════════════════════════
STRUCTURAL MODEL UPGRADE TEMPLATES (Priority 4 — last resort only)
═══════════════════════════════════════════════════════════════════

⚠️  Do NOT use these unless IIV is already minimal (1 ETA on CL only)
    AND the structural model still shows clear misfit.

**Oral 1-cmt → 2-cmt (ADVAN2 → ADVAN4):**
```
$SUBROUTINE ADVAN4 TRANS4

$PK
CL = THETA(1) * EXP(ETA(1))
V1 = THETA(2) * EXP(ETA(2))
Q  = THETA(3)
V2 = THETA(4)
KA = THETA(5) * EXP(ETA(3))
S2 = V1

$THETA
(0.1, 3, 100)    ; CL (L/h)
(1, 30, 200)     ; V1 (L)
(0.1, 5, 50)     ; Q (L/h) - inter-compartmental clearance
(1, 50, 500)     ; V2 (L) - peripheral volume
(0.1, 1, 10)     ; KA (1/h)
```

**IV 1-cmt → 2-cmt (ADVAN1 → ADVAN3):**
```
$SUBROUTINE ADVAN3 TRANS4

$PK
CL = THETA(1) * EXP(ETA(1))
V1 = THETA(2) * EXP(ETA(2))
Q  = THETA(3)
V2 = THETA(4)
S1 = V1

$THETA
(0.1, 3, 100)    ; CL
(1, 30, 200)     ; V1
(0.1, 5, 50)     ; Q
(1, 50, 500)     ; V2
```

═══════════════════════════════════════════════════════════════════
RESPONSE FORMAT
═══════════════════════════════════════════════════════════════════

STRUCTURAL DIAGNOSIS:
[Is the current compartment model adequate? Cite specific evidence from output]

PRIORITY ACTION CHOSEN:
[Which priority level (1-4) are you applying and why?
 If choosing Priority 3 or 4, explain why lower priorities were insufficient.]

RECOMMENDATION:
[Specific action: adjust bounds / simplify IIV / change error model / change structure]

IMPROVED CODE:
```
[Complete NONMEM control stream - with structural changes if needed]
```

CHANGES MADE:
[List specific modifications and rationale]

EXPECTED OUTCOME:
[Better fit if structure changed, or confirmation that structure is adequate]
"""

        return prompt
