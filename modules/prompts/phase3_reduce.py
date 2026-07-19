"""
Phase 3: Reduce Overfitting
Focus: Simplify model by removing uninformative parameters
"""

from typing import Dict, List, Optional


class Phase3Reduce:
    """Phase 3 prompts for reducing model overfitting"""

    @staticmethod
    def generate_prompt(
        iteration: int,
        current_code: str,
        parsed_results: Dict,
        shrinkage_data: List[Dict],
        current_omega_count: int,
        n_subjects: int
    ) -> str:
        """
        Generate Phase 3 prompt focused on reducing overfitting

        Goals:
        - Remove OMEGA parameters with shrinkage > 90%
        - Simplify error model if needed
        - Maintain at least 1 OMEGA for population modeling
        - Improve model parsimony
        """

        ofv = parsed_results.get('objective_function', 'N/A')
        avg_shrink = max([s['shrinkage'] for s in shrinkage_data]) if shrinkage_data else 0

        # Format shrinkage by ETA
        shrinkage_text = ""
        if shrinkage_data:
            shrinkage_text = "ETA Shrinkage:\n"
            for s in shrinkage_data:
                eta_num = s.get('eta_number', '?')
                shrink = s.get('shrinkage', 0)
                status = "✓ OK" if shrink < 50 else ("⚠ High" if shrink < 90 else "✗ Critical")
                shrinkage_text += f"  ETA({eta_num}): {shrink:.1f}% {status}\n"
        else:
            shrinkage_text = "Shrinkage data not available"

        # Critical warning if only 1 OMEGA
        omega_warning = ""
        if current_omega_count == 1:
            omega_warning = f"""
{'='*70}
🛑 CRITICAL: ONLY 1 OMEGA REMAINS - MUST PROTECT IT! 🛑
{'='*70}

Current model has ONLY 1 OMEGA parameter.

**YOU MUST NOT REMOVE IT!**

Why:
- Removing it → Becomes non-population model
- Cannot use population methods (FOCE, etc.)
- For N={n_subjects}: Shrinkage <75% with 1 OMEGA is ACCEPTABLE

What you CAN do:
✅ Adjust THETA bounds
✅ Simplify error model
✅ Fix THETA values to typical values
✅ Change estimation method

What you CANNOT do:
❌ Remove last OMEGA
❌ Delete $OMEGA block

Instead, focus on other refinements.
{'='*70}
"""

        prompt = f"""You are reducing model overfitting by removing uninformative parameters.

═══════════════════════════════════════════════════════════════════
PHASE 3: REDUCE OVERFITTING - Simplify the Model
═══════════════════════════════════════════════════════════════════

ITERATION: {iteration}
DATASET: N={n_subjects} subjects
CURRENT OMEGA COUNT: {current_omega_count}
MAX SHRINKAGE: {avg_shrink:.1f}%
OFV: {ofv}

{omega_warning}

CURRENT MODEL:
```
{current_code}
```

{shrinkage_text}

═══════════════════════════════════════════════════════════════════
OVERFITTING DETECTION CRITERIA
═══════════════════════════════════════════════════════════════════

**Strong Overfitting Indicators:**

1. **Extremely High Shrinkage (>90%)**
   - ETAs are uninformative
   - Not learning individual parameters from data
   - Action: Remove that OMEGA (but keep ≥1)

2. **Extremely Negative OFV (<-50)**
   - Model fitting noise rather than signal
   - Often happens with small datasets
   - Action: Simplify random effects or error model

3. **OMEGA Collapsed (<0.001)**
   - Parameter variance numerically zero
   - Serving no purpose in model
   - Action: Remove that OMEGA

4. **Covariance Step Failure + High Shrinkage**
   - Model overparameterized for dataset
   - Cannot compute standard errors
   - Action: Reduce complexity

═══════════════════════════════════════════════════════════════════
SIMPLIFICATION STRATEGY
═══════════════════════════════════════════════════════════════════

**Priority 1: Remove 1 OMEGA (if count > 1)**

Decision tree:
```
IF current_omega_count > 1 AND any_shrinkage > 90%:
    → Remove OMEGA with highest shrinkage

ELSE IF current_omega_count > 2 AND avg_shrinkage > 70%:
    → Remove 1 OMEGA (lowest impact parameter)

ELSE IF current_omega_count == 1:
    → DO NOT REMOVE! Keep that OMEGA
    → Focus on other improvements instead
```

Example - Removing IIV on KA:
```
; BEFORE
$PK
CL = THETA(1) * EXP(ETA(1))
V  = THETA(2) * EXP(ETA(2))
KA = THETA(3) * EXP(ETA(3))  ; Has 95% shrinkage!

$OMEGA
0.1  ; CL
0.1  ; V
0.1  ; KA (remove this)

; AFTER
$PK
CL = THETA(1) * EXP(ETA(1))
V  = THETA(2) * EXP(ETA(2))
KA = THETA(3)  ; Fixed to typical value (no ETA)

$OMEGA
0.1  ; CL
0.1  ; V
```

**Priority 2: Simplify Error Model (if needed)**

If combined error model is unstable:
```
; BEFORE (Combined - complex)
$ERROR
W = SQRT(THETA(4)**2 + (THETA(5)*IPRED)**2)
Y = IPRED + W*EPS(1)

; AFTER (Proportional - simpler)
$ERROR
IPRED = F
Y = IPRED * (1 + EPS(1))

$SIGMA
0.04  ; Proportional error
```

**Priority 3: Fix Poorly Estimated THETA**

If parameter hitting boundaries or highly variable:
```
; BEFORE
$THETA
(0.1, 0.11, 100)  ; KA hitting lower bound

; AFTER
$THETA
1.0 FIX  ; KA fixed to typical value
```

**Priority 4: Change Estimation Method (if needed)**

If FOCE-I failing with small dataset:
```
; BEFORE
$ESTIMATION METHOD=1 INTER MAXEVAL=9999

; AFTER (more robust for small N)
$ESTIMATION METHOD=ZERO MAXEVAL=9999
```

═══════════════════════════════════════════════════════════════════
ACCEPTABLE SHRINKAGE BY DATASET SIZE
═══════════════════════════════════════════════════════════════════

For N < 20 subjects:
- 1 OMEGA: Shrinkage <75% is ACCEPTABLE
- 2 OMEGAs: Shrinkage <60% is GOOD
- >2 OMEGAs: Likely overfitting, reduce to 1-2

For N = 20-50 subjects:
- Shrinkage <50% is GOOD
- Shrinkage 50-70% is ACCEPTABLE
- Shrinkage >70% suggests removing that OMEGA

For N > 50 subjects:
- Shrinkage <30% is EXCELLENT
- Shrinkage <50% is GOOD
- Shrinkage >70% suggests problem

═══════════════════════════════════════════════════════════════════
RESPONSE FORMAT
═══════════════════════════════════════════════════════════════════

OVERFITTING DIAGNOSIS:
[Is the model overfitting? Which parameters are uninformative?]

SIMPLIFICATION PLAN:
[Which OMEGA/THETA to remove or fix? Why?]

IMPROVED CODE:
```
[Complete NONMEM control stream with simplified structure]
```

CHANGES MADE:
- Removed OMEGA({{omega_num}}): {{reason}}
- Other changes: {{list}}

VERIFICATION:
- New OMEGA count: {current_omega_count - 1} (decreased by 1)
- At least 1 OMEGA remains: ✓ YES

EXPECTED OUTCOME:
[More parsimonious model, better shrinkage, similar or slightly higher OFV]
"""

        return prompt
