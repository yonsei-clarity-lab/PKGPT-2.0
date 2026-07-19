"""
Phase 1: Establish Base Model
Focus: Fix syntax errors and get NONMEM running successfully
"""

from typing import Dict, List, Optional


class Phase1Establish:
    """Phase 1 prompts for establishing a working base model"""

    @staticmethod
    def generate_prompt(
        iteration: int,
        current_code: str,
        nonmem_output: str,
        issues_found: List[str],
        history: List[Dict],
        n_subjects: int
    ) -> str:
        """
        Generate Phase 1 prompt focused on fixing syntax and execution errors

        Goals:
        - Fix NONMEM syntax errors
        - Fix parameter boundary issues
        - Get minimization to succeed
        - Establish working base model
        """

        issues_text = "\n".join([f"- {issue}" for issue in issues_found]) if issues_found else "None specified"

        # Format history
        history_text = ""
        if history:
            history_text = "\nPrevious attempts:\n"
            for h in history[-3:]:  # Last 3 only
                iter_num = h.get('iteration', '?')
                ofv = h.get('ofv', 'N/A')
                status = h.get('status', 'Unknown')
                history_text += f"  Iteration {iter_num}: OFV={ofv}, Status={status}\n"

        prompt = f"""You are fixing NONMEM syntax and execution errors.

═══════════════════════════════════════════════════════════════════
PHASE 1: ESTABLISH BASE MODEL - Get NONMEM Running
═══════════════════════════════════════════════════════════════════

ITERATION: {iteration}
DATASET: N={n_subjects} subjects
GOAL: Fix errors and achieve successful minimization

CURRENT MODEL:
```
{current_code}
```

NONMEM OUTPUT:
```
{nonmem_output}
```

ISSUES IDENTIFIED:
{issues_text}
{history_text}

═══════════════════════════════════════════════════════════════════
FOCUSED STRATEGY: Fix One Thing at a Time
═══════════════════════════════════════════════════════════════════

**Priority 1: Syntax Errors**

Common ADVAN-specific issues:
- ADVAN2 (oral, 1-cmt): MUST have K=CL/V and S2=V in $PK
- ADVAN1 (IV, 1-cmt): MUST have K=CL/V and S1=V in $PK
- ADVAN4 (oral, 2-cmt): S2=V1 in $PK
- ADVAN3 (IV, 2-cmt): S1=V1 in $PK

$INPUT column order:
- MUST match CSV file EXACTLY
- Example: If CSV has "ID,TIME,AMT,DV,EVID,MDV,WT"
  Then: $INPUT ID TIME AMT DV EVID MDV WT

Semicolon requirements:
- EVERY THETA, OMEGA, SIGMA needs comment with semicolon
- Example: THETA(1) ; CL (L/h)

**Priority 2: Parameter Boundaries**

THETA bounds (lower, initial, upper):
- Lower bound: Prevent collapse to zero
- Initial: Physiologically plausible starting value
- Upper bound: Prevent unrealistic explosion

Example fixes:
```
; If CL estimate hitting lower bound (0.1)
$THETA
(0.001, 3, 100)  ; CL - widened lower bound

; If parameter hitting upper bound
$THETA
(0.1, 3, 500)    ; CL - widened upper bound

; If parameter oscillating - fix it temporarily
$THETA
3 FIX            ; CL - fixed to typical value
```

**Priority 3: Estimation Method**

If convergence failing:
1. Try METHOD=ZERO (First Order - most robust)
   ```
   $ESTIMATION METHOD=ZERO MAXEVAL=9999
   ```

2. If that works, later try METHOD=1 INTER (FOCE-I)
   ```
   $ESTIMATION METHOD=1 INTER MAXEVAL=9999
   ```

3. Avoid METHOD=1 without INTER for small N<20

**Priority 4: OMEGA Structure**

For N<20 subjects:
- Use DIAGONAL OMEGA only (no BLOCK)
- Start with 1-2 OMEGAs maximum
- Example for small N:
  ```
  $OMEGA
  0.1  ; IIV on CL only
  ```

For N>20:
- Can use 2-3 OMEGAs DIAGONAL
- Example:
  ```
  $OMEGA
  0.1  ; IIV on CL
  0.1  ; IIV on V
  0.1  ; IIV on KA
  ```

**FORBIDDEN in Phase 1:**
❌ Changing structural model (ADVAN2→ADVAN4)
❌ Adding covariates
❌ Complex error models
❌ Multiple changes at once

**ALLOWED in Phase 1:**
✅ Fix K=CL/V, S1=V, S2=V
✅ Fix $INPUT order
✅ Adjust THETA bounds (lower, initial, upper)
✅ Simplify to METHOD=ZERO
✅ Reduce OMEGA count to 1-2
✅ Fix problematic THETA values

═══════════════════════════════════════════════════════════════════
RESPONSE FORMAT
═══════════════════════════════════════════════════════════════════

DIAGNOSIS:
[1-2 sentences: What is the main error preventing execution?]

SOLUTION:
[1-2 sentences: What specific fix will be applied?]

IMPROVED CODE:
```
[Complete corrected NONMEM control stream]
```

CHANGES MADE:
[Bullet list of specific modifications]

EXPECTED OUTCOME:
[What should happen: "Minimization successful" or "Error fixed, ready for next iteration"]
"""

        return prompt
