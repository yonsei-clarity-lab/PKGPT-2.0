"""
Phase 5: Covariate Analysis
Focus: Implement the ONE directed covariate test for this iteration.
"""

from typing import Dict, List, Optional


class Phase5Covariates:
    """Phase 5 prompts for systematic covariate analysis"""

    @staticmethod
    def generate_prompt(
        iteration: int,
        current_code: str,
        parsed_results: Dict,
        shrinkage_data: List[Dict],
        available_covariates: List[Dict],
        current_covariates_in_model: List[str],
        n_subjects: int
    ) -> str:
        """
        Generate Phase 5 prompt for implementing the single directed covariate.

        available_covariates contains EXACTLY ONE entry: the target for this iteration.
        Do NOT select from multiple options — implement what is given.
        """

        ofv = parsed_results.get('objective_function', 'N/A')
        minimization_ok = parsed_results.get('minimization_successful', False)

        # Extract the single target covariate
        if available_covariates:
            tgt = available_covariates[0]
            cov_name   = tgt.get('name', '?')
            cov_type   = tgt.get('type', 'continuous')
            cov_median = tgt.get('median', '?')
            cov_min    = tgt.get('min', '?')
            cov_max    = tgt.get('max', '?')
            parameter  = tgt.get('parameter', 'CL')   # 'CL' or 'V1'
            model_type = tgt.get('suggested_model', 'power' if cov_type == 'continuous' else 'categorical')
        else:
            cov_name   = 'UNKNOWN'
            cov_type   = 'continuous'
            cov_median = '?'
            cov_min    = '?'
            cov_max    = '?'
            parameter  = 'CL'
            model_type = 'power'

        # TV parameter name (TVCL, TVV1, etc.)
        tv_param  = f'TV{parameter}'          # e.g. TVCL, TVV1
        theta_idx = '1' if parameter == 'CL' else '2'  # 일반적 THETA 인덱스 힌트

        # Confirmed covariates from prior rounds (already baked into base code)
        current_cov_text = (
            f"Confirmed from prior rounds (already in base code): {', '.join(current_covariates_in_model)}"
            if current_covariates_in_model
            else "No covariates in model yet (clean base model)"
        )

        # Shrinkage
        if shrinkage_data:
            max_shrink = max(s['shrinkage'] for s in shrinkage_data)
            shrinkage_text = f"Max ETA shrinkage: {max_shrink:.1f}% {'✓' if max_shrink < 50 else '⚠ High'}"
        else:
            shrinkage_text = "Shrinkage: N/A"

        # Build implementation template — tv_param/parameter are dynamic, no hardcoding
        if cov_type == 'categorical':
            impl_template = f"""Covariate type: CATEGORICAL — target parameter: {parameter}
Coding: indicator/factor, multiplicative, no centering needed.
Reference category = 0, non-reference = 1.

Example ({cov_name} on {parameter}, reference = category 0):
  {tv_param} = THETA({theta_idx})
  IF ({cov_name}.EQ.1) {tv_param} = {tv_param} * THETA(N)

New THETA(N): initial = 1.0 (no effect), bounds = (0.5, 2.0)
"""
        elif model_type == 'power':
            impl_template = f"""Covariate type: CONTINUOUS — Power model — target parameter: {parameter}
Centering: {cov_name} / {cov_median}  (median = {cov_median}, range {cov_min}–{cov_max})

Example ({cov_name} on {parameter}):
  {tv_param} = THETA({theta_idx}) * ({cov_name}/{cov_median})**THETA(N)

New THETA(N): initial = 0.75 if CL-related, 1.0 if V-related
Bounds: (0, initial, 2.0) — use 0 as lower bound, NOT 0.5 (avoids boundary failures)
CRITICAL: Power model guarantees {parameter} > 0 for all WT values — do NOT use linear form.
"""
        else:  # linear
            impl_template = f"""Covariate type: CONTINUOUS — Linear model — target parameter: {parameter}
Centering: {cov_name} - {cov_median}  (median = {cov_median}, range {cov_min}–{cov_max})

Example ({cov_name} on {parameter}):
  {tv_param} = THETA({theta_idx}) * (1 + THETA(N) * ({cov_name} - {cov_median}))

New THETA(N): initial = 0.001 (must be non-zero; NONMEM rejects 0 initial for non-fixed THETAs)
Bounds: (-0.1, 0.001, 0.1)
"""

        prompt = f"""You are implementing ONE directed covariate test in NONMEM.

═══════════════════════════════════════════════════════════════════
PHASE 5: IMPLEMENT DIRECTED COVARIATE — DO NOT DEVIATE
═══════════════════════════════════════════════════════════════════

Iteration   : {iteration}
Subjects    : N={n_subjects}
Base OFV    : {ofv}
{shrinkage_text}
{current_cov_text}

YOUR ONLY TASK THIS ITERATION:
  ► Add covariate  : {cov_name}
  ► Target param   : {parameter}  (modify TV{parameter} in $PK)
  ► Covariate type : {cov_type}
  ► Suggested form : {model_type}

DO NOT add any other covariate.
DO NOT substitute a different covariate or a different parameter.
The SCM STATUS block above specifies exactly what to test — follow it.

═══════════════════════════════════════════════════════════════════
IMPLEMENTATION GUIDE FOR {cov_name}
═══════════════════════════════════════════════════════════════════

{impl_template}

ABSOLUTE RULES (cannot be overridden):
  ❌ DO NOT change $ESTIMATION (keep FOCE-I / METHOD=1 INTER)
  ❌ DO NOT change $ERROR or $SIGMA
  ❌ DO NOT change $OMEGA structure
  ❌ DO NOT change ADVAN or compartment count
  ❌ DO NOT add any covariate other than {cov_name}
  ❌ DO NOT apply {cov_name} to any parameter other than {parameter}
  ❌ DO NOT remove or modify any existing covariate relationships in the base code
  ✅ Modify TV{parameter} in $PK to include {cov_name}
  ✅ Add exactly ONE new THETA for the {cov_name} effect
  ✅ Append the new THETA at the END of the $THETA block — do NOT insert in the middle
     (Inserting in the middle shifts existing THETA indices and breaks $ERROR and $PK)
  ✅ Keep all other model components identical to the base model

═══════════════════════════════════════════════════════════════════
CURRENT MODEL (modify this)
═══════════════════════════════════════════════════════════════════

```
{current_code}
```

═══════════════════════════════════════════════════════════════════
RESPONSE FORMAT
═══════════════════════════════════════════════════════════════════

IMPROVED CODE:
```
[Complete NONMEM control stream with {cov_name} added]
```

CHANGES MADE:
- Added covariate: {cov_name} on [parameter as directed by SCM STATUS]
- Model type: {model_type}
- New THETA: [initial estimate and bounds]
- Centering: {f'{cov_name}/{cov_median}' if model_type == 'power' else (f'{cov_name} - {cov_median}' if cov_type == 'continuous' else 'N/A (categorical)')}
"""
        return prompt

    @staticmethod
    def generate_removal_prompt(
        iteration: int,
        current_code: str,
        parsed_results: Dict,
        target_covariate_name: str,
        remaining_covariates: List[str],
        n_subjects: int
    ) -> str:
        """
        Generate a backward-elimination prompt: remove ONE specific covariate
        term from the current (full) model to test whether it is still
        statistically significant now that other covariates are already present.

        target_covariate_name is e.g. "SEX on CL" — parsed into covariate/parameter.
        """
        ofv = parsed_results.get('objective_function', 'N/A')

        parts = target_covariate_name.split(' on ')
        cov_name = parts[0] if len(parts) == 2 else target_covariate_name
        parameter = parts[1] if len(parts) == 2 else 'CL'

        others = [c for c in remaining_covariates if c != target_covariate_name]
        remaining_text = ', '.join(others) if others else 'None (this is the last remaining covariate)'

        prompt = f"""You are performing BACKWARD ELIMINATION in NONMEM SCM.

═══════════════════════════════════════════════════════════════════
BACKWARD ELIMINATION: TEST REMOVAL OF ONE COVARIATE — DO NOT DEVIATE
═══════════════════════════════════════════════════════════════════

Iteration   : {iteration}
Subjects    : N={n_subjects}
Full model OFV (with ALL current covariates): {ofv}
Other covariates that MUST remain unchanged: {remaining_text}

YOUR ONLY TASK THIS ITERATION:
  ► Remove covariate : {cov_name}
  ► From parameter   : {parameter}  (revert TV{parameter} in $PK to its pre-covariate form)

WHY: This is a diagnostic test, not a final decision. {cov_name} was accepted during
forward selection, but that test was against a base model that may not have included
all of today's other covariates. Some of {cov_name}'s apparent effect may actually be
explained by collinearity with another covariate now in the model (e.g. two correlated
markers of the same underlying physiology). Backward elimination re-tests {cov_name}
against the FULL model using a stricter significance threshold (p<0.01 instead of
p<0.05) to catch covariates that only looked significant due to this effect.

DO NOT remove or modify any OTHER covariate ({remaining_text}).
DO NOT change $ESTIMATION, $ERROR, $SIGMA, ADVAN, compartment structure, or $OMEGA.

═══════════════════════════════════════════════════════════════════
HOW TO REMOVE {cov_name} FROM {parameter}
═══════════════════════════════════════════════════════════════════

1. In $PK, revert TV{parameter} to NOT include the {cov_name} term — restore it to
   plain TV{parameter} = THETA(k) (remove the (COV/median)**THETA(N) or
   (1 + THETA(N)*(COV-median)) or IF(...) THETA(N) multiplier, whichever form is used).
2. Delete the $THETA line for the {cov_name}-on-{parameter} effect.
3. RENUMBER every remaining THETA index so they stay contiguous (no gaps) — update
   every THETA(n) reference in $PK and $ERROR (including any OTHER covariate terms,
   e.g. {remaining_text}) to match the new numbering.
4. Every other covariate term must remain functionally identical — only its THETA
   index may shift due to the renumbering.

═══════════════════════════════════════════════════════════════════
CURRENT MODEL (modify this — remove ONLY {cov_name} on {parameter})
═══════════════════════════════════════════════════════════════════

```
{current_code}
```

═══════════════════════════════════════════════════════════════════
RESPONSE FORMAT
═══════════════════════════════════════════════════════════════════

IMPROVED CODE:
```
[Complete NONMEM control stream with {cov_name} on {parameter} removed, THETA renumbered]
```

CHANGES MADE:
- Removed covariate: {cov_name} on {parameter}
- THETA renumbering: [describe index shifts, if any]
"""
        return prompt
