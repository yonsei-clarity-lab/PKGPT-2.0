"""
Recursive NONMEM Optimization Engine
Iteratively generates and improves NONMEM control stream files

Phase-aware optimization following systematic pharmacometric model development
"""

import os
import re
import subprocess
from typing import Dict, List, Optional, Tuple
from enum import Enum
import time

from .openrouter_client import MultiModelOpenRouterClient
from .data_loader import PKDataLoader
from .prompt_templates import PromptTemplates
from .nonmem_parser import NONMEMParser
from .phase_transition_manager import PhaseTransitionManager
from .prompts.phase5_covariates import Phase5Covariates


class ModelPhase(Enum):
    """
    Systematic model development phases

    Following best practices in pharmacometric model building
    """
    ESTABLISH_BASE = 1       # Phase 1: Fix syntax/execution, get minimization working
    DIAGNOSE_STRUCTURE = 2   # Phase 2: Check structural model adequacy (compartments)
    REDUCE_OVERFITTING = 3   # Phase 3: Simplify model if overfitting detected
    OPTIMIZE_IIV = 4         # Phase 4: Fine-tune random effects structure
    COVARIATE_ANALYSIS = 5   # Phase 5: Add covariates to explain variability

    def __str__(self):
        phase_names = {
            ModelPhase.ESTABLISH_BASE: "Phase 1: Establish Base Model",
            ModelPhase.DIAGNOSE_STRUCTURE: "Phase 2: Diagnose Structure",
            ModelPhase.REDUCE_OVERFITTING: "Phase 3: Reduce Overfitting",
            ModelPhase.OPTIMIZE_IIV: "Phase 4: Optimize IIV",
            ModelPhase.COVARIATE_ANALYSIS: "Phase 5: Covariate Analysis"
        }
        return phase_names.get(self, "Unknown Phase")


class NONMEMOptimizer:
    """Recursive optimizer for NONMEM models"""

    def __init__(
        self,
        data_file: str,
        output_base: str,
        api_key: Optional[str] = None,
        min_iterations: int = 3,
        max_iterations: int = 20,
        nmfe_command: str = 'nmfe75',
        model: str = 'flash',
        prior_info: Optional[Dict] = None,
    ):
        """
        Initialize NONMEM optimizer

        Args:
            data_file: Path to input dataset
            output_base: Base name for output files (without extension)
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            min_iterations: Minimum number of optimization iterations
            max_iterations: Maximum number of optimization iterations
            nmfe_command: NONMEM execution command
            model: Model to use ('claude-sonnet', 'claude-opus', 'gemini-flash', 'gpt-4o')
            prior_info: Optional user-supplied prior information and SCM settings
        """
        self.data_file = data_file
        self.output_base = output_base
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.nmfe_command = nmfe_command
        self.model = model
        self.prior_info = prior_info or {}

        # Initialize components
        print("=" * 70)
        print("NONMEM RECURSIVE OPTIMIZER")
        print("=" * 70)

        self.data_loader = PKDataLoader(data_file)
        self.gemini_client = MultiModelOpenRouterClient(api_key)   
        
        # 변수명은 그대로 둬도 동작함
        self.gemini_client.current_model_type = model
        current_model = self.gemini_client.clients[model].get_current_model()
        print(f"Using model: {current_model}")
        self._print_covariate_selection_summary()

        # Optimization state
        self.iteration = 0
        self.current_code = None
        self.improvement_history = []
        self.last_ai_recommendations = []  # recommendations from last AI quality eval
        self.last_ai_critical_issues  = []  # critical issues from last AI quality eval
        self.best_ofv = None
        self.best_iteration = 0
        self.best_composite_score = float('inf')  # Lower is better
        self.best_code = None  # Store code from best iteration for revert
        self.code_history = []  # Store control stream code from each iteration

        # Phase-based optimization tracking
        self.current_phase = ModelPhase.ESTABLISH_BASE
        self.phase_history = []  # Track phase transitions
        self.iterations_in_phase = 0

        # Phase transition manager (cleaner state machine logic)
        self.phase_manager = None  # Will be initialized on first use

        # Failed strategy tracking to prevent infinite loops
        self.failed_strategies = []  # Track what we've tried that failed
        self.strategy_repeat_count = {}  # signature -> count, for repeated-failure reset
        self.plausibility_bounds = {}  # Drug-specific physiological bounds (built once)
        self.plausibility_report = {}   # Latest plausibility check result
        self.last_revert_info = None  # Revert 발생 시 LLM에 전달할 컨텍스트

        # SCM (Stepwise Covariate Modeling) tracking for Phase 5
        # Multi-round forward selection: ΔOFV < -3.84 (p<0.05, df=1) to accept
        # followed by backward elimination: ΔOFV < 6.63 (p<0.01, df=1) on removal
        # to eliminate (standard PsN-style forward/backward SCM combination).
        #
        # Algorithm:
        #   Forward round k: test ALL remaining candidates independently against round base
        #            (base = Phase 4 final model + confirmed winners from rounds 1..k-1)
        #            → select winner (most negative ΔOFV, < -3.84, cov success)
        #            → if winner: add to confirmed, update base, start round k+1
        #            → if no winner and scm_confirmed non-empty: switch to backward mode
        #            → if no winner and scm_confirmed empty: SCM complete
        #   Backward round k: test removing EACH confirmed covariate from the full model
        #            → eliminate the one with smallest ΔOFV increase if < 6.63
        #            → if nothing eliminable: backward (and SCM) complete
        #
        # scm_round_tested / scm_round_results / scm_round_base_ofv / scm_round_base_code
        # are shared between forward and backward modes (only one mode active at a time).
        self.covariate_history = []        # 전체 기록 [{name, delta_ofv, result, iteration, round, mode}]
        self.scm_confirmed = []            # 현재 확정된 covariate [{name, covariate, parameter, model_type, delta_ofv, code, ofv, iteration, round}]
        self.scm_eliminated = []           # backward에서 제거된 covariate [{name, delta_ofv, iteration, round}]
        self.scm_mode = 'forward'          # 'forward' | 'backward'
        self.scm_backward_threshold = 6.63  # p<0.01, df=1 — forward(3.84)보다 엄격
        self.scm_current_round = 1         # 현재 라운드 번호 (forward/backward 각각 1부터 시작)
        self.scm_round_tested = set()      # 현재 라운드에서 테스트 완료된 candidate 이름
        self.scm_round_results = []        # 현재 라운드 결과 [{name, ofv, delta_ofv, cov_ok, code, iteration, mode}]
        self.current_covariate_instruction = None  # 이번 iter에서 LLM이 테스트해야 할 covariate dict (mode='add'|'remove')
        self.scm_round_base_ofv = None     # 현재 라운드의 base OFV (라운드 시작 시 고정, 모든 후보의 ΔOFV 기준)
        self.scm_round_base_code = None    # 현재 라운드의 base code (매 후보 테스트 후 여기로 revert)

        # Parameter stabilization history (THETA/OMEGA/SIGMA across iterations)
        # Used ONLY to suggest narrower initial values and bounds; does not change other logic
        self.parameter_history: list[dict] = []

        # Print dataset summary
        print("\n" + "=" * 70)
        print("DATASET SUMMARY")
        print("=" * 70)
        print(self.data_loader.get_column_summary())
        print("\n" + self.data_loader.get_data_summary())
        print("=" * 70 + "\n")

    def run(self) -> Dict:
        """
        Run the recursive optimization process

        Returns:
            Dictionary with optimization results
        """
        print("\n" + "=" * 70)
        print("STARTING OPTIMIZATION")
        print("=" * 70)
        print(f"Min iterations: {self.min_iterations}")
        print(f"Max iterations (Phase 1-4): {self.max_iterations}")
        print(f"Phase 5 (SCM): forward selection (p<0.05) then backward elimination (p<0.01)")
        print(f"Output base: {self.output_base}")
        print("=" * 70 + "\n")

        # Step 1: Build drug-specific physiological plausibility bounds (once).
        # This now runs BEFORE initial code generation so the "typical" values
        # can be used as THETA initial estimates in the very first model,
        # instead of the generic template numbers.
        print("\n[INFO] Building physiological plausibility bounds...")
        self.plausibility_bounds = self._build_plausibility_bounds()

        # Step 1b: Generate initial NONMEM code
        self._generate_initial_code()

        # Step 2: Recursive improvement loop with phase-based control
        # Phase 1-4: max_iterations hard ceiling
        # Phase 5: runs until SCM forward selection complete (all candidates tested)
        self.iteration = 0
        while True:
            self.iteration += 1

            in_phase5 = (self.current_phase == ModelPhase.COVARIATE_ANALYSIS)

            # Phase 1-4 ceiling
            if not in_phase5 and self.iteration > self.max_iterations:
                print(f"\n[INFO] Max iterations ({self.max_iterations}) reached for Phase 1-4")
                print("[INFO] Proceeding to final summary")
                break

            # Phase 5 display: show SCM progress
            if in_phase5:
                tested_this_round = len(self.scm_round_tested)
                if self.scm_mode == 'backward':
                    remaining = len([c for c in self.scm_confirmed if c['name'] not in self.scm_round_tested])
                    iter_label = (f"{self.iteration} (Phase 5 SCM Backward Round {self.scm_current_round}: "
                                  f"{tested_this_round} tested, {remaining} remaining)")
                else:
                    all_cands = self._get_all_covariate_candidates()
                    confirmed_names = {c['name'] for c in self.scm_confirmed}
                    remaining = len([c for c in all_cands
                                     if c['name'] not in confirmed_names
                                     and c['name'] not in self.scm_round_tested])
                    iter_label = (f"{self.iteration} (Phase 5 SCM Round {self.scm_current_round}: "
                                  f"{tested_this_round} tested, {remaining} remaining)")
            else:
                iter_label = f"{self.iteration}/{self.max_iterations}"

            print(f"\n{'=' * 70}")
            print(f"ITERATION {iter_label}")
            print(f"CURRENT PHASE: {self.current_phase}")
            print(f"Iterations in phase: {self.iterations_in_phase}")
            print("=" * 70)

            # Run NONMEM
            success = self._run_nonmem()

            if success:
                # Parse results
                parsed_results = self._parse_results()

                # Determine and update current phase
                new_phase = self._determine_current_phase(parsed_results)
                self._update_phase(new_phase)

                # Phase 5 SCM: 현재 라운드의 모든 후보 테스트 완료 시 라운드 결과 처리
                if self.current_phase == ModelPhase.COVARIATE_ANALYSIS:
                    next_cand = self._get_next_covariate_to_test()
                    if next_cand is None and self.scm_round_results:
                        has_next = self._complete_scm_round()
                        if has_next:
                            # 다음 라운드(같은 모드)로 계속 진행
                            label = "Backward Round" if self.scm_mode == 'backward' else "Round"
                            print(f"\n[SCM] Starting {label} {self.scm_current_round}")
                        elif self.scm_mode == 'forward' and self.scm_confirmed:
                            # forward selection 종료 + 확정된 covariate 있음
                            # → 더 엄격한 기준(p<0.01)으로 backward elimination 시작
                            print("\n" + "=" * 70)
                            print(f"[SCM] Forward selection complete after {self.scm_current_round} round(s)")
                            print(f"  Confirmed covariates entering backward elimination: "
                                  f"{[c['name'] for c in self.scm_confirmed]}")
                            print(f"[SCM] Starting BACKWARD ELIMINATION "
                                  f"(retain only if ΔOFV on removal >= {self.scm_backward_threshold}, p<0.01)")
                            print("=" * 70)
                            self.scm_mode = 'backward'
                            self.scm_current_round = 1
                            self.scm_round_tested = set()
                            self.scm_round_results = []
                            self.scm_round_base_ofv = None  # _init_scm_round_base 중복 호출 가드 해제
                            self._init_scm_round_base(self.best_ofv, self.best_code)
                        else:
                            # forward에서 확정된 covariate가 없거나, backward elimination이
                            # 더 이상 제거할 것이 없어 완전히 끝남 → SCM 완료
                            confirmed_names = [c['name'] for c in self.scm_confirmed]
                            eliminated_names = [e['name'] for e in self.scm_eliminated]
                            print("\n" + "=" * 70)
                            print(f"[SCM COMPLETE] Forward + backward selection finished")
                            print(f"  Final covariates: {confirmed_names if confirmed_names else 'None'}")
                            if eliminated_names:
                                print(f"  Eliminated in backward step: {eliminated_names}")
                            print("[OK] Optimization complete")
                            print("=" * 70)
                            break

                # Check for improvement
                should_continue = self._evaluate_improvement(parsed_results)

                # Decide whether to continue
                if self.iteration >= self.min_iterations and not should_continue:
                    print("\n[OK] Optimization converged successfully!")
                    break

                # Generate improved code (phase-aware)
                # Phase 5 always generates next iter; Phase 1-4 checks ceiling
                if in_phase5 or self.iteration < self.max_iterations:
                    self._generate_improved_code(parsed_results)
            else:
                # NONMEM failed
                # Phase 5: 현재 phase 유지 (Phase 5 → Phase 1 regression 방지)
                # Phase 1-4: ESTABLISH_BASE로 리셋해서 syntax fix 시도
                if not in_phase5:
                    self.current_phase = ModelPhase.ESTABLISH_BASE
                if in_phase5 or self.iteration < self.max_iterations:
                    print("[WARNING] NONMEM execution failed - attempting to fix...")
                    self._generate_improved_code(None)
                else:
                    print("[ERROR] Maximum iterations reached with errors")
                    break

        # Final summary
        return self._generate_final_summary()

    def _generate_initial_code(self):
        """Generate initial NONMEM control stream"""
        print("Generating initial NONMEM control stream...")

        metadata = self.data_loader.get_metadata()

        dose_scaling_hint = self._detect_dose_unit_mismatch()
        if dose_scaling_hint:
            print(f"  [DOSE-CHECK] {dose_scaling_hint}")

        # WT(및 다른 covariate)는 CL이든 V1이든 무조건 고정 반영하지 않고,
        # 전부 Phase 5 SCM의 일반 후보로 넘겨 데이터가 유의성을 근거로
        # 채택 여부를 결정하게 한다 — "allometric을 걸어보고 안 걸어보고
        # 비교해서 정한다"는 human pharmacometrician의 과정을 SCM의
        # forward(p<0.05)/backward(p<0.01) 유의성 검증으로 그대로 모사.
        # (WT를 V1에 무조건 고정 적용했다가 이 데이터셋에서 ETA(V)-ETA(KA)
        # 상관관계가 0.998까지 치솟아 모델이 붕괴하는 부작용을 warfarin에서
        # 확인 — 고정 반영은 이런 식으로 특정 데이터셋을 오히려 해칠 수 있음)
        prompt = PromptTemplates.initial_generation_prompt(
            dataset_info=self.data_loader.get_column_summary(),
            data_summary=self.data_loader.get_data_summary(),
            columns=metadata['columns'],
            nonmem_columns=metadata.get('nonmem_columns', {}),
            covariates=metadata.get('covariates', []),
            plausibility_bounds=getattr(self, 'plausibility_bounds', None),
            dose_scaling_hint=dose_scaling_hint,
            prior_information=self._format_model_prior_information()
        )

        response = self.gemini_client.generate(prompt, model_type=self.model)

        # Extract NONMEM code from response
        self.current_code = self._enforce_foce_estimation(
            self._enforce_input_line(self._extract_nonmem_code(response))
        )

        # Save to file
        iteration_file = f"{self.output_base}_iter0.txt"
        with open(iteration_file, 'w', encoding='utf-8') as f:
            f.write(self.current_code)

        # Store in code history
        self.code_history.append({
            'iteration': 0,
            'code': self.current_code,
            'description': 'Initial generation'
        })

        print(f"[OK] Initial code generated and saved to: {iteration_file}")
        print(f"  Lines of code: {len(self.current_code.splitlines())}")

    def _extract_nonmem_code(self, response: str) -> str:
        """Extract NONMEM code from Gemini response"""
        # Try to find code block (relaxed: allow optional language tag, optional newlines around fences)
        code_block_pattern = r'```(?:[a-zA-Z]+)?\s*\n?(.*?)\n?```'
        match = re.search(code_block_pattern, response, re.DOTALL | re.IGNORECASE)

        if match:
            return self._strip_fences(match.group(1).strip())

        # If no code block, look for $PROBLEM (start of NONMEM code)
        problem_match = re.search(r'\$PROBLEM', response, re.IGNORECASE)
        if problem_match:
            # Extract from $PROBLEM to end or until analysis text
            code = response[problem_match.start():]

            # Try to find where the code ends
            end_patterns = [
                r'\n\s*```',  # Trailing markdown fence (any amount of whitespace before)
                r'\n\n[A-Z]{2,}:',  # Section headers like "ANALYSIS:"
                r'\n\n\*\*',  # Markdown headers
                r'\n\nNote:',  # Explanation notes
            ]

            for pattern in end_patterns:
                end_match = re.search(pattern, code)
                if end_match:
                    code = code[:end_match.start()]
                    break

            return self._strip_fences(code.strip())

        # Last resort - return full response
        return self._strip_fences(response.strip())

    def _strip_fences(self, code: str) -> str:
        """
        Remove any leftover markdown fence lines (```, ```nonmem, ```txt, etc.)
        from extracted code. Acts as a final safety net regardless of which
        extraction branch was taken.
        """
        if not code:
            return code
        lines = code.split('\n')
        # Drop lines that are purely a code fence (with optional language tag)
        clean_lines = [
            line for line in lines
            if not re.match(r'^\s*```[a-zA-Z]*\s*$', line)
        ]
        return '\n'.join(clean_lines).strip()

    def _enforce_input_line(self, code: str) -> str:
        """
        Force the $INPUT line to match the actual CSV column order.
        LLM occasionally reorders columns to a "standard" pattern, which
        silently corrupts NONMEM data mapping (NONMEM maps by POSITION).
        Safety net to guarantee correctness regardless of LLM output.
        """
        csv_columns = self.data_loader.get_metadata().get('columns', [])
        if not csv_columns:
            return code

        correct_input = "$INPUT " + " ".join(csv_columns)

        # $INPUT 줄을 다음 $ 토큰 직전까지 통째로 치환
        pattern = re.compile(r'\$INPUT\b[^\$]*', re.IGNORECASE | re.DOTALL)
        if pattern.search(code):
            new_code = pattern.sub(correct_input + "\n\n", code, count=1)
            if new_code != code:
                print(f"  [INPUT-FIX] $INPUT line normalized to CSV order: {' '.join(csv_columns)}")
            return new_code

        # $INPUT 자체가 없으면 $DATA 다음에 끼워넣기
        data_pattern = re.compile(r'(\$DATA[^\n]*\n)', re.IGNORECASE)
        if data_pattern.search(code):
            print(f"  [INPUT-FIX] $INPUT missing - inserted after $DATA")
            return data_pattern.sub(r'\1' + correct_input + "\n", code, count=1)

        return code


    def _get_model_prior_information(self) -> dict:
        """Return non-covariate prior information that is actually populated."""
        empty_markers = {'', 'none', 'null', 'not_provided', 'not provided', 'n/a'}

        def clean(value):
            if isinstance(value, dict):
                cleaned = {key: clean(item) for key, item in value.items()}
                return {key: item for key, item in cleaned.items() if item is not None}
            if isinstance(value, list):
                cleaned = [clean(item) for item in value]
                return [item for item in cleaned if item is not None]
            if isinstance(value, str):
                stripped = value.strip()
                return None if stripped.lower() in empty_markers else stripped
            return value

        fields = ('drug', 'nonclinical_data', 'published_data', 'population')
        result = {}
        for field in fields:
            value = clean(self.prior_info.get(field))
            if value not in (None, {}, []):
                result[field] = value
        return result

    def _format_model_prior_information(self) -> str:
        """Format optional user context for LLM prompts; return empty when absent."""
        import json

        prior = self._get_model_prior_information()
        if not prior:
            return ''
        return json.dumps(prior, ensure_ascii=False, indent=2)

    def _build_plausibility_bounds(self) -> dict:
        """
        LLM에게 약물별 생리학적 PK 파라미터 한계값을 한 번만 물어봄.
        output_base와 데이터 파일명에서 약물명을 추론하고
        CL, V1, Q, V2에 대한 min/max 범위를 JSON으로 반환받아 저장.
        """
        import json

        data_file = str(getattr(self.data_loader, 'file_path', ''))
        drug_hint = f"{self.output_base} / {data_file}"
        metadata  = self.data_loader.get_metadata()
        route_hint = "IV" if metadata.get('route', '').upper() == 'IV' else "unknown"
        n_subjects = metadata.get('n_subjects', '?')
        prior_information = self._format_model_prior_information()
        prior_section = ''
        if prior_information:
            prior_section = f"""
User-provided prior information (optional context):
{prior_information}

Use supplied values as additional prior context, not as automatically verified truth.
If a drug name is supplied, use it unless it clearly conflicts with the dataset.
Fill missing information from your existing pharmacokinetic knowledge. Do not claim
that an external literature search was performed.
"""

        prompt = f"""You are an expert clinical pharmacokineticist.

Dataset hint: "{drug_hint}"
Route of administration: {route_hint}
Number of subjects: {n_subjects}
{prior_section}

Task: Identify the drug from the hints above, then provide physiologically plausible
POPULATION-LEVEL ranges for its PK parameters (typical values, NOT individual extremes).
Also provide the typical SINGLE administered dose for this drug/route, in absolute
milligrams for an average adult patient — this is used to detect dose-unit mismatches
in the dataset (e.g. a dose column recorded as mg/kg instead of absolute mg).

Respond ONLY with valid JSON — no markdown fences, no explanation:
{{
  "drug_identified": "<drug name or 'unknown'>",
  "route": "<IV or oral>",
  "notes": "<1-sentence pharmacokinetic rationale>",
  "typical_single_dose_mg": <number, typical single dose in ABSOLUTE mg for an average adult>,
  "parameters": {{
    "CL":  {{"min": <number>, "typical": <number>, "max": <number>, "unit": "L/h",   "rationale": "<brief>"}},
    "V1":  {{"min": <number>, "typical": <number>, "max": <number>, "unit": "L",     "rationale": "<brief>"}},
    "Q":   {{"min": <number>, "typical": <number>, "max": <number>, "unit": "L/h",   "rationale": "<brief>"}},
    "V2":  {{"min": <number>, "typical": <number>, "max": <number>, "unit": "L",     "rationale": "<brief>"}},
    "Ka":  {{"min": <number>, "typical": <number>, "max": <number>, "unit": "1/h",   "rationale": "<brief>"}}
  }}
}}

"typical" = the single best population-representative point estimate (this is
what should be used as a THETA initial estimate) — NOT the same as min or max.

Be specific to the identified drug class. Use literature population estimates.
"""

        try:
            response = self.gemini_client.generate(prompt, model_type=self.model)
            json_text = response.strip()
            if json_text.startswith("```"):
                lines = json_text.split("\n")[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                json_text = "\n".join(lines).strip()
            bounds = json.loads(json_text)
            drug = bounds.get("drug_identified", "unknown")
            print(f"  [PLAUSIBILITY] Drug identified: {drug}")
            typical_dose = bounds.get("typical_single_dose_mg")
            if typical_dose is not None:
                print(f"  [PLAUSIBILITY]   Typical single dose: {typical_dose} mg")
            for pname, pvals in bounds.get("parameters", {}).items():
                typical = pvals.get('typical')
                typical_s = f", typical={typical}" if typical is not None else ""
                print(f"  [PLAUSIBILITY]   {pname}: {pvals.get('min')} – {pvals.get('max')} {pvals.get('unit')}{typical_s}")
            return bounds
        except Exception as e:
            print(f"  [WARNING] Could not build plausibility bounds: {e}")
            return {}

    def _detect_dose_unit_mismatch(self) -> Optional[str]:
        """
        데이터셋의 AMT 평균값과 약물의 전형적 절대 투여량(mg)을 비교해
        AMT가 실제로는 mg/kg, mcg, mcg/kg 등 다른 단위/스케일로 기록되어
        있는지 결정론적으로(코드 레벨에서, LLM 판단에 기대지 않고) 탐지한다.

        F1(bioavailability multiplier)은 선형 스케일 factor를 그대로 흡수할 수
        있으므로, 여러 후보 배율 중 실제 투여량과 가장 잘 맞는 것을 찾아
        F1 값을 제안하는 힌트 문자열을 반환한다. 애매하면 None을 반환해
        LLM이 임의로 넘겨짚지 않도록 한다.
        """
        dose_stats = self.data_loader.get_metadata().get('dose_stats', {})
        amt_mean = dose_stats.get('amt_mean')
        if amt_mean is None:
            return None

        typical_dose_mg = self.plausibility_bounds.get('typical_single_dose_mg') if self.plausibility_bounds else None
        if typical_dose_mg is None or typical_dose_mg <= 0:
            return None

        wt_mean = dose_stats.get('wt_mean')
        cv_div = dose_stats.get('cv_amt_div_wt')     # CV(AMT/WT)  — "이미 절대 mg" 가설의 일관성
        cv_times = dose_stats.get('cv_amt_times_wt') # CV(AMT*WT) — "아직 mg/kg 비율" 가설의 일관성

        # mg/kg, mcg/kg 후보가 성립하려면 "AMT가 아직 체중으로 안 곱해진 비율값"
        # 이어야 하는데, 이를 LLM이 추측한 "전형적 임상 용량" 하나와의 비율만으로
        # 판단하면 그 추측이 이 데이터셋의 실제 투여 관행과 다를 때(예: 만성
        # 유지용량 vs 연구 프로토콜의 절대 mg 단회 loading dose) 오탐지가
        # 발생한다 (warfarin.csv에서 실측: AMT=105는 원래 절대 mg 고정 용량인데
        # "전형적 용량=5mg"과 우연히 맞아떨어져 mcg/kg으로 오판, F1=WT*0.001이
        # 잘못 적용되어 투여량이 ~15배 축소되고 fit이 완전히 실패함).
        #
        # 대신 두 가설 각각의 내적 일관성(변동계수 CV)을 직접 비교한다 —
        # 사람이 데이터를 눈으로 보고 "어느 쪽이 처방 규칙에 더 잘 들어맞나"
        # 판단하는 것과 같은 방식이며, LLM의 typical-dose 추측에 의존하지 않는다.
        #   - CV(AMT/WT)가 뚜렷이 더 낮다  → AMT는 이미 절대 mg (mg/kg 비율로
        #     사이징만 됐을 뿐) → mg/kg·mcg/kg 후보 배제 (warfarin 케이스)
        #   - CV(AMT*WT)가 뚜렷이 더 낮다  → AMT는 아직 안 곱해진 비율값
        #     → mg/kg·mcg/kg 후보만 허용 (theo.csv 케이스)
        #   - 둘 다 애매하거나 계산 불가 → 기존처럼 두 후보 다 허용(안전한 fallback)
        CV_CONFIDENT = 0.3   # 이 정도로 낮아야 "일관된 처방 규칙"으로 신뢰
        CV_MARGIN = 0.5      # 상대편보다 최소 이 비율만큼은 더 낮아야 확실한 승자로 인정
        weight_conversion_needed = None  # True=mg/kg 변환 필요, False=이미 절대값, None=애매
        if cv_div is not None and cv_times is not None:
            if cv_div < CV_CONFIDENT and cv_div < cv_times * CV_MARGIN:
                weight_conversion_needed = False
            elif cv_times < CV_CONFIDENT and cv_times < cv_div * CV_MARGIN:
                weight_conversion_needed = True

        if wt_mean and weight_conversion_needed is False:
            print(f"  [DOSE-CHECK] AMT is consistent per-kg-normalized (CV(AMT/WT)={cv_div:.2f} "
                  f"vs CV(AMT*WT)={cv_times:.2f}) — AMT already reflects the administered absolute "
                  f"dose (weight-proportional sizing, not a not-yet-converted mg/kg rate); "
                  f"ruling out mg/kg-style scaling.")
        elif wt_mean and weight_conversion_needed is True:
            print(f"  [DOSE-CHECK] AMT*WT is consistent (CV(AMT*WT)={cv_times:.2f} vs "
                  f"CV(AMT/WT)={cv_div:.2f}) — AMT looks like a not-yet-converted per-kg dose rate.")

        allow_weight_candidates = wt_mean and (weight_conversion_needed is not False)

        # 후보 배율: (설명, F1 제안값, 검증용 실제투여량 = amt_mean * F1제안값의 역수... )
        # candidate: (label, scale_factor) 여기서 실제투여량(mg) = amt_mean * scale_factor
        candidates = [("absolute mg (no scaling needed)", 1.0)]
        if allow_weight_candidates:
            candidates.append((f"mg/kg (F1 = WT, mean WT={wt_mean:.1f})", wt_mean))
        candidates.append(("mcg, not mg (F1 = 0.001)", 0.001))
        if allow_weight_candidates:
            candidates.append((f"mcg/kg (F1 = WT * 0.001, mean WT={wt_mean:.1f})", wt_mean * 0.001))
        candidates.append(("g, not mg (F1 = 1000)", 1000.0))

        best_label, best_factor, best_ratio = None, None, None
        for label, factor in candidates:
            implied_dose = amt_mean * factor
            ratio = implied_dose / typical_dose_mg
            # 0.3~3배 이내면 "그럴듯하게 맞다"고 판단 (약물마다 체중/용량 편차 있으므로)
            if 0.3 <= ratio <= 3.0:
                # 여러 후보가 맞을 수 있으니, ratio가 1에 가장 가까운 것을 채택
                score = abs(1.0 - ratio)
                if best_ratio is None or score < best_ratio:
                    best_label, best_factor, best_ratio = label, factor, score

        if best_label is None:
            return None
        if best_factor == 1.0:
            return None  # AMT가 이미 절대 mg로 보임 — 스케일링 불필요

        return (
            f"DOSE UNIT MISMATCH DETECTED: dataset AMT mean = {amt_mean:.3g}, "
            f"typical clinical dose for this drug ≈ {typical_dose_mg}mg. "
            f"This matches: {best_label}. "
            f"MANDATORY: add the appropriate F1 scaling in $PK "
            f"(F1 = WT if mg/kg, F1 = 0.001 if mcg, F1 = WT*0.001 if mcg/kg, etc.) "
            f"so the actual administered dose reaching the compartment is in mg, "
            f"consistent with DV's concentration units."
        )

    def _extract_theta_param_map(self, code: str) -> dict:
        """
        $PK 블록을 파싱해서 THETA 인덱스 → 파라미터명 매핑 반환.
        예: {1: "CL", 2: "V1", 3: "Q", 4: "V2"}
        """
        import re
        param_map = {}

        patterns = [
            # TV 접두어 패턴
            (r"TVCL\s*=\s*THETA\((\d+)\)", "CL"),
            (r"TVV1\s*=\s*THETA\((\d+)\)", "V1"),
            (r"TVV(?!2)\s*=\s*THETA\((\d+)\)", "V1"),
            (r"TVQ\s*=\s*THETA\((\d+)\)",  "Q"),
            (r"TVV2\s*=\s*THETA\((\d+)\)", "V2"),
            (r"TVK(?:A|a)\s*=\s*THETA\((\d+)\)", "Ka"),
            # 직접 할당 패턴 (THETA가 직접 나오는 경우)
            (r"^\s*CL\s*=\s*THETA\((\d+)\)", "CL"),
            (r"^\s*V1\s*=\s*THETA\((\d+)\)", "V1"),
            (r"^\s*V\s*=\s*THETA\((\d+)\)",  "V1"),
            (r"^\s*Q\s*=\s*THETA\((\d+)\)",  "Q"),
            (r"^\s*V2\s*=\s*THETA\((\d+)\)", "V2"),
        ]

        for pattern, param_name in patterns:
            match = re.search(pattern, code, re.IGNORECASE | re.MULTILINE)
            if match:
                idx = int(match.group(1))
                if idx not in param_map:
                    param_map[idx] = param_name

        return param_map

    def _identify_boundary_theta(self, lst_output: str) -> list:
        """
        .lst 출력의 마지막 GRADIENT 라인에서 gradient = 0인 THETA 인덱스 목록 반환.

        NONMEM은 파라미터가 lower/upper bound에 고착되면
        해당 위치의 gradient를 정확히 0.0000E+00으로 설정한다.
        이를 이용해 실제로 boundary에 걸린 파라미터를 식별할 수 있다.

        Returns:
            List of THETA indices (1-based) with zero gradient (at boundary)
        """
        if not lst_output:
            return []

        # GRADIENT 라인 전체 추출 (마지막 iteration 값 사용)
        gradient_lines = re.findall(
            r'GRADIENT:\s+((?:[+-]?\s*\d+\.\d+E[+-]\d+\s*)+)',
            lst_output,
            re.IGNORECASE
        )
        if not gradient_lines:
            return []

        last_gradient = gradient_lines[-1]
        values = re.findall(r'[+-]?\d+\.\d+E[+-]\d+', last_gradient)

        # gradient가 정확히 0인 THETA 인덱스 수집 (1-based)
        boundary_indices = [
            i for i, val in enumerate(values, start=1)
            if float(val) == 0.0
        ]
        return boundary_indices

    def _check_plausibility(self, parsed_data: dict, code: str) -> dict:
        """
        현재 THETA 추정값과 사전 생성된 생리학적 한계값 비교.
        위반 시 심각도(MILD/MODERATE/SEVERE) 포함한 violations 리스트 반환.

        Returns:
            plausibility_report dict (quality_evaluation_prompt에 직접 전달)
        """
        if not self.plausibility_bounds:
            return {}

        bounds_params = self.plausibility_bounds.get("parameters", {})
        if not bounds_params:
            return {}

        params      = parsed_data.get("parameter_estimates", {})
        theta_list  = params.get("theta", [])
        if not theta_list:
            return {}

        theta_map   = self._extract_theta_param_map(code)
        violations  = []
        checked     = []
        score       = 100

        for theta_entry in theta_list:
            idx      = theta_entry.get("index")
            estimate = theta_entry.get("estimate")
            if idx is None or estimate is None:
                continue

            param_name = theta_map.get(idx)
            if not param_name or param_name not in bounds_params:
                continue

            bound     = bounds_params[param_name]
            min_val   = bound.get("min")
            max_val   = bound.get("max")
            unit      = bound.get("unit", "")
            rationale = bound.get("rationale", "")

            checked.append(f"{param_name}={estimate:.3g} {unit}")

            violation = None

            if max_val is not None and estimate > max_val:
                fold = estimate / max_val
                if fold > 10:
                    severity = "SEVERE";  score -= 30
                elif fold > 3:
                    severity = "MODERATE"; score -= 15
                else:
                    severity = "MILD";    score -= 5
                violation = (
                    f"{severity}: {param_name} = {estimate:.2g} {unit} "
                    f"exceeds physiological upper limit ({max_val} {unit}, "
                    f"{fold:.1f}\xd7 over). {rationale}"
                )
            elif min_val is not None and estimate < min_val:
                fold = min_val / max(estimate, 1e-9)
                if fold > 10:
                    severity = "SEVERE";  score -= 30
                elif fold > 3:
                    severity = "MODERATE"; score -= 15
                else:
                    severity = "MILD";    score -= 5
                violation = (
                    f"{severity}: {param_name} = {estimate:.2g} {unit} "
                    f"below physiological lower limit ({min_val} {unit}). {rationale}"
                )

            if violation:
                violations.append(violation)

        return {
            "violations": violations,
            "plausibility_score": max(0, score),
            "checked_parameters": checked,
            "drug": self.plausibility_bounds.get("drug_identified", "unknown"),
            "notes": self.plausibility_bounds.get("notes", ""),
        }

    def _enforce_phase5_structure(self, generated_code: str, base_code: str = None) -> str:
        """
        Phase 5에서 생성된 코드의 구조적 변경을 자동 감지하고 복원.

        허용:
          ✅ $THETA에 covariate 파라미터 1개 추가
          ✅ $PK에 covariate 관계식 추가
          ✅ $TABLE 파일명 업데이트

        금지 (감지 시 base model에서 자동 복원):
          ❌ $SUBROUTINES / ADVAN 변경 (구획 변경)
          ❌ $OMEGA 블록 수 감소 (IIV 제거)
          ❌ $ERROR 변경 (잔차 오차 모델)

        base_code: 비교 기준 코드 (반드시 best_code를 넘겨야 함).
                   self.current_code는 호출 시점에 이미 generated_code로 업데이트되어
                   있으므로 self.current_code를 base로 쓰면 자기 자신과 비교하게 됨.
        """
        reference = base_code if base_code is not None else self.best_code
        if not reference:
            return generated_code

        base = reference
        result = generated_code
        violations = []

        # 1. ADVAN 변경 감지
        base_advan = re.search(r'ADVAN(\d+)', base,   re.IGNORECASE)
        gen_advan  = re.search(r'ADVAN(\d+)', result, re.IGNORECASE)
        if base_advan and gen_advan and base_advan.group(1) != gen_advan.group(1):
            violations.append(
                f"ADVAN{gen_advan.group(1)} → ADVAN{base_advan.group(1)} (구조 변경 금지)"
            )
            base_sub = re.search(r'\$SUBROUTINES[^\$]+', base,   re.IGNORECASE | re.DOTALL)
            gen_sub  = re.search(r'\$SUBROUTINES[^\$]+', result, re.IGNORECASE | re.DOTALL)
            if base_sub and gen_sub:
                result = result.replace(gen_sub.group(0), base_sub.group(0), 1)

        # 2. OMEGA 블록 수 감소 감지
        base_omega_blocks = re.findall(r'\$OMEGA[^\$]+', base,   re.IGNORECASE | re.DOTALL)
        gen_omega_blocks  = re.findall(r'\$OMEGA[^\$]+', result, re.IGNORECASE | re.DOTALL)
        if len(gen_omega_blocks) < len(base_omega_blocks):
            violations.append(
                f"OMEGA 블록 {len(base_omega_blocks)}개 → {len(gen_omega_blocks)}개 (IIV 제거 금지)"
            )
            result_no_omega = re.sub(r'\$OMEGA[^\$]+', '', result,
                                     flags=re.IGNORECASE | re.DOTALL)
            sigma_m = re.search(r'\$SIGMA', result_no_omega, re.IGNORECASE)
            if sigma_m:
                ins = sigma_m.start()
                omega_str = ''.join(base_omega_blocks) + '\n'
                result = result_no_omega[:ins] + omega_str + result_no_omega[ins:]
            else:
                result = result_no_omega + '\n' + ''.join(base_omega_blocks)

        # 3. $ERROR 변경 감지
        base_error = re.search(r'\$ERROR[^\$]+', base,   re.IGNORECASE | re.DOTALL)
        gen_error  = re.search(r'\$ERROR[^\$]+', result, re.IGNORECASE | re.DOTALL)
        if base_error and gen_error:
            b = re.sub(r'\s+', ' ', base_error.group(0)).strip()
            g = re.sub(r'\s+', ' ', gen_error.group(0)).strip()
            if b != g:
                violations.append("$ERROR 블록 변경 → base 복원")
                result = result.replace(gen_error.group(0), base_error.group(0), 1)

        if violations:
            print(f"\n{'!'*70}")
            print("PHASE 5 STRUCTURAL VIOLATION — AUTO-CORRECTED")
            print(f"{'!'*70}")
            for v in violations:
                print(f"  ❌ {v}")
            print(f"  → 구조 요소를 base model에서 복원")
            print(f"  → Covariate 변경($PK 수정, $THETA 추가)만 유지")
            print(f"{'!'*70}")
            sig = "PHASE5_STRUCTURAL_CHANGE_BLOCKED"
            if sig not in self.failed_strategies:
                self.failed_strategies.append(sig)

        return result

    def _enforce_foce_estimation(self, code: str) -> str:
        """
        Block SAEM, ITS, BAYES, MCMC and any non-FOCE-I estimation methods.

        PKGPT's entire scoring and convergence logic is built on FOCE-I OFV
        (-2*log-likelihood scale ~100-500 for typical PopPK datasets).
        EM/Bayesian methods produce OFV on completely different scales, breaking
        OFV comparison, overfitting detection, and all phase-transition criteria.

        This method detects and replaces any non-FOCE-I $ESTIMATION block with
        the standard FOCE-I specification, ensuring OFV comparability across all
        iterations.
        """
        import re

        # Detect any non-FOCE-I estimation method
        # Includes: SAEM, ITS (Iterative Two Stage), BAYES, MCMC, NUTS
        non_foce_pattern = re.compile(
            r'\$EST(?:IMATION)?[^\$]*(?:'
            r'METHOD\s*=\s*(?:SAEM|ITS|BAYES|MCMC|NUTS)'
            r'|NBURN\s*='
            r'|NITER\s*='
            r')[^\$]*',
            re.IGNORECASE | re.DOTALL
        )

        if non_foce_pattern.search(code):
            # Identify which method was used for logging
            method_match = re.search(
                r'METHOD\s*=\s*(SAEM|ITS|BAYES|MCMC|NUTS)',
                code, re.IGNORECASE
            )
            method_name = method_match.group(1).upper() if method_match else 'non-FOCE-I'
            print(f"  [{method_name}-BLOCK] {method_name} detected — replacing with FOCE-I "
                  f"(PKGPT requires FOCE-I for OFV comparability)")

            # Failed strategy로 기록 → 다음 iteration LLM에게 "DO NOT REPEAT" 전달
            sig = f"ESTIMATION_METHOD_{method_name}_BLOCKED"
            if sig not in self.failed_strategies:
                self.failed_strategies.append(sig)
                print(f"  [TRACK] Recorded blocked method: {sig} — LLM will not retry {method_name}")

            # Replace full $ESTIMATION block with standard FOCE-I
            est_pattern = re.compile(
                r'\$EST(?:IMATION)?[^\$]*',
                re.IGNORECASE | re.DOTALL
            )
            foce_block = "$ESTIMATION METHOD=1 INTER MAXEVAL=9999 PRINT=5 POSTHOC NOABORT\n"
            code = est_pattern.sub(foce_block, code, count=1)
            # Remove any stray EM-specific keywords that survived replacement
            code = re.sub(r'\bNBURN\s*=\s*\d+[^\n]*\n', '', code, flags=re.IGNORECASE)
            code = re.sub(r'\bNITER\s*=\s*\d+[^\n]*\n', '', code, flags=re.IGNORECASE)
            code = re.sub(r'\bISAMPLE\s*=\s*\d+[^\n]*\n', '', code, flags=re.IGNORECASE)
            code = re.sub(r'\bNSIG\s*=\s*\d+[^\n]*\n', '', code, flags=re.IGNORECASE)

        return code


    def _validate_advan_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate ADVAN-specific syntax requirements before execution

        Args:
            code: NONMEM control stream code

        Returns:
            (is_valid, error_message) tuple
        """
        # Extract ADVAN type
        advan_match = re.search(r'ADVAN(\d+)', code, re.IGNORECASE)
        if not advan_match:
            return True, None  # No ADVAN specified, let NONMEM handle it

        advan_num = int(advan_match.group(1))

        # ADVAN4/5/6 (2-compartment) validation
        if advan_num in [4, 5, 6]:
            # Check for V3 error: ADVAN4 uses compartments 1,2,3 but we define V1,V2 (not V3)
            # The correct approach is to use K10, K12, K21 (not V3)

            # Common error: Defining S2 and S3 when using V1 and V2
            # ADVAN4 compartments: 1=Depot, 2=Central, 3=Peripheral
            # We should define S2 = V1 (central) but NOT S3 unless using V3

            pk_block = re.search(r'\$PK(.*?)(?=\$|$)', code, re.DOTALL | re.IGNORECASE)
            if pk_block:
                pk_content = pk_block.group(1)

                # Check if S3 is defined
                has_s3 = re.search(r'S3\s*=', pk_content, re.IGNORECASE)
                # Check if V3 is defined
                has_v3 = re.search(r'\bV3\s*=', pk_content, re.IGNORECASE)

                # Error: S3 defined but V3 not defined (common mistake)
                if has_s3 and not has_v3:
                    error_msg = (
                        f"ADVAN{advan_num} syntax error: S3 is defined but V3 is not defined. "
                        f"For 2-compartment models with ADVAN4, use:\n"
                        f"  - V1 (central volume) and V2 (peripheral volume)\n"
                        f"  - S2 = V1 (scaling for central compartment)\n"
                        f"  - Do NOT define S3 unless you explicitly define V3\n"
                        f"OR use micro-rate constants (K10, K12, K21) without S3."
                    )
                    return False, error_msg

                # Check if using V naming (V, not V1) which can confuse NONMEM
                has_v_not_v1 = re.search(r'\bV\s*=\s*TV', pk_content, re.IGNORECASE)
                has_v1 = re.search(r'\bV1\s*=\s*TV', pk_content, re.IGNORECASE)

                if advan_num == 4 and has_v_not_v1 and not has_v1:
                    error_msg = (
                        f"ADVAN4 requires explicit V1 for central volume, not just 'V'. "
                        f"Use 'V1 = TVV1 * EXP(ETA(2))' instead of 'V = TVV * EXP(ETA(2))'"
                    )
                    return False, error_msg

        return True, None

    def _run_nonmem(self) -> bool:
        """
        Execute NONMEM

        Returns:
            True if execution completed (even with errors), False if command failed
        """
        input_file = f"{self.output_base}_iter{self.iteration}.txt"
        output_file = f"{self.output_base}_iter{self.iteration}.lst"

        # Validate ADVAN syntax before execution
        is_valid, validation_error = self._validate_advan_syntax(self.current_code)
        if not is_valid:
            print(f"\n{'!'*70}")
            print("ADVAN SYNTAX VALIDATION FAILED")
            print(f"{'!'*70}")
            print(f"{validation_error}")
            print(f"{'!'*70}")
            print("\n[ERROR] Skipping NONMEM execution due to syntax error")
            print("[INFO] Will attempt to fix in next iteration")

            # Create a mock error output file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"SYNTAX VALIDATION ERROR\n\n{validation_error}\n")
            return False

        # Write current code to input file
        with open(input_file, 'w', encoding='utf-8') as f:
            # Update $DATA line to point to actual data file
            code = self.current_code
            # Use absolute path to ensure NONMEM can find the data file
            # Convert to forward slashes for cross-platform compatibility
            absolute_data_path = os.path.abspath(self.data_file).replace('\\', '/')
            # Find $DATA line and replace filename with absolute path
            code = re.sub(
                r'\$DATA\s+\S+',
                f'$DATA {absolute_data_path}',
                code,
                flags=re.IGNORECASE
            )
            f.write(code)

        print(f"\nExecuting NONMEM: {self.nmfe_command} {input_file} {output_file}")
        print("  [INFO] This may take a few minutes...")

        try:
            # Execute NONMEM
            # Simple execution, rely on lst file for results
            result = subprocess.run(
                [self.nmfe_command, input_file, output_file],
                timeout=600,
                cwd=os.path.dirname(os.path.abspath(input_file)) or '.'
            )

            print(f"  [INFO] NONMEM process finished (exit code: {result.returncode})")

            # Wait for output file to be fully written
            # NONMEM might still be writing even after process returns
            if os.path.exists(output_file):
                print("  [INFO] Waiting for output file to be complete...")
                max_wait = 30  # seconds
                wait_interval = 1  # second
                waited = 0
                prev_size = 0
                stable_count = 0

                while waited < max_wait:
                    time.sleep(wait_interval)
                    waited += wait_interval

                    # Check if file size is stable
                    try:
                        current_size = os.path.getsize(output_file)
                        if current_size == prev_size:
                            stable_count += 1
                            if stable_count >= 3:  # Stable for 3 seconds
                                # Also check if "Stop Time" appears in file
                                with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
                                    content = f.read()
                                    if 'Stop Time' in content or 'Stop time' in content.lower():
                                        print("  [OK] Output file complete")
                                        return True
                        else:
                            stable_count = 0
                            prev_size = current_size
                    except:
                        pass

                print(f"  [OK] NONMEM execution completed")
                return True
            else:
                print("[WARNING] Output file not created")
                return False

        except FileNotFoundError:
            print(f"[WARNING] NONMEM command '{self.nmfe_command}' not found")
            print("  This is expected if NONMEM is not installed on this machine")
            print("  Creating mock output for testing...")

            # Create a mock output file for testing
            self._create_mock_output(output_file)
            return True

        except subprocess.TimeoutExpired:
            print("[ERROR] NONMEM execution timed out (>10 minutes)")
            return False

        except Exception as e:
            print(f"[ERROR] Error executing NONMEM: {e}")
            return False

    def _create_mock_output(self, output_file: str):
        """Create mock NONMEM output for testing when NONMEM is not available"""
        mock_output = f"""Mock NONMEM output for testing
ITERATION: {self.iteration}

This is a placeholder because NONMEM is not installed on this system.
When running on a system with NONMEM, this will contain actual output.

MINIMIZATION TERMINATED
OBJECTIVE FUNCTION VALUE: {1000 + self.iteration * 10}

This mock allows the optimizer to continue and demonstrate the recursive improvement process.
"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(mock_output)

    def _parse_results(self) -> Optional[NONMEMParser]:
        """Parse NONMEM output file"""
        output_file = f"{self.output_base}_iter{self.iteration}.lst"

        try:
            # Pass gemini_client to parser for AI-based parsing
            # Get the underlying client for the selected model
            selected_client = self.gemini_client.clients.get(self.model)
            parser = NONMEMParser(
                output_file,
                gemini_client=selected_client,
                use_ai_parsing=True
            )
            print("\n" + parser.get_summary())
            return parser

        except Exception as e:
            print(f"[WARNING] Error parsing NONMEM output: {e}")
            return None

    def _calculate_composite_score(self, ofv: Optional[float], shrinkage: Optional[float],
                                   cov_success: bool, minimization_ok: bool,
                                   omega_values: List[float],
                                   max_rse: Optional[float] = None) -> float:
        """
        Calculate composite quality score (lower is better)
        Penalizes overfitting, extreme shrinkage, and numerical instability

        Args:
            ofv: Objective function value
            shrinkage: Average ETA shrinkage percentage
            cov_success: Whether covariance step succeeded
            minimization_ok: Whether minimization succeeded
            omega_values: List of OMEGA diagonal values

        Returns:
            Composite score (lower = better model)
        """
        score = 0

        # OFV component
        if ofv is not None:
            if ofv < -50:
                # CRITICAL: Negative OFV in SAEM often indicates overfitting
                # Give huge penalty - likely fitting noise
                score += 50000
                print(f"  [WARNING] Negative OFV penalty: +50000")
            elif ofv < 0:
                # Small negative OFV might be acceptable in some methods
                score += 10000
                print(f"  [WARNING] Small negative OFV penalty: +10000")
            else:
                # Normal positive OFV - use as-is
                score += ofv
        else:
            # No OFV available - large penalty
            score += 100000

        # Shrinkage penalty (CRITICAL: extreme shrinkage = overfitting)
        if shrinkage is not None:
            if shrinkage > 95:
                # Catastrophic shrinkage - model has lost individual variability
                score += 20000
                print(f"  [CRITICAL] Extreme shrinkage (>95%) penalty: +20000")
            elif shrinkage > 90:
                # Severe shrinkage - likely overparameterized
                score += 10000
                print(f"  [CRITICAL] Severe shrinkage (>90%) penalty: +10000")
            elif shrinkage > 70:
                # High shrinkage - concerning
                score += 2000
                print(f"  [WARNING] High shrinkage (>70%) penalty: +2000")
            elif shrinkage > 50:
                # Moderate shrinkage - some concern
                score += 500

        # OMEGA near-zero detection (informational; mild penalty)
        if omega_values:
            collapsed_omegas = [o for o in omega_values if o < 0.0001]
            if collapsed_omegas:
                penalty = len(collapsed_omegas) * 500
                score += penalty
                print(f"  [WARNING] {len(collapsed_omegas)} OMEGA(s) very small (<0.0001): +{penalty}")

        # Covariance failure penalty (context-dependent)
        if not cov_success:
            if shrinkage is not None and shrinkage < 40:
                # If shrinkage is good (<40%), covariance failure is less critical
                # Model may still be useful for simulation/prediction
                score += 100
                print(f"  [INFO] Covariance failed but shrinkage good (<40%): +100")
            elif shrinkage is not None and shrinkage < 60:
                # Moderate shrinkage - covariance failure is concerning
                score += 200
                print(f"  [WARNING] Covariance failed with moderate shrinkage: +200")
            else:
                # Poor shrinkage + covariance failure = serious problem
                score += 300
                print(f"  [WARNING] Covariance failed with poor shrinkage: +300")
        else:
            print(f"  [OK] Covariance step successful: +0")

        # RSE penalty (Phase 1-4 only)
        # Phase 5에서는 적용 안 함: covariate 추가 시 RSE가 일시적으로 높아져도
        # ΔOFV 기준으로 acceptance 판단해야 하므로 composite 패널티가 방해되면 안 됨
        in_phase5 = (self.current_phase == ModelPhase.COVARIATE_ANALYSIS)
        if not in_phase5 and max_rse is not None and cov_success:
            if max_rse > 200:
                # 심각: 파라미터 비식별성 수준 → 큰 패널티
                rse_penalty = min((max_rse - 200) * 2.0 + 100, 1000)
                score += rse_penalty
                print(f"  [WARNING] Severe RSE penalty (max RSE {max_rse:.0f}%): +{rse_penalty:.0f}")
            elif max_rse > 100:
                # 불량: 추정 정밀도 부족 → 중간 패널티
                rse_penalty = (max_rse - 100) * 1.0
                score += rse_penalty
                print(f"  [INFO] High RSE penalty (max RSE {max_rse:.0f}%): +{rse_penalty:.0f}")
            # RSE ≤ 100%: 패널티 없음 (50% 기준도 있지만 소규모 데이터셋 허용)

        # Minimization failure penalty
        if not minimization_ok:
            score += 2000

        return score

    def _is_true_overfitting(self, ofv: Optional[float], avg_shrink: Optional[float]) -> bool:
        """
        Distinguish true overfitting from underparameterization

        True overfitting: Model too complex, shrinkage high AND model stable/improving
        Underparameterization: Shrinkage high BUT model got worse (OFV increased significantly)

        Args:
            ofv: Current objective function value
            avg_shrink: Average ETA shrinkage

        Returns:
            True if this is true overfitting (should simplify)
        """
        if avg_shrink is None or avg_shrink <= 90:
            # Shrinkage not critical
            return False

        # Check recent OFV trend
        if len(self.improvement_history) >= 2:
            recent_ofvs = [h.get('ofv') for h in self.improvement_history[-2:]]
            if all(o is not None for o in recent_ofvs):
                prev_ofv, current_ofv = recent_ofvs

                # If OFV got much worse (>30%), this is likely underparameterization
                # (we removed too much and model can't fit data anymore)
                if current_ofv > prev_ofv * 1.3:
                    shrink_val = avg_shrink
                    prev_val = prev_ofv
                    curr_val = current_ofv
                    if shrink_val is not None:
                        print(f"  [ANALYSIS] High shrinkage ({shrink_val:.1f}%) BUT OFV worsened")
                    else:
                        print(f"  [ANALYSIS] High shrinkage (N/A%) BUT OFV worsened")
                    print(f"  [ANALYSIS] OFV changed from {prev_val:.1f} to {curr_val:.1f}")
                    print(f"  [ANALYSIS] This suggests UNDERPARAMETERIZATION, not overfitting")
                    return False

        # If OFV very negative, definitely overfitting
        if ofv is not None and ofv < -50:
            return True

        # High shrinkage without OFV worsening -> likely overfitting
        return True

    def _determine_current_phase(self, parser: Optional[NONMEMParser]) -> ModelPhase:
        """
        Determine current optimization phase based on model state

        Uses PhaseTransitionManager for cleaner, explicit state machine logic

        Phase progression:
        1. ESTABLISH_BASE: Until minimization successful
        2. DIAGNOSE_STRUCTURE: Check if compartment model is adequate
        3. REDUCE_OVERFITTING: If TRUE overfitting detected (not underparameterization)
        4. OPTIMIZE_IIV: Fine-tune random effects
        5. COVARIATE_ANALYSIS: If base model stable and covariates available

        CRITICAL: Phases can ONLY move forward, never backward!
        - Prevents infinite loops (Phase 4 → Phase 1 → Phase 4 ...)
        - Once a phase is complete, it stays complete
        - If issues occur in later phases, fix within current phase

        CRITICAL: For very small datasets (N<20), avoid Phase 2 (DIAGNOSE_STRUCTURE)
        after successful minimization, as structural changes often break small-sample models.

        Args:
            parser: NONMEM output parser

        Returns:
            Current model phase (always >= self.current_phase)
        """
        if parser is None:
            # Can't determine phase without output - stay in current phase
            return self.current_phase

        # Initialize phase manager on first use
        if self.phase_manager is None:
            self.phase_manager = PhaseTransitionManager(
                self.data_loader,
                self.current_phase,
                self.iterations_in_phase,
                self.improvement_history
            )

        # Update phase manager state
        self.phase_manager.current_phase = self.current_phase
        self.phase_manager.iterations_in_phase = self.iterations_in_phase

        # Get parsed results
        parsed_results = parser.get_parsed_data() if parser else {}

        # Determine next phase using clean state machine logic
        next_phase = self.phase_manager.determine_next_phase(parsed_results)

        return next_phase

    def _update_phase(self, new_phase: ModelPhase):
        """
        Update current phase and track transition

        CRITICAL: Only allows FORWARD phase transitions!
        - Prevents backward movement (e.g., Phase 4 → Phase 1)
        - Prevents infinite loops
        - Ensures systematic progression
        """
        # SAFETY CHECK: Prevent backward phase transitions
        if new_phase.value < self.current_phase.value:
            print(f"\n{'⚠'*70}")
            print(f"[WARNING] Attempted backward phase transition blocked!")
            print(f"[WARNING] Current: {self.current_phase}, Requested: {new_phase}")
            print(f"[WARNING] Phases can only move FORWARD. Staying in {self.current_phase}")
            print(f"{'⚠'*70}\n")
            # Stay in current phase
            self.iterations_in_phase += 1
            return

        if new_phase != self.current_phase:
            print(f"\n{'='*70}")
            print(f"PHASE TRANSITION: {self.current_phase} -> {new_phase}")
            print(f"{'='*70}")
            self.phase_history.append({
                'iteration': self.iteration,
                'from_phase': self.current_phase,
                'to_phase': new_phase,
                'iterations_in_previous_phase': self.iterations_in_phase
            })
            self.current_phase = new_phase
            self.iterations_in_phase = 0

            # Phase 5 진입 시점의 self.best_code/self.best_ofv는 이 전환을 유발한
            # "이번 iteration"의 결과가 아직 반영되지 않은 stale 값이다 (run()에서
            # _update_phase()가 _evaluate_improvement()보다 먼저 실행되기 때문).
            # 여기서 self.current_code를 stale한 self.best_code로 되돌리면 방금
            # Phase 4를 완료시킨 실제 최종 모델이 유실된다.
            # → 아무것도 하지 않고, base freeze는 _evaluate_improvement()에서
            #   current_ofv/self.current_code(=이번 iteration 결과)로 수행한다.
        else:
            self.iterations_in_phase += 1

    def _should_revert_to_best(self, current_composite: float, current_ofv: Optional[float]) -> bool:
        """
        Check if we should revert to best iteration

        Revert if:
        1. Phase 5: covariance step failed → always revert (covariate rejected)
        2. Current model is 2x worse than best (by composite score)
        3. OFV increased by >2x
        4. We've been getting worse for 2+ consecutive iterations

        Args:
            current_composite: Current composite quality score
            current_ofv: Current objective function value

        Returns:
            True if should revert to best model
        """
        # Need at least best model and 2 iterations to compare
        if not self.best_code or len(self.improvement_history) < 2:
            return False

        # Phase 5 전용: covariance 실패 = 즉시 revert (covariate reject)
        # base model 구조를 바꿀 기회를 주지 않음
        if self.current_phase == ModelPhase.COVARIATE_ANALYSIS:
            last_entry = self.improvement_history[-1] if self.improvement_history else {}
            cov_ok = last_entry.get('covariance_successful', True)
            min_ok = last_entry.get('minimization_successful', True)
            if not cov_ok or not min_ok:
                print(f"  [REVERT TRIGGER] Phase 5: covariance/minimization failed → covariate rejected")
                return True

        # Don't revert too early
        if self.iteration - self.best_iteration < 2:
            return False

        # Check 1: Composite score 2x worse
        if self.best_composite_score != float('inf'):
            if current_composite > self.best_composite_score * 2:
                print(f"  [REVERT TRIGGER] Composite score 2x worse than best")
                return True

        # Check 2: OFV doubled
        if self.best_ofv is not None and current_ofv is not None:
            if current_ofv > self.best_ofv * 2:
                print(f"  [REVERT TRIGGER] OFV doubled from best")
                return True

        # Check 3: Consistent deterioration for 2+ iterations
        if len(self.improvement_history) >= 3:
            recent_scores = [h.get('composite_score', float('inf'))
                           for h in self.improvement_history[-3:]]
            # All getting worse
            if (recent_scores[0] < recent_scores[1] < recent_scores[2] and
                recent_scores[2] > self.best_composite_score * 1.5):
                print(f"  [REVERT TRIGGER] Consistent deterioration for 3 iterations")
                return True

        return False

    def _should_stop_early(self) -> tuple[bool, str]:
        """
        Check if optimization should stop early due to CATASTROPHIC persistent issues

        Philosophy: Let the model keep trying to improve. Only stop if truly hopeless.

        Returns:
            (should_stop, reason) tuple
        """
        if len(self.improvement_history) < 5:
            # Don't stop before 5 iterations - give it a real chance
            return False, ""

        # Check last N iterations for persistent problems
        recent_n = 6
        recent_history = self.improvement_history[-recent_n:]

        # 1. CATASTROPHIC SHRINKAGE: >95% for 4+ consecutive iterations
        #    (This means model is truly hopeless, not just suboptimal)
        catastrophic_shrink = [h for h in recent_history[-4:]
                              if h.get('avg_eta_shrinkage') is not None
                              and h.get('avg_eta_shrinkage') > 95]
        if len(catastrophic_shrink) >= 4:
            reason = (f"ETA shrinkage >95% (catastrophic) for {len(catastrophic_shrink)} consecutive iterations. "
                     "The model has completely failed to estimate individual variability despite multiple attempts. "
                     "Dataset may be too small or model fundamentally misspecified.")
            return True, reason

        # 2. PERSISTENT NEGATIVE OFV: Negative for 4+ consecutive iterations
        #    (Clear sign of severe overfitting that won't resolve)
        negative_ofv = [h for h in recent_history[-4:]
                       if h.get('ofv') is not None and h.get('ofv') < -50]
        if len(negative_ofv) >= 4:
            reason = (f"Negative OFV (<-50) for {len(negative_ofv)} consecutive iterations. "
                     "Severe overfitting persists despite attempts to fix. Model is fitting noise.")
            return True, reason

        # 3. TOTAL COLLAPSE: All OMEGAs collapsed for 3+ iterations
        #    (Not just some, but ALL OMEGAs collapsed = hopeless)
        recent_with_omega = [h for h in recent_history[-3:]
                           if h.get('omega_values') is not None and len(h.get('omega_values', [])) > 0]
        if len(recent_with_omega) >= 3:
            all_collapsed_count = 0
            for entry in recent_with_omega:
                omega_values = entry.get('omega_values', [])
                collapsed = [o for o in omega_values if o < 0.001]
                if len(collapsed) == len(omega_values) and len(omega_values) > 0:
                    all_collapsed_count += 1

            if all_collapsed_count >= 3:
                reason = (f"ALL OMEGA parameters collapsed (<0.001) for {all_collapsed_count} consecutive iterations. "
                         f"Individual variability structure is completely lost and cannot be recovered.")
                return True, reason

        # 4. NO PROGRESS AT ALL: Score getting worse for 5+ iterations
        #    (Model is actively degrading, not improving)
        if len(self.improvement_history) >= 6:
            recent_scores = [h.get('composite_score', float('inf'))
                           for h in self.improvement_history[-6:]]
            if all(score != float('inf') for score in recent_scores):
                # Check if scores are monotonically increasing (getting worse)
                getting_worse = all(recent_scores[i] <= recent_scores[i+1]
                                   for i in range(len(recent_scores)-1))
                if getting_worse and recent_scores[-1] > recent_scores[0] * 1.5:
                    reason = (f"Composite score has been degrading for 6 consecutive iterations "
                             f"({recent_scores[0]:.1f} -> {recent_scores[-1]:.1f}). "
                             f"Model quality is getting worse, not better.")
                    return True, reason

        # OTHERWISE: Keep trying! Don't give up too easily.
        return False, ""

    def _init_scm_round_base(self, base_ofv: Optional[float], base_code: str,
                              composite: float = float('inf')) -> None:
        """
        SCM 라운드의 base를 설정한다.

        Round 1: Phase 4 최종 모델의 OFV/code로 호출됨.
        Round 2+: 직전 라운드 winner의 OFV/code로 호출됨 (_complete_scm_round에서).

        같은 라운드 내 중복 호출은 가드로 무시한다.
        """
        if self.scm_round_base_ofv is not None:
            return

        self.scm_round_base_ofv = base_ofv
        self.scm_round_base_code = base_code
        self.best_ofv = base_ofv
        self.best_code = base_code
        self.best_composite_score = composite
        self.best_iteration = self.iteration

        ofv_s = f"{base_ofv:.2f}" if base_ofv is not None else "N/A"
        confirmed_names = [c['name'] for c in self.scm_confirmed]
        if confirmed_names:
            print(f"  [SCM Round {self.scm_current_round}] Base OFV = {ofv_s} "
                  f"(includes confirmed: {confirmed_names})")
        else:
            print(f"  [SCM Round {self.scm_current_round}] Base OFV = {ofv_s} "
                  f"(Phase 4 final model, iter {self.iteration})")

    def _evaluate_improvement(self, parser: Optional[NONMEMParser]) -> bool:
        """
        Evaluate if improvement occurred and should continue

        Args:
            parser: NONMEM output parser

        Returns:
            True if should continue optimizing
        """
        if parser is None:
            self.improvement_history.append({
                'iteration': self.iteration,
                'status': 'failed',
                'ofv': None,
                'issues': ['Failed to parse output'],
                'changes': 'N/A',
                'composite_score': float('inf')
            })
            return True  # Try to fix

        parsed_data = parser.get_parsed_data()
        # Update parameter stabilization history based on latest successful run
        self._update_parameter_history(parsed_data)
        current_ofv = parsed_data.get('objective_function')
        minimization_ok = parsed_data.get('minimization_successful', False)
        issues = parser.get_issues()

        # Extract covariance step status early (needed for Phase 5 base OFV guard)
        cov_step = parsed_data.get('covariance_step', {})
        cov_success = cov_step.get('successful', False)

        # Extract RSE and Shrinkage metrics
        rse_data = parsed_data.get('rse_percent', {})
        eta_shrinkage = parsed_data.get('eta_shrinkage', [])

        max_rse = rse_data.get('max_rse')
        high_rse_count = rse_data.get('high_rse_count', 0)
        avg_eta_shrinkage = max([s['shrinkage'] for s in eta_shrinkage]) if eta_shrinkage else None

        # Extract OMEGA values for collapse detection
        params = parsed_data.get('parameter_estimates', {})
        omega_values = [float(o.get('estimate', 1.0)) for o in params.get('omega', []) if 'estimate' in o]

        # Calculate composite quality score
        print(f"\n{'='*70}")
        print("COMPOSITE QUALITY SCORE CALCULATION")
        print(f"{'='*70}")
        current_composite = self._calculate_composite_score(
            current_ofv, avg_eta_shrinkage, cov_success, minimization_ok, omega_values,
            max_rse=max_rse
        )
        print(f"Total Composite Score: {current_composite:.2f} (lower is better)")
        print(f"{'='*70}")

        # Phase 5 SCM: 라운드 base 초기화 + 테스트 결과 기록
        if self.current_phase == ModelPhase.COVARIATE_ANALYSIS:
            self._init_scm_round_base(current_ofv, self.current_code, current_composite)
            self._record_phase5_covariate_result(current_ofv, cov_success)

        # Get dataset size
        metadata = self.data_loader.get_metadata()
        n_subjects = metadata.get('n_subjects', 0)

        # Record history
        history_entry = {
            'iteration': self.iteration,
            'status': 'success' if minimization_ok else 'failed',
            'ofv': current_ofv,
            'max_rse': max_rse,
            'high_rse_count': high_rse_count,
            'avg_eta_shrinkage': avg_eta_shrinkage,
            'issues': issues,
            'minimization_successful': minimization_ok,
            'covariance_successful': cov_success,
            'composite_score': current_composite,
            'omega_values': omega_values,
            'n_subjects': n_subjects
        }

        self.improvement_history.append(history_entry)

        # ====================================================================
        # Phase 5 (SCM): 매 후보 테스트 후 라운드 base로 revert
        # 코드는 scm_round_results에 이미 저장됨 (_record_phase5_covariate_result)
        # ACCEPT/REJECT 판정은 라운드 종료 시 _complete_scm_round()에서 수행
        # ====================================================================
        if self.current_phase == ModelPhase.COVARIATE_ANALYSIS:
            if self.scm_round_results:
                last = self.scm_round_results[-1]
                ofv_s = f"{current_ofv:.2f}" if current_ofv is not None else "N/A"
                shrink_s = f"{avg_eta_shrinkage:.1f}%" if avg_eta_shrinkage is not None else "N/A"
                rse_s = f"{max_rse:.1f}%" if max_rse is not None else "N/A"
                print(f"\n[SCM Round {self.scm_current_round}] {last['name']} recorded "
                      f"(OFV={ofv_s}, Shrink={shrink_s}, MaxRSE={rse_s})")

            if self.scm_round_base_code and self.current_code != self.scm_round_base_code:
                self.current_code = self.scm_round_base_code
                history_entry['reverted_to_round_base'] = True
                print(f"  [SCM Round {self.scm_current_round}] Reverted to round base for next candidate")

        else:
            # Phase 1-4: composite score 기반 best_code 관리
            if current_composite < self.best_composite_score:
                self.best_composite_score = current_composite
                self.best_ofv = current_ofv
                self.best_iteration = self.iteration
                self.best_code = self.current_code
                parts = [f"Composite={current_composite:.2f}"]
                if current_ofv is not None: parts.append(f"OFV={current_ofv:.2f}")
                if avg_eta_shrinkage is not None: parts.append(f"Shrink={avg_eta_shrinkage:.1f}%")
                print(f"\n[OK] NEW BEST MODEL: {', '.join(parts)}")


        # Check for early stopping conditions FIRST
        should_stop, stop_reason = self._should_stop_early()
        if should_stop:
            print(f"\n{'='*70}")
            print("EARLY STOPPING TRIGGERED")
            print(f"{'='*70}")
            print(f"Reason: {stop_reason}")
            print(f"{'='*70}")
            return False  # Stop optimization

        # Phase 5 (SCM): skip the AI quality evaluation below entirely.
        # Its output (recommendations/critical_issues) is never consumed by Phase 5 —
        # _generate_improved_code()'s in_phase5 branch uses phase_guidance directly
        # and explicitly does NOT inject ai_recommendations (to avoid contaminating
        # the single-covariate SCM instruction). SCM's continue/stop decision is
        # already fully deterministic (ΔOFV threshold in _record_phase5_covariate_result,
        # plus the "next untested candidate" check in run() before this function is
        # even called). Calling the LLM here only adds ~20-100s of dead-weight latency
        # per iteration, and a stray should_continue=False verdict from it could cut
        # the SCM sweep short before every candidate has been tested (the Phase-5
        # override guard no longer blocks stopping once current_phase >= Phase 5).
        if self.current_phase == ModelPhase.COVARIATE_ANALYSIS:
            return True

        # ALWAYS perform AI quality evaluation first (for all cases)
        print("\n[INFO] Performing comprehensive AI quality evaluation...")
        # .lst 파일 전체를 읽어서 quality eval LLM에게 전달
        # → LLM이 THETA boundary, gradient, 경고 등을 직접 읽고 구체적 권고 생성 가능
        _lst_file = f"{self.output_base}_iter{self.iteration}.lst"
        try:
            with open(_lst_file, 'r', encoding='utf-8', errors='ignore') as _f:
                _lst_output = _f.read()
        except Exception:
            _lst_output = ""

        _ai_result = self._ai_quality_check(parsed_data,
                                             max_rse=max_rse,
                                             high_rse_count=high_rse_count,
                                             lst_output=_lst_output)

        # Extract should_continue bool and store recommendations for next improvement
        if _ai_result is not None:
            should_continue_ai = _ai_result['should_continue']
            self.last_ai_recommendations = _ai_result.get('recommendations', [])
            self.last_ai_critical_issues  = _ai_result.get('critical_issues', [])
        else:
            should_continue_ai = None
            self.last_ai_recommendations = []
            self.last_ai_critical_issues  = []

        # If AI evaluation succeeded, use its decision
        if should_continue_ai is not None:
            # Override: If AI says stop but model isn't actually good, keep going
            if not should_continue_ai:
                # Stricter quality checks before accepting stop decision

                # Check 1: Minimization must be successful
                if not minimization_ok:
                    print("\n[OVERRIDE] AI suggested stopping but minimization failed - continuing")
                    return True

                # Check 2: Shrinkage must be acceptable for dataset size
                metadata = self.data_loader.get_metadata()
                num_subjects = metadata.get('n_subjects', 100)

                shrinkage_threshold = 70 if num_subjects < 20 else 60 if num_subjects < 50 else 50

                if avg_eta_shrinkage is not None and avg_eta_shrinkage > shrinkage_threshold:
                    shrink_val = avg_eta_shrinkage
                    thresh_val = shrinkage_threshold
                    n_val = num_subjects
                    print(f"\n[OVERRIDE] AI suggested stopping but shrinkage too high")
                    if shrink_val is not None:
                        print(f"[OVERRIDE] Shrinkage {shrink_val:.1f}% > threshold {thresh_val}% for N={n_val}")
                    else:
                        print(f"[OVERRIDE] Shrinkage N/A% > threshold {thresh_val}% for N={n_val}")
                    print(f"[OVERRIDE] Continuing optimization")
                    return True

                # Check 3: Model must have reasonable quality score
                # Only stop if quality score is at least 60/100
                if self.improvement_history and 'ai_evaluation' in self.improvement_history[-1]:
                    last_eval = self.improvement_history[-1]['ai_evaluation']
                    quality_score = last_eval.get('quality_score', 0)
                    if quality_score < 60:
                        score_val = quality_score
                        print(f"\n[OVERRIDE] AI suggested stopping but quality score too low")
                        print(f"[OVERRIDE] Quality score {score_val}/100 is less than 60")
                        print(f"[OVERRIDE] Continuing optimization")
                        return True

                # Check 5: Must have reached at least Phase 5 before stopping
                # Full workflow: Phase 1 → 2 → 3 → 4 → 5 (Covariates)
                # Only allow stopping after covariate analysis has been attempted
                min_phase_to_stop = ModelPhase.COVARIATE_ANALYSIS.value  # Phase 5
                current_phase_val = self.current_phase.value if hasattr(self.current_phase, 'value') else 0
                if current_phase_val < min_phase_to_stop:
                    phase_name = str(self.current_phase)
                    # Phase 5 진입 전제조건: best model의 covariance step 성공 여부 확인
                    best_entry = next(
                        (e for e in self.improvement_history
                         if e.get('iteration') == self.best_iteration), None
                    )
                    best_cov_ok = best_entry.get('covariance_successful', False) if best_entry else False

                    if not best_cov_ok:
                        # Covariance 미성공 → Phase 5 진입 보류, Phase 4에서 계속 시도
                        print(f"\n[PHASE OVERRIDE] AI suggested stopping at {phase_name}")
                        print(f"[PHASE OVERRIDE] Phase 5 blocked: best model (Iter {self.best_iteration}) "
                              f"has no successful covariance step")
                        print(f"[PHASE OVERRIDE] Continuing Phase 1-4 to achieve stable covariance")
                        return True  # Phase 4에서 계속

                    print(f"\n[PHASE OVERRIDE] AI suggested stopping at {phase_name}")
                    print(f"[PHASE OVERRIDE] Phase 5 (Covariate Analysis) not yet attempted")
                    print(f"[PHASE OVERRIDE] Forcing continuation to complete full workflow")
                    self._update_phase(ModelPhase.COVARIATE_ANALYSIS)
                    self._init_scm_round_base(current_ofv, self.current_code, current_composite)
                    self._record_phase5_covariate_result(current_ofv, cov_success)
                    return True

                # Check 5.5: Phase 5 SCM — 미시도 covariate가 남아있으면 계속
                if self.current_phase == ModelPhase.COVARIATE_ANALYSIS:
                    next_cand = self._get_next_covariate_to_test()
                    if next_cand is not None:
                        self.current_covariate_instruction = next_cand
                        print(f"\n[SCM OVERRIDE] Next untested: '{next_cand['name']}'")
                        print(f"[SCM OVERRIDE] Forcing continuation for forward selection")
                        return True
                    else:
                        confirmed = len(self.scm_confirmed)
                        tested = len(self.scm_round_tested)
                        print(f"\n[SCM] Round {self.scm_current_round} complete: "
                              f"tested={tested}, confirmed total={confirmed}")
                        # 모두 시도했으면 정상 종료 허용

                # All checks passed - model is genuinely good enough
                print("\n[ACCEPTED] AI suggests stopping and model quality is sufficient:")
                min_status = 'OK' if minimization_ok else 'FAILED'
                print(f"  - Minimization: {min_status}")
                shrink_val = avg_eta_shrinkage
                thresh_val = shrinkage_threshold
                if shrink_val is not None:
                    print(f"  - Shrinkage: {shrink_val:.1f}% (threshold: <{thresh_val}%)")
                else:
                    print(f"  - Shrinkage: N/A (threshold: <{thresh_val}%)")
                if self.improvement_history and 'ai_evaluation' in self.improvement_history[-1]:
                    last_eval = self.improvement_history[-1]['ai_evaluation']
                    quality_score = last_eval.get('quality_score', 0)
                    print(f"  - Quality Score: {quality_score}/100")
                iter_val = self.iteration
                print(f"  - Iterations: {iter_val}")
                return False
            else:
                # AI says continue - but check if we should override to STOP
                # CRITICAL: If only 1 OMEGA left and shrinkage is acceptable, must STOP
                if len(omega_values) == 1 and avg_eta_shrinkage is not None:
                    # For small datasets (N<20), shrinkage 50-75% is acceptable
                    # Further simplification would destroy the model
                    metadata = self.data_loader.get_metadata()
                    num_subjects = metadata.get('n_subjects', 100)

                    if num_subjects < 20 and avg_eta_shrinkage < 75:
                        n_val = num_subjects
                        shrink_val = avg_eta_shrinkage
                        print(f"\n{'='*70}")
                        print("[CRITICAL OVERRIDE] AI suggested CONTINUE, but STOPPING instead")
                        print(f"{'='*70}")
                        print(f"  Reason:")
                        print(f"    - Only 1 OMEGA parameter remains")
                        print(f"    - Dataset size: {n_val} subjects (small)")
                        if shrink_val is not None:
                            print(f"    - Shrinkage {shrink_val:.1f}% is ACCEPTABLE for N<20")
                        else:
                            print(f"    - Shrinkage N/A% is ACCEPTABLE for N<20")
                        print(f"    - Further simplification would remove last OMEGA")
                        print(f"    - Model would become non-population (NONMEM will fail)")
                        print(f"  Decision: STOP to preserve model viability")
                        print(f"{'='*70}")
                        return False

                # Otherwise trust AI's decision to continue
                return True

        # Fallback: Traditional decision logic if AI evaluation failed
        print("\n[WARNING] AI evaluation unavailable, using fallback logic...")

        # Default: Keep trying to improve
        if not minimization_ok:
            print("\n[WARNING] Minimization not successful - continuing optimization")
            return True

        if issues:
            print(f"\n[INFO] Found {len(issues)} issue(s) - continuing to address them")
            return True

        # Even if things look ok, keep going unless really converged
        if len(self.improvement_history) >= 2:
            prev_entry = self.improvement_history[-2]
            prev_ofv = prev_entry.get('ofv')

            if current_ofv is not None and prev_ofv is not None:
                ofv_change = current_ofv - prev_ofv
                print(f"\nOFV change: {ofv_change:.2f}")

                # Only stop if truly converged (very small change + good quality)
                if abs(ofv_change) < 0.1 and avg_eta_shrinkage is not None and avg_eta_shrinkage < 50:
                    print("  Model appears to have converged with good quality")
                    return False

        # Default: Continue optimization
        print("\n[DEFAULT] Continuing optimization")
        return True

    def _ai_quality_check(self, parsed_data: Dict,
                      max_rse: Optional[float] = None,
                      high_rse_count: int = 0,
                      lst_output: str = "") -> Optional[dict]:
        """
        Use AI to evaluate model quality comprehensively

        Args:
            parsed_data: Parsed NONMEM results dictionary

        Returns:
            True to continue optimization, False to stop, None if evaluation failed
        """
        import json

        # Pre-check: Detect obvious overfitting before AI call
        ofv = parsed_data.get('objective_function')
        eta_shrinkage = parsed_data.get('eta_shrinkage', [])
        avg_shrink = max([s['shrinkage'] for s in eta_shrinkage]) if eta_shrinkage else None
        params = parsed_data.get('parameter_estimates', {})
        omega_values = [float(o.get('estimate', 1.0)) for o in params.get('omega', []) if 'estimate' in o]

        # Detect critical overfitting signals
        overfitting_warnings = []
        if ofv is not None and ofv < -50:
            overfitting_warnings.append(f"CRITICAL: Negative OFV ({ofv:.2f}) indicates overfitting")
        if avg_shrink is not None and avg_shrink > 95:
            overfitting_warnings.append(f"CRITICAL: Catastrophic shrinkage ({avg_shrink:.1f}%) - IIV lost")
        collapsed_omegas = [o for o in omega_values if o < 0.001]
        if len(collapsed_omegas) > 0:
            overfitting_warnings.append(f"CRITICAL: {len(collapsed_omegas)} OMEGA(s) collapsed - overparameterization")

        # ── Boundary + Covariance failure detection (Phase 1-4 only) ──────────
        # Phase 5 (SCM)에서는 이 패턴이 정상 SCM 결과:
        #   covariate THETA가 lower bound에 수렴 = 통계적으로 무의미한 covariate
        #   → SCM revert trigger가 이미 처리 → 여기서 감지 불필요 (noise만 추가)
        # Phase 1-4에서만 실행.
        nonmem_warnings = parsed_data.get('warnings', [])
        nonmem_errors   = parsed_data.get('errors', [])
        # MINIMIZATION TERMINATED 케이스에서 BOUNDARY가 warnings 대신 errors에 들어오는 경우 있음
        boundary_hit = any('BOUNDARY' in str(w).upper() for w in nonmem_warnings + nonmem_errors)
        cov_failed_now = not parsed_data.get('covariance_step', {}).get('successful', False)
        in_phase5 = (self.current_phase == ModelPhase.COVARIATE_ANALYSIS)

        if boundary_hit and cov_failed_now and not in_phase5:
            # Count consecutive covariance failures in recent history
            recent_cov_fails = sum(
                1 for h in self.improvement_history[-4:]
                if not h.get('covariance_successful', True)
            )

            if recent_cov_fails >= 2:
                # 어떤 THETA가 boundary에 걸렸는지 파악 (.lst 파일 직접 읽기)
                lst_file = f"{self.output_base}_iter{self.iteration}.lst"
                try:
                    with open(lst_file, 'r', encoding='utf-8', errors='ignore') as _f:
                        lst_output = _f.read()
                except Exception:
                    lst_output = ""

                boundary_indices = self._identify_boundary_theta(lst_output)
                theta_map = self._extract_theta_param_map(self.current_code)

                boundary_params = [
                    (idx, theta_map.get(idx, f"THETA({idx})"))
                    for idx in boundary_indices
                ]

                STRUCTURAL_PK = {'CL', 'V1', 'V2', 'V', 'Q', 'Ka', 'KA'}
                is_structural_boundary = any(
                    name in STRUCTURAL_PK for _, name in boundary_params
                )

                print(f"\n{'!'*70}")
                print("BOUNDARY→COVARIANCE FAILURE DETECTED")
                print(f"{'!'*70}")
                print(f"  Pattern: 'NEAR ITS BOUNDARY' + covariance failed x {recent_cov_fails}")

                if boundary_params:
                    names_str = ', '.join(
                        f"THETA({idx})={name}" for idx, name in boundary_params
                    )
                    print(f"  Boundary parameter(s): {names_str}")

                if is_structural_boundary:
                    struct_names = ', '.join(
                        name for _, name in boundary_params if name in STRUCTURAL_PK
                    )
                    struct_indices = [
                        idx for idx, name in boundary_params if name in STRUCTURAL_PK
                    ]
                    idx_str = ', '.join(f"THETA({i})" for i in struct_indices)
                    print(f"  Diagnosis: Structural PK parameter ({struct_names}) at lower bound")
                    print(f"  Passing context to LLM; LLM will determine appropriate fix")
                    print(f"{'!'*70}")
                    overfitting_warnings.append(
                        f"DIAGNOSTIC: Structural PK parameter ({struct_names}) [{idx_str}] "
                        f"converged to its lower bound for {recent_cov_fails} consecutive "
                        f"iterations, causing Hessian singularity and covariance failure. "
                        f"The error model is NOT the cause. "
                        f"Hard constraint: Do NOT change the error model structure."
                    )
                    overfitting_warnings.append("STRUCTURAL_BOUNDARY_DETECTED")
                    sig = f"STRUCTURAL_BOUNDARY_{struct_names.replace(', ', '_')}_FAILED"
                    if sig not in self.failed_strategies:
                        self.failed_strategies.append(sig)
                        print(f"  [TRACK] Recording failed strategy: {sig}")
                else:
                    print(f"  Diagnosis: Error model THETA at lower bound -> Hessian singular")
                    print(f"  Fix: Switch to proportional-only error model")
                    print(f"{'!'*70}")
                    overfitting_warnings.append(
                        f"CRITICAL: Error model parameter hit its boundary for "
                        f"{recent_cov_fails} consecutive iterations. "
                        "ERROR_MODEL_BOUNDARY_FIX: "
                        "Root cause: combined error model -- the additive error THETA "
                        "is collapsing to its lower bound (~0), making the Hessian "
                        "singular. MANDATORY FIX: Remove the additive error THETA and "
                        "use proportional-only error ($ERROR: W = THETA(n)*IPRED). "
                        "Do NOT try SAEM or re-parameterization -- this is an error "
                        "model issue, not an estimation algorithm issue."
                    )
                    self._track_combined_error_failure(self.current_code)


        if overfitting_warnings:
            print(f"\n{'!'*70}")
            print("OVERFITTING DETECTED - IMMEDIATE ACTION REQUIRED")
            print(f"{'!'*70}")
            for warning in overfitting_warnings:
                print(f"  {warning}")
            print(f"{'!'*70}")

        try:
            # Run plausibility check against drug-specific physiological bounds
            plausibility_report = self._check_plausibility(parsed_data, self.current_code)
            self.plausibility_report = plausibility_report  # store for later use

            # Print plausibility violations
            if plausibility_report.get("violations"):
                print(f"\n{'!'*70}")
                print("PHYSIOLOGICAL PLAUSIBILITY VIOLATIONS")
                print(f"Checked: {plausibility_report.get('checked_parameters', [])}")
                for v in plausibility_report["violations"]:
                    print(f"  {v}")
                print(f"Plausibility Score: {plausibility_report.get('plausibility_score', 'N/A')}/100")
                print(f"{'!'*70}")

            prompt = PromptTemplates.quality_evaluation_prompt(
                iteration=self.iteration,
                parsed_data=parsed_data,
                previous_improvements=self.improvement_history,
                overfitting_warnings=overfitting_warnings,
                max_rse=max_rse,
                high_rse_count=high_rse_count,
                plausibility_report=plausibility_report or None,
                lst_output=lst_output
            )

            # Use JSON mode for structured output
            response = self.gemini_client.generate(prompt, model_type=self.model)

            # Extract JSON from code block if present
            json_text = response.strip()

            # Remove markdown code block markers if present
            if json_text.startswith('```'):
                # Find the actual JSON content
                lines = json_text.split('\n')
                # Skip first line (```json or ```)
                lines = lines[1:]
                # Remove last line if it's ```
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                json_text = '\n'.join(lines).strip()

            # Parse JSON response
            evaluation = json.loads(json_text)

            should_continue = evaluation.get('should_continue', True)

            # Python이 직접 비중 계산 (LLM 위임 안 함)
            WEIGHTS = {
                'convergence': 0.20,
                'precision':   0.20,
                'shrinkage':   0.20,
                'stability':   0.20,
                'utility':     0.20,
            }
            score_breakdown_raw = evaluation.get('score_breakdown', {})
            quality_score = round(sum(
                score_breakdown_raw.get(dim, 0) * weight
                for dim, weight in WEIGHTS.items()
            ))

            if   quality_score >= 90: grade = 'A'
            elif quality_score >= 80: grade = 'B'
            elif quality_score >= 70: grade = 'C'
            elif quality_score >= 60: grade = 'D'
            elif quality_score >= 45: grade = 'E'
            else:                     grade = 'F'
            
            reason = evaluation.get('decision_reason', '')
            score_breakdown = score_breakdown_raw
            critical_issues = evaluation.get('critical_issues', [])
            recommendations = evaluation.get('recommendations_if_continuing', [])

            # Inject hardcoded boundary fix recommendation if pattern detected
            # This overrides LLM recommendations when root cause is clear
            if any('ERROR_MODEL_BOUNDARY_FIX' in str(w) for w in overfitting_warnings):
                boundary_fix = (
                    "MANDATORY: Switch to proportional-only error model. "
                    "Remove the additive error THETA from $THETA and simplify $ERROR to: "
                    "W = THETA(n) * IPRED  (proportional-only, no additive term). "
                    "This is the specific fix for the boundary-induced covariance failure. "
                    "Do NOT change estimation method (no SAEM), do NOT re-parameterize structure."
                )
                # Prepend as highest-priority recommendation
                recommendations = [boundary_fix] + [
                    r for r in recommendations
                    if 'SAEM' not in r.upper() and 'METHOD' not in r.upper()
                ]
                critical_issues = [
                    "Additive error THETA at lower boundary → Hessian singular → covariance failure. "
                    "Root cause: combined error model not supported by data. "
                    "Use proportional-only error."
                ] + critical_issues

            elif any('STRUCTURAL_BOUNDARY_FIX' in str(w) for w in overfitting_warnings):
                # 구조 파라미터 boundary → error model 건드리지 말고 THETA 범위 수정
                struct_fix = (
                    "MANDATORY: Do NOT change the error model. "
                    "A structural PK parameter (Ka, CL, V, etc.) is hitting its lower bound. "
                    "Fix options: (1) Widen the lower bound for the offending THETA in $THETA "
                    "(e.g., change (0.1, 1.0, 10) to (0.001, 1.0, 10)), OR "
                    "(2) Use the final parameter estimates from the best previous iteration "
                    "as new initial values to start closer to the true minimum. "
                    "Do NOT switch error models, do NOT add SAEM."
                )
                recommendations = [struct_fix] + [
                    r for r in recommendations
                    if 'ERROR MODEL' not in r.upper() and 'SAEM' not in r.upper()
                ]
                critical_issues = [
                    "Structural PK parameter at lower boundary → Hessian singular → covariance failure. "
                    "Root cause: THETA lower bound too restrictive. "
                    "Widen the bound or use previous best estimates as initials."
                ] + critical_issues

            # Display evaluation results
            print(f"\n{'=' * 70}")
            print("AI QUALITY EVALUATION")
            print(f"{'=' * 70}")
            print(f"Quality Score: {quality_score}/100")
            print(f"Model Grade: {grade}")
            print(f"\nScore Breakdown:")
            for criterion, score in score_breakdown.items():
                print(f"  - {criterion.capitalize()}: {score}/100")

            print(f"\nDecision: {'CONTINUE OPTIMIZATION' if should_continue else 'STOP - MODEL GOOD ENOUGH'}")
            print(f"Reason: {reason}")

            if critical_issues:
                print(f"\nCritical Issues:")
                for issue in critical_issues:
                    print(f"  - {issue}")

            if should_continue and recommendations:
                print(f"\nRecommendations:")
                for rec in recommendations:
                    print(f"  - {rec}")

            print(f"{'=' * 70}")

            # Store evaluation in history
            if self.improvement_history:
                self.improvement_history[-1]['ai_evaluation'] = {
                    'quality_score': quality_score,
                    'grade': grade,
                    'should_continue': should_continue,
                    'reason': reason
                }

            return {
                'should_continue': should_continue,
                'critical_issues': critical_issues,
                'recommendations': recommendations
            }

        except json.JSONDecodeError as e:
            print(f"[WARNING] AI quality evaluation returned invalid JSON: {e}")
            if response:
                snippet = response[:200]
            else:
                snippet = 'N/A'
            print(f"  Response snippet: {snippet}...")
            return None

        except Exception as e:
            print(f"[WARNING] AI quality evaluation failed: {e}")
            return None

    def _detect_simplification_needed(self) -> tuple[bool, str]:
        """
        Detect if mandatory simplification is required based on history

        Returns:
            (is_mandatory, reason) tuple
        """
        # Phase 5 (Covariate Analysis)에서는 simplification 불필요:
        # - SCM은 covariate 추가만 하므로 "단순화" 개념이 적용 안 됨
        # - 이전 실패 iteration의 shrinkage 수치가 history에 남아 false positive 유발
        if self.current_phase == ModelPhase.COVARIATE_ANALYSIS:
            return False, ""

        if len(self.improvement_history) < 2:
            return False, ""

        # Check recent shrinkage history
        recent_shrinkage = []
        for entry in self.improvement_history[-3:]:
            shrink = entry.get('avg_eta_shrinkage')
            if shrink is not None:
                recent_shrinkage.append(shrink)

        # CRITICAL: 2+ consecutive iterations with >90% shrinkage
        extreme_shrink_count = sum(1 for s in recent_shrinkage if s > 90)
        if extreme_shrink_count >= 2:
            avg_shrink = max(recent_shrinkage)
            reason = (
                f"MANDATORY SIMPLIFICATION: ETA shrinkage >{90}% for {extreme_shrink_count} consecutive iterations "
                f"(average: {avg_shrink:.1f}%). The model is severely overparameterized. "
                f"You MUST reduce model complexity - DO NOT just adjust boundaries or initial values."
            )
            return True, reason

        # Check for persistent covariance failures with high shrinkage
        cov_failures = [e for e in self.improvement_history[-3:]
                       if not e.get('covariance_successful', False)]
        if len(cov_failures) >= 2 and recent_shrinkage and recent_shrinkage[-1] > 70:
            reason = (
                f"MANDATORY SIMPLIFICATION: Covariance failed {len(cov_failures)} times "
                f"with shrinkage >{70}%. Model complexity exceeds data information content. "
                f"You MUST simplify the random effects structure."
            )
            return True, reason

        # Check for OMEGA collapse (informational only; do not force simplification)
        if self.improvement_history:
            last_entry = self.improvement_history[-1]
            omega_values = last_entry.get('omega_values', [])
            collapsed = [o for o in omega_values if o < 0.0001]
            if len(collapsed) >= len(omega_values) // 2 and len(omega_values) > 0:
                reason = (
                    f"SUGGESTED REVIEW: {len(collapsed)}/{len(omega_values)} OMEGA parameters "
                    f"are very small (<0.0001). This may indicate limited information for some random effects. "
                    f"Consider reviewing the structural and random-effects model; simplification of clearly "
                    f"redundant ETAs can be considered, but avoid blindly fixing OMEGA to zero."
                )
                print(f"[INFO] {reason}")

        return False, ""

    def _get_phase_specific_guidance(self) -> str:
        """
        Generate phase-specific guidance for the improvement prompt

        Returns:
            Focused guidance text for current phase
        """
        if self.current_phase == ModelPhase.ESTABLISH_BASE:
            return """
**CURRENT FOCUS: ESTABLISH BASE MODEL (Phase 1)**

Priority: Get minimization working
Actions ONLY:
1. Fix syntax errors (missing K=CL/V, S2=V, wrong $INPUT order)
2. Fix parameter boundaries (THETA out of bounds -> adjust or FIX)
3. Fix estimation convergence (try METHOD=ZERO if METHOD=1 INTER fails)
4. Ensure NONMEM executes without fatal errors

DO NOT:
- Change structural model (ADVAN)
- Add/remove OMEGA
- Add covariates
- Make complex changes

Goal: Minimization successful, even if quality is poor
"""
        elif self.current_phase == ModelPhase.DIAGNOSE_STRUCTURE:
            return """
**CURRENT FOCUS: DIAGNOSE STRUCTURAL MODEL (Phase 2)**

Priority: Check if compartment structure is adequate
Actions:
1. Review NONMEM output for systematic patterns
2. If residuals show bi-phasic decline -> Consider ADVAN2->ADVAN4 (2-compartment)
3. If residuals show U-shape -> Structure inadequate
4. Check error model adequacy (proportional vs combined)
5. Fix boundary issues on structural parameters (Ka, CL, V)

Key indicators:
- Flat random residuals -> Structure OK, proceed to Phase 4
- Systematic bias -> Need structural change
- Funnel-shaped residuals -> Add proportional error component

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
ABSOLUTE PROHIBITIONS IN PHASE 2 (cannot be overridden):
  ❌ DO NOT add any covariate (WT, AGE, SEX, CLCR, etc.)
     Covariates belong EXCLUSIVELY in Phase 5 — not here.
  ❌ DO NOT act on AI quality evaluation recommendations
     that say "add covariates" or "investigate covariates"
     — those suggestions are for Phase 5, not Phase 2.
  ❌ DO NOT remove OMEGA (unless shrinkage >90%)
  ❌ DO NOT change compartment count unless residuals
     clearly show bi-phasic decline
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Goal: Stable structural model (successful minimization + covariance step)
"""
        elif self.current_phase == ModelPhase.REDUCE_OVERFITTING:
            return """
**CURRENT FOCUS: REDUCE OVERFITTING (Phase 3)**

CRITICAL: Model shows signs of overfitting - consider simplification

Priority: Reduce model complexity CAREFULLY
Suggested actions (check OFV after each change):
1. **IF OMEGA count >1**: Consider removing 1 OMEGA (highest shrinkage first)
   - WARNING: Only remove if OFV remains similar or improves
   - If OFV worsens by >30%, model may already be optimal

2. Simplify error model if using combined (additive+proportional)
   - Try proportional-only first

3. Consider fixing problematic THETA to typical value
   - If parameter hitting upper/lower bound repeatedly

4. Try METHOD=ZERO if using METHOD=1 INTER
   - More robust for small datasets

Overfitting indicators:
- Shrinkage >90% AND model stable/improving
- OFV <-50 (extremely negative)
- OMEGA <0.001 (collapsed)

**CRITICAL CHECK**: After simplification, compare OFV to previous iteration:
- OFV similar/better -> Simplification successful ✓
- OFV much worse (>30%) -> Simplification too aggressive, revert strategy ✗

IMPORTANT: Keep at least 1 OMEGA for population model
"""
        elif self.current_phase == ModelPhase.OPTIMIZE_IIV:
            return """
**CURRENT FOCUS: OPTIMIZE RANDOM EFFECTS (Phase 4)**

Priority: Fine-tune IIV structure
Actions:
1. Check shrinkage for each ETA
   - >70%: Consider removing if OMEGA>1
   - <50%: Good, retain
2. Optimize OMEGA structure
   - For N<30: Keep DIAGONAL
   - For N>30: Can try BLOCK if warranted
3. Adjust THETA bounds if parameters near boundaries
4. Fine-tune estimation method if needed

Goal: Shrinkage <50-60% with stable estimates

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
ABSOLUTE PROHIBITIONS IN PHASE 4 (cannot be overridden):
  ❌ DO NOT add any covariate (WT, AGE, SEX, CLCR, etc.)
     Covariates belong EXCLUSIVELY in Phase 5 — not here.
  ❌ DO NOT change ADVAN or compartment structure
  ❌ DO NOT act on AI recommendations that say "add covariates"
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""
        elif self.current_phase == ModelPhase.COVARIATE_ANALYSIS:
            # SCM 현황 텍스트 생성
            scm_status = self._build_scm_status_text()

            # Phase5Covariates 상세 프롬프트 생성
            metadata = self.data_loader.get_metadata()
            last_entry = self.improvement_history[-1] if self.improvement_history else {}
            parsed_results = last_entry.get('parsed_data', {})
            shrinkage_data = parsed_results.get('eta_shrinkage', []) if parsed_results else []

            # Backward elimination: current_covariate_instruction has mode='remove'
            # → generate a removal prompt instead of an addition prompt.
            if self.current_covariate_instruction and self.current_covariate_instruction.get('mode') == 'remove':
                target = self.current_covariate_instruction
                remaining = [c['name'] for c in self.scm_confirmed]
                phase5_prompt = Phase5Covariates.generate_removal_prompt(
                    iteration=self.iteration,
                    current_code=self.current_code,
                    parsed_results=parsed_results,
                    target_covariate_name=target.get('name', ''),
                    remaining_covariates=remaining,
                    n_subjects=metadata.get('n_subjects', 0)
                )
                return f"{scm_status}\n\n{phase5_prompt}"

            # 이번 iter에서 테스트할 단일 covariate만 넘김
            # 전체 목록을 넘기면 LLM이 약리학적 선호에 따라 다른 covariate를 선택할 수 있음
            if self.current_covariate_instruction:
                target = self.current_covariate_instruction
                cov_name = target.get('covariate', '')
                covariate_info = metadata.get('covariate_info', {})
                info = covariate_info.get(cov_name, {})
                cov_type = info.get('type', 'continuous')
                med = info.get('median', None)
                cov_min = info.get('min', '?')
                cov_max = info.get('max', '?')
                available_covariates = [{
                    'name': cov_name,
                    'type': cov_type,
                    'parameter': target.get('parameter', 'CL'),
                    'median': round(med, 1) if med is not None else '?',
                    'min': round(cov_min, 1) if isinstance(cov_min, float) else cov_min,
                    'max': round(cov_max, 1) if isinstance(cov_max, float) else cov_max,
                    'suggested_model': target.get('model_type', 'power' if cov_type == 'continuous' else 'categorical')
                }]
            else:
                available_covariates = self._build_available_covariates_for_phase5()

            phase5_prompt = Phase5Covariates.generate_prompt(
                iteration=self.iteration,
                current_code=self.current_code,
                parsed_results=parsed_results,
                shrinkage_data=shrinkage_data,
                available_covariates=available_covariates,
                current_covariates_in_model=[c['name'] for c in self.scm_confirmed],
                n_subjects=metadata.get('n_subjects', 0)
            )

            return f"{scm_status}\n\n{phase5_prompt}"
        else:
            return ""



    # -------------------------------------------------------------------------
    # SCM (Stepwise Covariate Modeling) Methods — Phase 5
    # -------------------------------------------------------------------------

    def _get_user_selected_covariates(self) -> Optional[Dict[str, Optional[set]]]:
        """Return requested covariate/parameter limits, or None for auto mode."""
        settings = self.prior_info.get('covariates', {})
        if not isinstance(settings, dict) or settings.get('mode', 'auto') != 'user_selected':
            return None

        selected = {}
        for item in settings.get('candidates', []):
            if isinstance(item, str):
                name, parameters = item, None
            elif isinstance(item, dict):
                name = item.get('name')
                parameters = item.get('target_parameters')
            else:
                continue
            if not isinstance(name, str) or not name.strip():
                continue
            if parameters is None:
                selected[name.upper()] = None
                continue
            if isinstance(parameters, str):
                parameters = [parameters]
            if isinstance(parameters, list):
                selected[name.upper()] = {str(parameter).upper() for parameter in parameters}
        return selected

    def _print_covariate_selection_summary(self) -> None:
        """Report which SCM candidate policy will be applied to this dataset."""
        selected = self._get_user_selected_covariates()
        if selected is None:
            print("SCM covariates: auto-detected dataset covariates")
            return

        metadata = self.data_loader.get_metadata()
        available = {
            (item if isinstance(item, str) else item.get('name', '')).upper()
            for item in metadata.get('covariates', [])
        }
        requested = set(selected)
        missing = sorted(requested - available)
        if not requested:
            print("[WARNING] covariates.mode is user_selected but no candidates were supplied; SCM will have no candidates.")
        else:
            print(f"SCM covariates: user-selected only ({', '.join(sorted(requested))})")
        if missing:
            print(f"[WARNING] Requested covariates not found in dataset: {', '.join(missing)}")

    def _is_selected_covariate_parameter(self, covariate: str, parameter: str) -> bool:
        selected = self._get_user_selected_covariates()
        if selected is None:
            return True
        allowed = selected.get(covariate.upper())
        return covariate.upper() in selected and (allowed is None or parameter.upper() in allowed)

    def _get_all_covariate_candidates(self) -> list:
        """데이터셋 공변량 × PK 파라미터 조합으로 SCM 후보 목록 생성.

        공변량 모델 타입(power/linear/categorical)은 data_loader.py의
        suggest_covariate_model_type()이 결정한 값을 그대로 사용한다
        (single source of truth — 여기서 다시 판단하지 않음).
        """
        metadata = self.data_loader.get_metadata()
        covariates = metadata.get('covariates', [])
        covariate_info = metadata.get('covariate_info', {})
        # WT를 포함한 모든 covariate이 CL/V1 둘 다에 대해 동일한 자격으로
        # SCM 후보가 된다 — allometric scaling도 예외 없이 forward/backward
        # 유의성 검증을 통과해야만 채택됨 (사전에 무조건 고정 반영하지 않음).

        cov_names = []
        for c in covariates:
            name = c if isinstance(c, str) else c.get('name', str(c))
            if name:
                cov_names.append(name)

        candidates = []
        for cov in cov_names:
            allow_cl = self._is_selected_covariate_parameter(cov, 'CL')
            allow_v1 = self._is_selected_covariate_parameter(cov, 'V1')
            if not allow_cl and not allow_v1:
                continue
            info = covariate_info.get(cov, {})
            model_type = info.get('suggested_model', 'linear')

            if model_type == 'categorical':
                cl_example = f'IF({cov}.EQ.1) TVCL = TVCL * THETA(X)'
                v1_example = f'IF({cov}.EQ.1) TVV1 = TVV1 * THETA(X)'
            elif model_type == 'power':
                cl_example = f'TVCL = THETA(1) * ({cov}/REF) ** THETA(X)'
                v1_example = f'TVV1 = THETA(2) * ({cov}/REF) ** THETA(X)'
            else:  # linear
                cl_example = f'TVCL = THETA(1) * (1 + THETA(X) * ({cov} - MEDIAN))'
                v1_example = f'TVV1 = THETA(2) * (1 + THETA(X) * ({cov} - MEDIAN))'

            if allow_cl:
                candidates.append({
                    'name': f'{cov} on CL',
                    'covariate': cov,
                    'parameter': 'CL',
                    'model_type': model_type,
                    'example': cl_example
                })
            if allow_v1:
                candidates.append({
                    'name': f'{cov} on V1',
                    'covariate': cov,
                    'parameter': 'V1',
                    'model_type': model_type,
                    'example': v1_example
                })
        return candidates

    def _get_next_covariate_to_test(self):
        """
        현재 라운드에서 다음으로 시도할 covariate dict 반환. 없으면 None.

        forward 모드: 아직 확정되지 않고 이번 라운드에 테스트되지 않은 추가 후보.
        backward 모드: 이미 확정된 covariate 중 이번 backward 라운드에서
                       제거 테스트가 안 된 것 (mode='remove'로 표시).
        """
        if self.scm_mode == 'backward':
            for c in self.scm_confirmed:
                if c['name'] in self.scm_round_tested:
                    continue
                return {
                    'name': c['name'],
                    'covariate': c.get('covariate', ''),
                    'parameter': c.get('parameter', ''),
                    'model_type': c.get('model_type', ''),
                    'mode': 'remove',
                }
            return None

        confirmed_names = {c['name'] for c in self.scm_confirmed}
        for cand in self._get_all_covariate_candidates():
            if cand['name'] in confirmed_names:
                continue
            if cand['name'] in self.scm_round_tested:
                continue
            cand = dict(cand)
            cand['mode'] = 'add'
            return cand
        return None

    def _detect_added_covariate(self, prev_code: str, curr_code: str) -> Optional[dict]:
        """
        $PK 블록 diff를 통해 LLM이 실제로 추가한 covariate를 감지.
        지시한 것과 실제로 한 것이 다를 경우 정확한 기록을 위해 사용.
        """
        def extract_pk_lines(code: str) -> set:
            """$PK 블록에서 주석 제거 후 의미있는 라인 추출"""
            pk_match = re.search(r'\$PK(.*?)(?=\n\s*\$|\Z)', code, re.DOTALL | re.IGNORECASE)
            if not pk_match:
                return set()
            lines = set()
            for line in pk_match.group(1).split('\n'):
                # 주석 제거
                if ';' in line:
                    line = line[:line.index(';')]
                line = line.strip().upper()
                if line:
                    lines.add(line)
            return lines

        prev_lines = extract_pk_lines(prev_code or '')
        curr_lines = extract_pk_lines(curr_code or '')
        added_lines = curr_lines - prev_lines

        if not added_lines:
            return None

        # 추가된 라인에서 covariate-parameter 조합 감지
        metadata = self.data_loader.get_metadata()
        covariates = metadata.get('covariates', [])
        cov_names = [c if isinstance(c, str) else c.get('name', '') for c in covariates]

        # PK 파라미터 키워드 (TVCL → CL, TVV1 → V1 등)
        param_map = {
            'TVCL': 'CL', 'TVV1': 'V1', 'TVV': 'V1',
            'TVQ': 'Q', 'TVV2': 'V2', 'TVKA': 'Ka',
        }

        for line in added_lines:
            for cov in cov_names:
                if cov.upper() not in line:
                    continue
                # 어떤 파라미터에 적용됐는지 확인
                for tv_key, param in param_map.items():
                    if tv_key in line:
                        name = f'{cov} on {param}'
                        # candidates 목록에서 매칭 항목 찾기
                        all_cands = self._get_all_covariate_candidates()
                        matched = next((c for c in all_cands if c['name'] == name), None)
                        if matched:
                            return matched
                        # 목록에 없는 경우 기본 dict 반환
                        return {
                            'name': name,
                            'covariate': cov,
                            'parameter': param,
                            'model_type': 'unknown',
                            'example': ''
                        }
        return None

    def _record_phase5_covariate_result(self, current_ofv, cov_success: bool = False) -> None:
        """
        SCM: 현재 라운드에서 테스트한 covariate의 결과를 기록한다.

        ACCEPT/REJECT 판정은 여기서 하지 않는다 — 라운드가 끝난 후
        _complete_scm_round()에서 모든 후보의 ΔOFV를 비교하여 winner를 선택한다.

        여기서는 ΔOFV 계산 + scm_round_results에 기록만 수행.
        """
        if self.current_covariate_instruction is None:
            return
        if self.current_phase != ModelPhase.COVARIATE_ANALYSIS:
            return

        cand = self.current_covariate_instruction
        name = cand['name']
        base_ofv = self.scm_round_base_ofv

        if current_ofv is None or base_ofv is None:
            delta_ofv = None
            cov_success = False
        else:
            delta_ofv = current_ofv - base_ofv

        mode = cand.get('mode', 'add')

        self.scm_round_tested.add(name)
        self.scm_round_results.append({
            'name': name,
            'covariate': cand.get('covariate', ''),
            'parameter': cand.get('parameter', ''),
            'model_type': cand.get('model_type', ''),
            'mode': mode,
            'ofv': current_ofv,
            'delta_ofv': delta_ofv,
            'cov_ok': cov_success,
            'code': self.current_code,
            'iteration': self.iteration,
            'round': self.scm_current_round,
        })
        self.covariate_history.append({
            'name': name,
            'covariate': cand.get('covariate', ''),
            'parameter': cand.get('parameter', ''),
            'mode': mode,
            'delta_ofv': delta_ofv,
            'cov_ok': cov_success,
            'result': 'TESTED',
            'iteration': self.iteration,
            'round': self.scm_current_round,
        })

        label = "Backward Round" if mode == 'remove' else "Round"
        delta_str = f"ΔOFV={delta_ofv:.2f}" if delta_ofv is not None else "ΔOFV=N/A"
        cov_str = "cov=OK" if cov_success else "cov=FAIL"
        print(f"  [SCM {label} {self.scm_current_round}] {name}: {delta_str}, {cov_str}")

        self.current_covariate_instruction = None

    def _build_scm_status_text(self) -> str:
        """LLM 프롬프트에 삽입할 SCM 현황 텍스트 생성 (forward/backward 모드에 따라 분기)"""
        if self.scm_mode == 'backward':
            return self._build_backward_status_text()

        confirmed_names = {c['name'] for c in self.scm_confirmed}

        lines = [
            "=" * 65,
            f"SCM STATUS — Round {self.scm_current_round} (Forward Selection)",
            "  Acceptance threshold: ΔOFV < -3.84 (p<0.05, df=1)",
            "=" * 65,
            "",
            "🔒 BASE MODEL STRUCTURE IS FROZEN:",
            "  ❌ DO NOT change $ESTIMATION (must remain FOCE-I / METHOD=1 INTER)",
            "  ❌ DO NOT change $ERROR or $SIGMA (residual error model is fixed)",
            "  ❌ DO NOT change $OMEGA (IIV structure is fixed)",
            "  ❌ DO NOT change ADVAN or compartment structure",
            "  ✅ ONLY allowed action: add ONE new covariate THETA as shown below",
            "",
        ]

        # 확정된 covariates (이전 라운드 winners — base code에 이미 포함됨)
        if self.scm_confirmed:
            lines.append("  CONFIRMED (already in base code from prior rounds):")
            for c in self.scm_confirmed:
                d = f"ΔOFV={c['delta_ofv']:.2f}" if c.get('delta_ofv') is not None else ""
                lines.append(f"    ✅ {c['name']} (Round {c['round']}, {d})")
            lines.append("")

        # 현재 라운드 후보 상태
        lines.append(f"  ROUND {self.scm_current_round} CANDIDATES:")
        for cand in self._get_all_covariate_candidates():
            name = cand['name']
            if name in confirmed_names:
                continue
            if name in self.scm_round_tested:
                entry = next((r for r in self.scm_round_results if r['name'] == name), None)
                if entry and entry.get('delta_ofv') is not None:
                    lines.append(f"    ✔ TESTED : {name} (ΔOFV={entry['delta_ofv']:.2f})")
                else:
                    lines.append(f"    ✔ TESTED : {name} (ΔOFV=N/A)")
            else:
                lines.append(f"    ⬜ UNTESTED : {name}")

        next_cand = self._get_next_covariate_to_test()
        lines.append("=" * 65)

        if next_cand:
            lines.append(f"▶▶ MANDATORY THIS ITERATION: Test '{next_cand['name']}'")
            lines.append(f"   Parameter : {next_cand['parameter']}")
            lines.append(f"   Model type: {next_cand['model_type']}")
            lines.append(f"   Example   : {next_cand['example']}")
            lines.append(f"   Add this ONE covariate to the base model code below.")
            lines.append(f"   DO NOT add any other covariate. DO NOT modify model structure.")
        else:
            lines.append("▶▶ ROUND COMPLETE — all candidates tested this round.")

        lines.append("=" * 65)
        return "\n".join(lines)

    def _build_backward_status_text(self) -> str:
        """Backward elimination 라운드용 SCM 현황 텍스트"""
        lines = [
            "=" * 65,
            f"SCM STATUS — Backward Round {self.scm_current_round} (Elimination)",
            f"  Retention threshold: ΔOFV on removal >= {self.scm_backward_threshold} (p<0.01, df=1)",
            "  A covariate is ELIMINATED if removing it costs LESS than this — i.e.",
            "  it was not truly significant, only appeared so during forward selection.",
            "=" * 65,
            "",
            "🔒 BASE MODEL STRUCTURE IS FROZEN:",
            "  ❌ DO NOT change $ESTIMATION, $ERROR, $SIGMA, $OMEGA, ADVAN, or compartment count",
            "  ✅ ONLY allowed action: remove ONE covariate term as shown below",
            "",
        ]

        if self.scm_eliminated:
            lines.append("  ALREADY ELIMINATED (prior backward rounds):")
            for e in self.scm_eliminated:
                lines.append(f"    ❌ {e['name']} (Round {e['round']}, ΔOFV on removal=+{e['delta_ofv']:.2f})")
            lines.append("")

        lines.append(f"  STILL IN MODEL (backward round {self.scm_current_round} candidates):")
        for c in self.scm_confirmed:
            name = c['name']
            if name in self.scm_round_tested:
                entry = next((r for r in self.scm_round_results if r['name'] == name), None)
                if entry and entry.get('delta_ofv') is not None:
                    lines.append(f"    ✔ TESTED : {name} (ΔOFV on removal=+{entry['delta_ofv']:.2f})")
                else:
                    lines.append(f"    ✔ TESTED : {name} (ΔOFV=N/A)")
            else:
                lines.append(f"    ⬜ UNTESTED : {name}")

        next_cand = self._get_next_covariate_to_test()
        lines.append("=" * 65)

        if next_cand:
            lines.append(f"▶▶ MANDATORY THIS ITERATION: Test REMOVAL of '{next_cand['name']}'")
            lines.append(f"   Remove this ONE covariate from the full model code below.")
            lines.append(f"   DO NOT remove any other covariate. DO NOT modify model structure.")
        else:
            lines.append("▶▶ BACKWARD ROUND COMPLETE — all remaining covariates tested for removal.")

        lines.append("=" * 65)
        return "\n".join(lines)

    def _build_available_covariates_for_phase5(self) -> list:
        """Phase5Covariates.generate_prompt()에 전달할 covariate 정보 구성
        data_loader의 실제 통계(median, min, max, type)를 사용"""
        metadata = self.data_loader.get_metadata()
        covariates = metadata.get('covariates', [])
        covariate_info = metadata.get('covariate_info', {})  # 실제 데이터 통계

        result = []
        selected = self._get_user_selected_covariates()
        for c in covariates:
            name = c if isinstance(c, str) else c.get('name', str(c))
            if selected is not None and name.upper() not in selected:
                continue
            info = covariate_info.get(name, {})

            # 실제 데이터에서 계산된 값 사용, 없으면 None.
            # suggested_model은 data_loader.suggest_covariate_model_type()이 이미
            # 확정한 값 — 여기서 다시 판단하지 않는다 (single source of truth).
            cov_type = info.get('type', 'continuous')
            median = info.get('median', None)
            cov_min = info.get('min', '?')
            cov_max = info.get('max', '?')
            suggested = info.get('suggested_model', 'linear' if cov_type == 'continuous' else 'categorical')

            result.append({
                'name': name,
                'type': cov_type,
                'median': round(median, 1) if median is not None else '?',
                'min': round(cov_min, 1) if isinstance(cov_min, float) else cov_min,
                'max': round(cov_max, 1) if isinstance(cov_max, float) else cov_max,
                'suggested_model': suggested
            })
        return result

    def _complete_scm_round(self) -> bool:
        """
        현재 SCM 라운드(forward 또는 backward) 완료 후 처리.

        forward: scm_round_results에 기록된 모든 후보를 비교하여
        - 유의한 후보(ΔOFV < -3.84, cov 성공) 중 가장 큰 감소폭을 보인 winner 선택
        - winner가 있으면: scm_confirmed에 추가, base를 winner code/OFV로 갱신,
          다음 라운드 준비 → True 반환 (계속 진행)
        - winner가 없으면: forward selection 종료 → False 반환
          (main loop에서 scm_confirmed가 있으면 backward elimination으로 전환)

        backward: _complete_backward_round()로 위임 (반대 방향 로직).

        covariate_history의 result 필드도 이 시점에 최종 확정.
        """
        if self.scm_mode == 'backward':
            return self._complete_backward_round()

        rnd = self.scm_current_round

        # 유의한 후보 필터링: ΔOFV < -3.84 AND covariance 성공
        significant = [r for r in self.scm_round_results
                       if r.get('delta_ofv') is not None
                       and r['delta_ofv'] < -3.84
                       and r.get('cov_ok', False)]

        if not significant:
            # winner 없음 → SCM 종료
            print(f"\n{'='*70}")
            print(f"SCM ROUND {rnd} COMPLETE — NO WINNER")
            print(f"{'='*70}")
            print(f"  No covariate passed threshold (ΔOFV < -3.84 with covariance success)")
            print(f"  Base OFV: {self.scm_round_base_ofv:.2f}")
            print(f"  Tested: {[r['name'] for r in self.scm_round_results]}")
            print(f"{'='*70}")
            # base 모델 유지
            if self.scm_round_base_code:
                self.current_code = self.scm_round_base_code
                self.best_code = self.scm_round_base_code
            # covariate_history 결과 확정
            for r in self.scm_round_results:
                self._finalize_covariate_result(r['name'], rnd, 'REJECTED')
            return False

        # winner 선택: 가장 큰 ΔOFV 감소폭 (most negative)
        winner = min(significant, key=lambda r: r['delta_ofv'])
        winner_name = winner['name']
        winner_delta = winner['delta_ofv']
        winner_code = winner.get('code')
        winner_ofv = winner.get('ofv')
        winner_iter = winner.get('iteration')

        if not winner_code:
            print(f"\n[SCM Round {rnd} WARNING] Saved code not found for '{winner_name}' — keeping base")
            if self.scm_round_base_code:
                self.current_code = self.scm_round_base_code
                self.best_code = self.scm_round_base_code
            for r in self.scm_round_results:
                self._finalize_covariate_result(r['name'], rnd, 'REJECTED')
            return False

        # winner 확정 (covariate/parameter/model_type도 저장 — backward elimination에서
        # 제거 프롬프트를 만들 때 이 정보가 필요하다)
        self.scm_confirmed.append({
            'name': winner_name,
            'covariate': winner.get('covariate', ''),
            'parameter': winner.get('parameter', ''),
            'model_type': winner.get('model_type', ''),
            'delta_ofv': winner_delta,
            'code': winner_code,
            'ofv': winner_ofv,
            'iteration': winner_iter,
            'round': rnd,
        })

        # covariate_history 결과 확정
        for r in self.scm_round_results:
            if r['name'] == winner_name:
                self._finalize_covariate_result(r['name'], rnd, 'ACCEPTED')
            else:
                self._finalize_covariate_result(r['name'], rnd, 'TESTED')

        # 출력
        base_ofv_s = f"{self.scm_round_base_ofv:.2f}" if self.scm_round_base_ofv is not None else "N/A"
        print(f"\n{'='*70}")
        print(f"SCM ROUND {rnd} COMPLETE — WINNER SELECTED")
        print(f"{'='*70}")
        print(f"  Round base OFV : {base_ofv_s}")
        print(f"  Winner         : {winner_name}")
        print(f"  Winner ΔOFV    : {winner_delta:.2f}")
        print(f"  Winner OFV     : {winner_ofv:.2f}")
        print(f"  Winner iter    : {winner_iter}")
        if len(significant) > 1:
            print(f"\n  All significant candidates (vs round base):")
            for r in sorted(significant, key=lambda x: x['delta_ofv']):
                star = " ← WINNER" if r['name'] == winner_name else ""
                print(f"    {r['name']}: ΔOFV={r['delta_ofv']:.2f}{star}")
        print(f"\n  Confirmed covariates so far: {[c['name'] for c in self.scm_confirmed]}")
        print(f"{'='*70}")

        # 다음 라운드 준비
        self.scm_current_round += 1
        self.scm_round_tested = set()
        self.scm_round_results = []
        self.scm_round_base_ofv = None  # _init_scm_round_base의 중복 호출 가드 해제
        self._init_scm_round_base(winner_ofv, winner_code)
        self.current_code = winner_code
        self.best_code = winner_code
        self.best_ofv = winner_ofv
        self.best_iteration = winner_iter

        return True

    def _complete_backward_round(self) -> bool:
        """
        Backward elimination 라운드 완료 후 처리 (forward의 반대 방향 로직).

        scm_round_results에는 이번 라운드에서 "제거해봤을 때"의 결과가 기록되어 있다
        (mode='remove', delta_ofv = OFV_제거후 - OFV_전체모델 = 제거 비용, 양수일수록 나쁨).

        - 제거 비용이 scm_backward_threshold(기본 6.63, p<0.01) 미만이고 cov 성공인
          후보 중 가장 작은 비용을 보인 covariate를 영구 제거
        - 제거 대상이 있으면: scm_confirmed에서 제외, scm_eliminated에 기록,
          base를 제거 후 모델로 갱신, 다음 backward 라운드 준비 → True
        - 제거 대상이 없으면 (모든 후보가 threshold 이상, 즉 여전히 유의함)
          → backward elimination 완전히 종료 → False
        """
        rnd = self.scm_current_round
        threshold = self.scm_backward_threshold

        removable = [r for r in self.scm_round_results
                     if r.get('delta_ofv') is not None
                     and r['delta_ofv'] < threshold
                     and r.get('cov_ok', False)]

        if not removable:
            base_ofv_s = f"{self.scm_round_base_ofv:.2f}" if self.scm_round_base_ofv is not None else "N/A"
            print(f"\n{'='*70}")
            print(f"BACKWARD ROUND {rnd} COMPLETE — NO ELIMINATION")
            print(f"{'='*70}")
            print(f"  All remaining covariates significant at p<0.01 (ΔOFV on removal >= {threshold})")
            print(f"  Full model OFV: {base_ofv_s}")
            print(f"  Tested for removal: {[r['name'] for r in self.scm_round_results]}")
            print(f"{'='*70}")
            if self.scm_round_base_code:
                self.current_code = self.scm_round_base_code
                self.best_code = self.scm_round_base_code
            for r in self.scm_round_results:
                self._finalize_covariate_result(r['name'], rnd, 'RETAINED')
            return False

        # 제거 대상: ΔOFV(제거 비용) 가장 작은 것 (제거해도 가장 덜 나빠지는 covariate)
        loser = min(removable, key=lambda r: r['delta_ofv'])
        loser_name = loser['name']
        loser_delta = loser['delta_ofv']
        loser_code = loser.get('code')
        loser_ofv = loser.get('ofv')
        loser_iter = loser.get('iteration')

        if not loser_code:
            print(f"\n[Backward Round {rnd} WARNING] Saved code not found for '{loser_name}' — keeping full model")
            if self.scm_round_base_code:
                self.current_code = self.scm_round_base_code
                self.best_code = self.scm_round_base_code
            for r in self.scm_round_results:
                self._finalize_covariate_result(r['name'], rnd, 'RETAINED')
            return False

        # 제거 확정
        self.scm_confirmed = [c for c in self.scm_confirmed if c['name'] != loser_name]
        self.scm_eliminated.append({
            'name': loser_name,
            'delta_ofv': loser_delta,
            'iteration': loser_iter,
            'round': rnd,
        })

        for r in self.scm_round_results:
            if r['name'] == loser_name:
                self._finalize_covariate_result(r['name'], rnd, 'ELIMINATED')
            else:
                self._finalize_covariate_result(r['name'], rnd, 'RETAINED')

        base_ofv_s = f"{self.scm_round_base_ofv:.2f}" if self.scm_round_base_ofv is not None else "N/A"
        print(f"\n{'='*70}")
        print(f"BACKWARD ROUND {rnd} COMPLETE — COVARIATE ELIMINATED")
        print(f"{'='*70}")
        print(f"  Full model OFV   : {base_ofv_s}")
        print(f"  Eliminated       : {loser_name}")
        print(f"  ΔOFV on removal  : +{loser_delta:.2f} (< {threshold} threshold — not significant at p<0.01)")
        print(f"  Model OFV after  : {loser_ofv:.2f}")
        if len(removable) > 1:
            print(f"\n  All removable candidates (ΔOFV on removal):")
            for r in sorted(removable, key=lambda x: x['delta_ofv']):
                star = " ← ELIMINATED" if r['name'] == loser_name else ""
                print(f"    {r['name']}: ΔOFV=+{r['delta_ofv']:.2f}{star}")
        print(f"\n  Remaining covariates: {[c['name'] for c in self.scm_confirmed]}")
        print(f"{'='*70}")

        # 다음 backward 라운드 준비
        self.scm_current_round += 1
        self.scm_round_tested = set()
        self.scm_round_results = []
        self.scm_round_base_ofv = None
        self._init_scm_round_base(loser_ofv, loser_code)
        self.current_code = loser_code
        self.best_code = loser_code
        self.best_ofv = loser_ofv
        self.best_iteration = loser_iter

        if not self.scm_confirmed:
            # 모든 covariate가 제거됨 — 더 이상 테스트할 후보가 없으므로 종료
            return False

        return True

    def _finalize_covariate_result(self, name: str, rnd: int, result: str) -> None:
        """covariate_history에서 해당 라운드의 result를 최종 확정"""
        for h in self.covariate_history:
            if h['name'] == name and h.get('round') == rnd:
                h['result'] = result
                break

    def _update_parameter_history(self, parsed_data: Dict) -> None:
        """Store THETA/OMEGA/SIGMA estimates for stabilization across iterations.

        This function does NOT change any decision logic. It only records the
        latest parameter estimates so that we can suggest narrower initial
        values/bounds for the next NONMEM run.
        """
        try:
            params = parsed_data.get('parameter_estimates', {}) or {}
        except Exception:
            return

        theta_list = params.get('theta', []) or []
        omega_list = params.get('omega', []) or []
        sigma_list = params.get('sigma', []) or []

        def _extract_vals(items, label):
            # NONMEMParser stores the parsed value under the key "value" (not
            # "estimate" — see nonmem_parser.py's theta/omega/sigma.append({...})).
            # THETA items have an "index" key; OMEGA/SIGMA items have "row"/"col"
            # instead — neither has a "name" key, so synthesize a readable one.
            names = []
            vals = []
            for it in items:
                est = it.get('value')
                try:
                    if est is None:
                        continue
                    v = float(est)
                except Exception:
                    continue
                if 'index' in it:
                    name = f"{label}({it['index']})"
                elif 'row' in it and 'col' in it:
                    name = f"{label}({it['row']},{it['col']})"
                else:
                    name = label
                names.append(name)
                vals.append(v)
            return names, vals

        th_names, th_vals = _extract_vals(theta_list, 'THETA')
        om_names, om_vals = _extract_vals(omega_list, 'OMEGA')
        sg_names, sg_vals = _extract_vals(sigma_list, 'SIGMA')

        if not (th_vals or om_vals or sg_vals):
            return

        entry = {
            'iteration': self.iteration,
            'theta_names': th_names,
            'theta_vals': th_vals,
            'omega_names': om_names,
            'omega_vals': om_vals,
            'sigma_names': sg_names,
            'sigma_vals': sg_vals,
        }
        self.parameter_history.append(entry)

    def _build_parameter_stabilization_guidance(self) -> str:
        """Build textual guidance to narrow THETA/OMEGA/SIGMA around previous estimates.

        This produces human-readable instructions that are appended to the
        NONMEM output given to the LLM so that the next control stream uses
        tighter initial values and boundaries, while keeping all other logic
        unchanged.
        """
        if not getattr(self, 'parameter_history', None):
            return ""

        last = self.parameter_history[-1]
        lines: list[str] = []
        lines.append("Use the following parameter estimates to tighten initial values and bounds")
        lines.append("for the NEXT model iteration (do NOT change structural model unless required).")
        lines.append("For each parameter, set the initial value close to the estimate and")
        lines.append("shrink the bounds to roughly 50-150% of the estimate (or positive-only for variances).")
        lines.append("")

        def _theta_bounds(v: float) -> tuple[float, float]:
            # Symmetric shrinkage around current estimate; keep wide enough to avoid trapping
            if v == 0.0:
                return -1.0, 1.0
            lower = v * 0.5
            upper = v * 1.5
            # Avoid collapsing very small parameters
            if 0 < abs(v) < 1e-3:
                lower = v * 0.1
                upper = v * 10.0
            return lower, upper

        def _var_bounds(v: float) -> tuple[float, float]:
            # Variances (OMEGA/SIGMA) must stay positive but can vary on log-scale
            base = max(v, 1e-6)
            lower = base * 0.3
            upper = base * 3.0
            return lower, upper

        th_names = last.get('theta_names') or []
        th_vals = last.get('theta_vals') or []
        if th_vals:
            lines.append("THETA (fixed effects):")
            for i, v in enumerate(th_vals, start=1):
                name = th_names[i-1] if i-1 < len(th_names) else f"THETA({i})"
                lo, hi = _theta_bounds(v)
                lines.append(f"  - {name}: estimate={v:.6g}, recommended bounds ≈ [{lo:.6g}, {hi:.6g}]")
            lines.append("")

        om_names = last.get('omega_names') or []
        om_vals = last.get('omega_vals') or []
        if om_vals:
            lines.append("OMEGA (IIV variances):")
            for i, v in enumerate(om_vals, start=1):
                name = om_names[i-1] if i-1 < len(om_names) else f"OMEGA({i})"
                lo, hi = _var_bounds(v)
                lines.append(f"  - {name}: estimate={v:.6g}, recommended bounds ≈ [{lo:.6g}, {hi:.6g}] (keep >0)")
            lines.append("")

        sg_names = last.get('sigma_names') or []
        sg_vals = last.get('sigma_vals') or []
        if sg_vals:
            lines.append("SIGMA (residual error variances):")
            for i, v in enumerate(sg_vals, start=1):
                name = sg_names[i-1] if i-1 < len(sg_names) else f"SIGMA({i})"
                lo, hi = _var_bounds(v)
                lines.append(f"  - {name}: estimate={v:.6g}, recommended bounds ≈ [{lo:.6g}, {hi:.6g}] (keep >0)")
            lines.append("")

        return "\n".join(lines)

    def _track_failed_strategy(self, code: str, error_type: str) -> int:
        """
        Track strategies that failed to prevent repeating them

        Args:
            code: The code that failed
            error_type: Type of error (e.g., "V3_ERROR", "MINIMIZATION_FAILED", "SYNTAX_ERROR")

        Returns:
            How many times this exact strategy signature has now failed (>=1).
            Previously this function only recorded membership (seen/not-seen) and
            never counted repeats, so nothing could act on "this has failed N times".
        """
        # Extract key characteristics
        advan_match = re.search(r'ADVAN(\d+)', code, re.IGNORECASE)
        advan = advan_match.group(1) if advan_match else "unknown"

        omega_count = 0
        omega_pattern = r'\$OMEGA\s*(.*?)(?=\n\s*\$|\Z)'
        omega_match = re.search(omega_pattern, code, re.DOTALL | re.IGNORECASE)
        if omega_match:
            omega_section = omega_match.group(1)
            for line in omega_section.split('\n'):
                if ';' in line:
                    line = line.split(';')[0]
                line = line.strip()
                if line:
                    numbers = re.findall(r'\d+\.?\d*(?:[eE][+-]?\d+)?', line)
                    omega_count += len(numbers)

        strategy_signature = f"ADVAN{advan}_OMEGA{omega_count}_{error_type}"

        self.strategy_repeat_count[strategy_signature] = self.strategy_repeat_count.get(strategy_signature, 0) + 1

        if strategy_signature not in self.failed_strategies:
            self.failed_strategies.append(strategy_signature)
            print(f"  [TRACK] Recording failed strategy: {strategy_signature}")

        return self.strategy_repeat_count[strategy_signature]

    # ADVAN -> compartment count. Fixed NONMEM facts, not drug/dataset-specific.
    _ADVAN_COMPARTMENTS = {1: 1, 2: 1, 3: 2, 4: 2, 11: 3, 12: 3}

    # (ADVAN, TRANS) -> required $PK parameter names. Fixed NONMEM facts.
    # Scoped to the special ADVANs (1,2,3,4,11,12) actually used by this project;
    # ADVAN5-10/13 are general/$DES-based and have no fixed parameter list, so
    # they're intentionally left out (checks below skip unknown ADVANs).
    _ADVAN_TRANS_PARAMS = {
        (1, 1):  ['K', 'V'],
        (1, 2):  ['CL', 'V'],
        (2, 1):  ['K', 'V', 'KA'],
        (2, 2):  ['CL', 'V', 'KA'],
        (3, 1):  ['K', 'K12', 'K21'],
        (3, 4):  ['CL', 'V1', 'Q', 'V2'],
        (4, 1):  ['K', 'K23', 'K32', 'KA'],
        (4, 4):  ['CL', 'V2', 'Q', 'V3', 'KA'],
        (11, 1): ['K', 'K12', 'K21', 'K13', 'K31'],
        (11, 4): ['CL', 'V1', 'Q2', 'V2', 'Q3', 'V3'],
        (12, 1): ['K', 'K23', 'K32', 'K24', 'K42', 'KA'],
        (12, 4): ['CL', 'V2', 'Q3', 'V3', 'Q4', 'V4', 'KA'],
    }

    def _check_compartment_invariance(self, code: str, prev_ofv: Optional[float]) -> Optional[str]:
        """
        Guard against a silent compartment-count change during ERROR RECOVERY
        (i.e. the previous iteration never reached estimation — prev_ofv is None,
        meaning it was an NM-TRAN compile/syntax error, not a real diagnostic run).

        Genuine Phase 2 structural decisions (based on residual pattern analysis)
        always have a real OFV from a completed run, so this check does not fire
        for those — only for "fixing a syntax error somehow changed the structure"
        cases, which is what actually happened in the tobramycin ADVAN3->ADVAN2
        cascade.

        Returns a corrective instruction string if violated, else None.
        """
        if prev_ofv is not None:
            return None  # last run had a real OFV -> not error recovery, don't gate

        hint_compartments = self.data_loader.get_metadata().get('compartments')
        if hint_compartments is None:
            return None

        advan_match = re.search(r'ADVAN(\d+)', code, re.IGNORECASE)
        if not advan_match:
            return None

        new_compartments = self._ADVAN_COMPARTMENTS.get(int(advan_match.group(1)))
        if new_compartments is None or new_compartments == hint_compartments:
            return None

        return (
            f"REJECTED: This code changes the structure to {new_compartments}-compartment "
            f"(ADVAN{advan_match.group(1)}), but the dataset's data-driven compartment "
            f"count is {hint_compartments} (PKGPT_STRUCT_HINT). The previous error was a "
            f"syntax/compile problem, NOT evidence that the compartment count is wrong. "
            f"Keep the model at {hint_compartments}-compartment and fix ONLY the specific "
            f"syntax/compile error from the previous iteration."
        )

    def _check_advan_trans_validity(self, code: str) -> Optional[str]:
        """
        Guard against invalid ADVAN/TRANS combinations and missing required $PK
        parameters for that combination. These are fixed NONMEM syntax facts
        (not drug/dataset-specific judgment calls), so this is safe to hardcode.

        Returns a corrective instruction string if violated, else None (including
        when the ADVAN isn't one of the special ADVANs tracked here — e.g. general
        ADVAN5-10/13 with $DES, which have no fixed parameter list).
        """
        m = re.search(r'\$SUBROUTINES?\s+ADVAN(\d+)\s+TRANS(\d+)', code, re.IGNORECASE)
        if not m:
            return None

        advan_num, trans_num = int(m.group(1)), int(m.group(2))
        key = (advan_num, trans_num)
        required = self._ADVAN_TRANS_PARAMS.get(key)

        if required is None:
            known_trans_for_advan = sorted(t for (a, t) in self._ADVAN_TRANS_PARAMS if a == advan_num)
            if known_trans_for_advan and trans_num not in known_trans_for_advan:
                valid = ', '.join(f"TRANS{t}" for t in known_trans_for_advan)
                return (
                    f"REJECTED: ADVAN{advan_num} TRANS{trans_num} is not a valid combination. "
                    f"ADVAN{advan_num} only supports: {valid}. Pick one of these and use its "
                    f"required $PK parameters."
                )
            return None  # unknown ADVAN entirely (general ADVAN5-10/13) -> skip

        pk_match = re.search(r'\$PK(.*?)(?=\n\s*\$|\Z)', code, re.DOTALL | re.IGNORECASE)
        pk_block = pk_match.group(1) if pk_match else ""
        missing = [p for p in required if not re.search(rf'\b{re.escape(p)}\s*=', pk_block, re.IGNORECASE)]

        if missing:
            return (
                f"REJECTED: ADVAN{advan_num} TRANS{trans_num} requires these parameters to be "
                f"assigned in $PK: {', '.join(required)}. Missing: {', '.join(missing)}."
            )
        return None

    def _track_combined_error_failure(self, code: str) -> None:
        """
        현재 error model이 boundary/covariance 문제를 일으킨 경우 기록.
        어떤 error model 구조가 실패했는지 파악해서 failed_strategies에 추가.
        특정 model을 금지하는 것이 아니라, 현재 구조가 실패했음을 LLM에 알려
        다른 error model 구조를 시도하도록 유도.
        """
        # 현재 error model 구조 파악
        eps_count = len(re.findall(r'EPS\s*\(\s*\d+\s*\)', code, re.IGNORECASE))
        sigma_diag = re.search(r'\$SIGMA\s+DIAGONAL\s*\(\s*(\d+)', code, re.IGNORECASE)
        sigma_count = int(sigma_diag.group(1)) if sigma_diag else 1
        sqrt_err = bool(re.search(r'W\s*=\s*SQRT\s*\(', code, re.IGNORECASE))

        # error model 유형 결정
        if eps_count >= 2 or sigma_count >= 2 or sqrt_err:
            error_model_type = "COMBINED_PROPORTIONAL_ADDITIVE"
        else:
            error_model_type = "PROPORTIONAL_ONLY"

        sig = f"ERROR_MODEL_{error_model_type}_BOUNDARY_FAILED"

        if sig not in self.failed_strategies:
            self.failed_strategies.append(sig)
            print(f"  [TRACK] Recording failed strategy: {sig}")
            print(f"  [TRACK] Current error model ({error_model_type}) caused boundary/covariance failure")

    def _build_ai_guidance_block(self) -> str:
        """
        직전 iteration AI quality evaluation의 critical_issues + recommendations를
        다음 iteration LLM 프롬프트 최상단에 prepend하기 위한 섹션 생성.

        '콘솔 출력만 하고 버려지던' AI 권고를 LLM이 실제로 읽도록 보장.
        내용이 없으면 빈 문자열 반환 (prepend 건너뜀).
        """
        issues = getattr(self, 'last_ai_critical_issues', [])
        recs   = getattr(self, 'last_ai_recommendations', [])

        if not issues and not recs:
            return ""

        sep = "═" * 54
        lines = [
            sep,
            "AI QUALITY EVALUATOR — MANDATORY ACTIONS FOR THIS ITERATION",
            sep,
        ]

        if issues:
            lines.append("\nCRITICAL ISSUES IDENTIFIED:")
            for issue in issues:
                lines.append(f"  ✗ {issue}")

        if recs:
            lines.append("\nREQUIRED ACTIONS (implement in priority order):")
            for i, rec in enumerate(recs, 1):
                lines.append(f"  {i}. {rec}")

        lines += [
            "",
            "⚠️  Implement ALL required actions above in the next control stream.",
            "    These are based on diagnostic analysis of the current run.",
            sep,
        ]
        return "\n".join(lines)

    def _generate_improved_code(self, parser: Optional[NONMEMParser]):
        """Generate improved NONMEM code based on results and current phase"""
        print(f"\nGenerating improved model for iteration {self.iteration + 1}...")
        print(f"Phase-specific guidance: {self.current_phase}")

        # ── Phase 5: covariate instruction 보장 ─────────────────────────────
        # current_covariate_instruction은 _record_phase5_covariate_result() 완료 후
        # None으로 초기화된다. AI가 CONTINUE를 반환하는 경우 Check 5.5 블록을
        # 거치지 않아 instruction이 None 상태로 유지되고, 다음 iteration에서
        # _record_phase5_covariate_result()가 early return → covariate가 영원히
        # tested로 기록되지 않는 무한루프가 발생한다.
        # → 코드 생성 직전에 항상 다음 테스트할 covariate를 확인해 instruction 보장.
        if self.current_phase == ModelPhase.COVARIATE_ANALYSIS:
            if self.current_covariate_instruction is None:
                next_cand = self._get_next_covariate_to_test()
                if next_cand:
                    self.current_covariate_instruction = next_cand
                    print(f"  [SCM] Covariate instruction set: '{next_cand['name']}'")

        # Get output text
        prev_ofv_for_guard = None
        if parser:
            nonmem_output = parser.get_full_output()
            issues = parser.get_issues()

            # Track failed strategy if minimization failed
            parsed_data = parser.get_parsed_data()
            prev_ofv_for_guard = parsed_data.get('objective_function')
            if not parsed_data.get('minimization_successful', False):
                # A None OFV means NM-TRAN never reached estimation (compile/syntax
                # error) — distinct from a real minimization that ran and failed.
                error_type = "SYNTAX_ERROR" if prev_ofv_for_guard is None else "MINIMIZATION_FAILED"
                repeat_count = self._track_failed_strategy(self.current_code, error_type)
                if repeat_count >= 2 and self.best_code and self.current_phase != ModelPhase.COVARIATE_ANALYSIS:
                    print(f"  [RESET] Same failure signature repeated {repeat_count}x "
                          f"({error_type}) — abandoning this approach, restoring best "
                          f"known model (iter {self.best_iteration}, OFV={self.best_ofv})")
                    self.current_code = self.best_code
        else:
            output_file = f"{self.output_base}_iter{self.iteration}.lst"
            try:
                with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
                    nonmem_output = f.read()

                # Check if this is a V3 error (ADVAN issue)
                if 'V3' in nonmem_output and 'ERROR' in nonmem_output.upper():
                    self._track_failed_strategy(self.current_code, "V3_ERROR")
            except:
                nonmem_output = "Could not read output file"
            issues = ["Output file could not be parsed properly"]

        # Gemini has 1M token context - can handle full NONMEM output
        # Only truncate if extremely large (>500KB = ~250K tokens)
        if len(nonmem_output) > 500000:
            output_len = len(nonmem_output)
            print(f"  [INFO] Large output detected")
            print(f"  [INFO] Output size: {output_len} chars, using smart truncation")
            nonmem_output = self._smart_truncate_output(nonmem_output, max_length=200000)

        # Check if mandatory simplification is needed
        simplification_required, simplification_reason = self._detect_simplification_needed()

        if simplification_required:
            print(f"\n{'!'*70}")
            print("MANDATORY SIMPLIFICATION REQUIRED")
            print(f"{'!'*70}")
            print(f"{simplification_reason}")
            print(f"{'!'*70}\n")

        # Count current OMEGA parameters to prevent destruction
        current_omega_count = 0
        # Match $OMEGA block (handles both "$OMEGA\n0.1" and "$OMEGA 0.1" formats)
        omega_pattern = r'\$OMEGA\s*(.*?)(?=\n\s*\$|\Z)'
        omega_match = re.search(omega_pattern, self.current_code, re.DOTALL | re.IGNORECASE)
        if omega_match:
            omega_section = omega_match.group(1)
            # Count numeric values (each OMEGA parameter)
            # Split by lines and look for numeric values
            for line in omega_section.split('\n'):
                # Remove comments
                if ';' in line:
                    line = line.split(';')[0]
                line = line.strip()
                # Skip empty lines
                if not line:
                    continue
                # Count numbers in the line (each number = 1 OMEGA)
                # Match floats like 0.1, 1.0, 0.001, etc.
                numbers = re.findall(r'\d+\.?\d*(?:[eE][+-]?\d+)?', line)
                current_omega_count += len(numbers)

        # Get phase-specific guidance
        phase_guidance = self._get_phase_specific_guidance()

        # Generate failed strategies warning
        failed_strategies_warning = ""
        if self.failed_strategies:
            failed_strategies_warning = "\n\n" + "="*70 + "\n"
            failed_strategies_warning += "FAILED STRATEGIES - DO NOT REPEAT THESE\n"
            failed_strategies_warning += "="*70 + "\n"
            failed_strategies_warning += "The following model configurations have already been tried and FAILED:\n"
            for strategy in self.failed_strategies[-5:]:  # Show last 5 failed attempts
                failed_strategies_warning += f"  ✗ {strategy}\n"
            failed_strategies_warning += "\nIMPORTANT: Do NOT generate code matching these patterns.\n"
            failed_strategies_warning += "If you see ADVAN4_OMEGA3_V3_ERROR, do NOT use ADVAN4 again.\n"
            # error model 실패 전용 안내 (데이터셋 특성에 따라 다른 model 제안)
            if any('ERROR_MODEL' in s for s in self.failed_strategies):
                failed_model = next((s for s in self.failed_strategies if 'ERROR_MODEL' in s), '')
                failed_strategies_warning += f"\n⚠️  RESIDUAL ERROR MODEL FAILURE DETECTED:\n"
                failed_strategies_warning += f"   The current error model structure ({failed_model}) caused\n"
                failed_strategies_warning += f"   repeated boundary/covariance failures in this dataset.\n"
                failed_strategies_warning += f"   Try a DIFFERENT error model structure:\n"
                if 'COMBINED' in failed_model:
                    failed_strategies_warning += f"   → Switch to proportional-only: W = THETA(n)*IPRED, $SIGMA 1 FIX\n"
                else:
                    failed_strategies_warning += f"   → Consider combined error: W = SQRT(THETA(a)**2 + (THETA(b)*IPRED)**2)\n"
            failed_strategies_warning += "="*70 + "\n\n"

        # Get last 2 models from code history for context
        previous_models = []
        if len(self.code_history) >= 2:
            previous_models = self.code_history[-2:]
        elif len(self.code_history) == 1:
            previous_models = self.code_history[-1:]

        # Prepare parsed_results for v2 API
        if parser:
            parsed_results = parser.get_parsed_data()
        else:
            parsed_results = {
                'objective_function': None,
                'minimization_successful': False,
                'warnings': [],
                'eta_shrinkage': []
            }

        # Get metadata
        metadata = self.data_loader.get_metadata()

        # Generate improvement prompt with phase-aware routing (V2)
        print(f"  [INFO] Using phase-specific prompt: {self.current_phase}")

        # Build parameter stabilization guidance and append to NONMEM output.
        # This guides the LLM to tighten THETA/OMEGA/SIGMA bounds around the
        # latest estimates without changing the rest of the optimization logic.
        param_guidance = self._build_parameter_stabilization_guidance()
        if param_guidance:
            augmented_output = (
                nonmem_output
                + "\n\n" + "="*70 + "\n"
                + "AUTO-GENERATED PARAMETER STABILIZATION GUIDANCE (FOR THETA/OMEGA/SIGMA)\n"
                + "="*70 + "\n"
                + param_guidance
            )
        else:
            augmented_output = nonmem_output

        # Phase 5 (Covariate Analysis): phase_guidance(=Phase5Covariates 단일 covariate 지시)를 직접 사용.
        # improvement_prompt_v2는 Phase 5에서 AI 권고/구조 변경 지시를 오염시키므로 건너뜀.
        # phase_guidance는 이미 current_code와 단일 covariate 지시를 모두 포함함.
        in_phase5 = (self.current_phase == ModelPhase.COVARIATE_ANALYSIS)

        if in_phase5:
            prompt = phase_guidance
            if augmented_output:
                prompt += (
                    f"\n\n{'='*70}\n"
                    "NONMEM OUTPUT (참고용 — 구조 변경 금지, covariate 추가만 허용)\n"
                    f"{'='*70}\n"
                    + augmented_output
                )
        else:
            # Phase 1-4: improvement_prompt_v2 사용 + AI 권고 포함
            try:
                prompt = PromptTemplates.improvement_prompt_v2(
                    iteration=self.iteration,
                    current_code=self.current_code,
                    nonmem_output=augmented_output,
                    parsed_results=parsed_results,
                    current_phase=self.current_phase,
                    metadata=metadata,
                    previous_improvements=self.improvement_history,
                    issues_found=issues,
                    ai_recommendations=getattr(self, 'last_ai_recommendations', []),
                    ai_critical_issues=getattr(self, 'last_ai_critical_issues', []),
                    plausibility_bounds=getattr(self, 'plausibility_bounds', None)
                )
            except TypeError:
                # 구버전 prompt_templates.py는 ai_* 파라미터 미지원 → 없이 호출
                prompt = PromptTemplates.improvement_prompt_v2(
                    iteration=self.iteration,
                    current_code=self.current_code,
                    nonmem_output=augmented_output,
                    parsed_results=parsed_results,
                    current_phase=self.current_phase,
                    metadata=metadata,
                    previous_improvements=self.improvement_history,
                    issues_found=issues,
                )

            # Phase 1-4만: failed strategies warning 및 AI 권고 prepend
            if failed_strategies_warning:
                prompt = failed_strategies_warning + "\n\n" + prompt

            ai_guidance_block = self._build_ai_guidance_block()
            if ai_guidance_block:
                prompt = ai_guidance_block + "\n\n" + prompt

        # Revert notice: revert가 발생한 직후 iteration에서 LLM에게 명시적으로 알림
        # LLM은 현재 코드가 revert된 best model임을 모르기 때문에
        # 동일한 실패 전략을 반복하는 것을 방지
        if getattr(self, 'last_revert_info', None):
            info = self.last_revert_info
            failed_ofv_str = f"{info['failed_ofv']:.2f}" if info['failed_ofv'] is not None else "N/A"
            failed_shrink_str = f"{info['failed_shrinkage']:.1f}%" if info['failed_shrinkage'] is not None else "N/A"

            if info.get('in_phase5'):
                # Phase 5 전용 revert notice: 구조/error model 변경 절대 금지
                rejected_cov = info.get('rejected_covariate', 'the previous covariate')
                next_cand = self._get_next_covariate_to_test()
                next_name = next_cand['name'] if next_cand else 'None (SCM complete)'
                revert_notice = (
                    f"\n{'!'*70}\n"
                    f"⚠️  SCM COVARIATE REJECTED — BASE MODEL STRUCTURE IS FROZEN\n"
                    f"{'!'*70}\n"
                    f"The covariate '{rejected_cov}' did not improve the model sufficiently\n"
                    f"and has been REJECTED (ΔOFV threshold not met or covariance failed).\n"
                    f"\n"
                    f"The model has been reverted to the BASE MODEL (Iter {info['reverted_to_iteration']}).\n"
                    f"\n"
                    f"ABSOLUTE RULES FOR PHASE 5 — THESE CANNOT BE OVERRIDDEN:\n"
                    f"  ❌ DO NOT change the error model ($ERROR, $SIGMA)\n"
                    f"  ❌ DO NOT change the IIV structure ($OMEGA)\n"
                    f"  ❌ DO NOT change the structural model (ADVAN, compartments)\n"
                    f"  ❌ DO NOT change $ESTIMATION method — FOCE-I (METHOD=1 INTER) only\n"
                    f"  ❌ DO NOT remove or modify existing covariate relationships in the base code\n"
                    f"  ❌ DO NOT retry '{rejected_cov}' — it has already been tested\n"
                    f"\n"
                    f"ONLY ALLOWED ACTION:\n"
                    f"  ✅ Add the NEXT untested covariate: '{next_name}'\n"
                    f"  ✅ Keep the base model code exactly as-is, add ONE new covariate THETA\n"
                    f"{'!'*70}\n"
                )
            else:
                # Phase 1-4 기존 revert notice
                revert_notice = (
                    f"\n{'!'*70}\n"
                    f"⚠️  REVERT NOTICE — READ THIS BEFORE MAKING ANY CHANGES\n"
                    f"{'!'*70}\n"
                    f"The previous iteration (Iter {info['reverted_from_iteration']}) FAILED and was REVERTED.\n"
                    f"The current code you are receiving is the BEST model (Iter {info['reverted_to_iteration']}).\n"
                    f"The NONMEM output below is from the FAILED iteration — NOT from the current code.\n"
                    f"\n"
                    f"WHY IT FAILED:\n"
                    f"  - Composite score: {info['failed_composite']:.1f} (best: {self.best_composite_score:.1f})\n"
                    f"  - OFV: {failed_ofv_str}\n"
                    f"  - Shrinkage: {failed_shrink_str}\n"
                    f"\n"
                    f"MANDATORY RULES FOR NEXT ITERATION:\n"
                    f"  1. DO NOT re-apply the same changes that caused Iter {info['reverted_from_iteration']} to fail.\n"
                    f"  2. DO NOT add IIV/OMEGA terms if high shrinkage caused the revert.\n"
                    f"  3. The current code is already stable — try a DIFFERENT direction:\n"
                    f"     - If in Phase 4: accept current IIV structure, move toward Phase 5\n"
                    f"     - If in Phase 5: add the ONE covariate specified in the SCM STATUS above\n"
                    f"     - If structural: do NOT change ADVAN or compartment count\n"
                    f"{'!'*70}\n"
                )

            prompt = revert_notice + "\n\n" + prompt
            # 한 번 사용 후 초기화 (다음 iteration에는 전달 안 함)
            self.last_revert_info = None

        response = self.gemini_client.generate(prompt)

        # Extract improved code
        self.current_code = self._enforce_foce_estimation(
            self._enforce_input_line(self._extract_nonmem_code(response))
        )

        # Phase 1-4 structural safety net: after a compile/syntax error, check the
        # LLM's fix didn't silently change compartment count or produce an invalid
        # ADVAN/TRANS combination. Bounded to 1 retry to avoid infinite loops —
        # if still invalid after the retry, proceed anyway (NONMEM will reject it
        # and the failed-strategy counter/reset above will catch repeats).
        if self.current_phase != ModelPhase.COVARIATE_ANALYSIS:
            for _guard_attempt in range(2):
                guard_problem = (
                    self._check_compartment_invariance(self.current_code, prev_ofv_for_guard)
                    or self._check_advan_trans_validity(self.current_code)
                )
                if not guard_problem:
                    break
                print(f"  [GUARD] {guard_problem}")
                if _guard_attempt == 1:
                    break  # already retried once, proceed with whatever we have
                retry_prompt = prompt + f"\n\n{'!'*70}\n{guard_problem}\n{'!'*70}\n"
                retry_response = self.gemini_client.generate(retry_prompt)
                self.current_code = self._enforce_foce_estimation(
                    self._enforce_input_line(self._extract_nonmem_code(retry_response))
                )

        # Phase 5: SCM 전용 — ADVAN/OMEGA/ERROR 구조 변경 자동 복원
        # best_code를 명시적으로 넘겨야 함: self.current_code는 이미 새 코드로 업데이트된 상태
        if self.current_phase == ModelPhase.COVARIATE_ANALYSIS:
            self.current_code = self._enforce_phase5_structure(
                self.current_code, base_code=self.best_code
            )

        # Extract analysis and changes
        analysis = self._extract_section(response, 'ANALYSIS')
        changes = self._extract_section(response, 'CHANGES MADE')

        if analysis:
            print(f"\nAnalysis: {analysis}")
        if changes:
            print(f"Changes: {changes}")

        # Store code in history
        self.code_history.append({
            'iteration': self.iteration + 1,
            'code': self.current_code,
            'description': changes or 'Not specified'
        })

        # Update history with changes
        if self.improvement_history:
            self.improvement_history[-1]['changes'] = changes or 'Not specified'

        print(f"[OK] Improved code generated for next iteration")

    def _smart_truncate_output(self, output: str, max_length: int = 8000) -> str:
        """
        Intelligently truncate NONMEM output to keep most important parts

        Priority:
        1. Error messages (first 2000 chars)
        2. Final parameter estimates (search for "FINAL")
        3. Objective function value
        4. Middle section if space remains
        """
        if len(output) <= max_length:
            return output

        # Always keep the beginning (errors usually here)
        head_size = min(2000, max_length // 2)
        head = output[:head_size]

        # Try to find and keep final parameter estimates
        final_match = re.search(r'FINAL PARAMETER ESTIMATE.*?(?=\n\s*\n|\Z)', output, re.DOTALL | re.IGNORECASE)

        if final_match:
            final_section = final_match.group(0)
            remaining = max_length - head_size - len(final_section) - 200

            if remaining > 0:
                # Include some middle context
                middle_start = head_size
                middle_end = middle_start + remaining
                middle = output[middle_start:middle_end]

                return f"{head}\n\n... [middle section truncated] ...\n\n{middle}\n\n{final_section}"
            else:
                return f"{head}\n\n... [truncated] ...\n\n{final_section}"
        else:
            # No final estimates found, keep head and tail
            tail_size = min(1000, max_length - head_size)
            tail = output[-tail_size:]
            return f"{head}\n\n... [middle section truncated ({len(output) - head_size - tail_size} chars)] ...\n\n{tail}"

    def _extract_section(self, text: str, section_name: str) -> Optional[str]:
        """Extract a section from response text"""
        pattern = rf'{section_name}:\s*(.+?)(?:\n\n|\n[A-Z]{{2,}}:|\Z)'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _generate_final_summary(self) -> Dict:
        """Generate final optimization summary"""
        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE")
        print("=" * 70)

        # ── Phase 5 진입 iteration 찾기 ──────────────────────────────────────
        phase5_start_iter = None
        for t in self.phase_history:
            if hasattr(t.get('to_phase'), 'value'):
                is_p5 = (t['to_phase'] == ModelPhase.COVARIATE_ANALYSIS)
            else:
                is_p5 = ('Covariate' in str(t.get('to_phase', '')))
            if is_p5:
                phase5_start_iter = t['iteration']
                break
        # covariate_history로 보완
        if phase5_start_iter is None and self.covariate_history:
            phase5_start_iter = min(h['iteration'] for h in self.covariate_history)

        # improvement_history를 Phase 1-4 / Phase 5로 분리
        # phase5_start_iter = Phase 5 전환이 기록된 iteration (해당 NONMEM은 Phase 4)
        # → base_history는 <= 포함 (전환 트리거 iteration도 Phase 4 실행)
        # → p5_history는 그 다음 iteration부터 (실제 첫 SCM 실행)
        base_history = [e for e in self.improvement_history
                        if phase5_start_iter is None or e['iteration'] <= phase5_start_iter]
        p5_history   = [e for e in self.improvement_history
                        if phase5_start_iter is not None and e['iteration'] > phase5_start_iter]

        # Phase 1-4에서의 best iteration
        base_best_iter = None
        if base_history:
            best_e = min(base_history,
                         key=lambda e: e.get('composite_score', float('inf')))
            base_best_iter = best_e.get('iteration')

        hist_by_iter = {e['iteration']: e for e in self.improvement_history}

        # ── Phase Progression ────────────────────────────────────────────────
        if self.phase_history:
            print(f"\n{'='*70}")
            print("PHASE PROGRESSION")
            print(f"{'='*70}")
            for t in self.phase_history:
                print(f"  Iter {t['iteration']:>3}: {t['from_phase']} -> {t['to_phase']} "
                      f"({t['iterations_in_previous_phase']} iters in prev phase)")
            # current_phase는 NONMEM 실패 후 ESTABLISH_BASE로 임시 리셋될 수 있음
            # → phase_history 마지막 전환 목적지로 실제 최종 단계 표시
            effective_final = (
                self.phase_history[-1]['to_phase']
                if self.phase_history else self.current_phase
            )
            print(f"  Final phase: {effective_final}")
            print(f"{'='*70}")

        # ── BASE MODEL DEVELOPMENT (Phase 1-4) ───────────────────────────────
        print(f"\n{'='*70}")
        print("BASE MODEL DEVELOPMENT  (Phase 1-4)")
        print(f"{'='*70}")
        print(f"{'Iter':<6} {'Status':<8} {'OFV':<12} {'Composite':<12} {'Shrink':<10} {'CoV':<5}")
        print(f"{'-'*70}")

        for entry in (base_history if base_history else self.improvement_history):
            status_s  = "OK" if entry.get('minimization_successful') else "FAIL"
            ofv_val   = entry.get('ofv')
            ofv_s     = f"{ofv_val:.2f}" if ofv_val is not None else "N/A"
            comp_val  = entry.get('composite_score', float('inf'))
            comp_s    = f"{comp_val:.1f}" if comp_val != float('inf') else "N/A"
            shrink    = entry.get('avg_eta_shrinkage')
            shrink_s  = f"{shrink:.1f}%" if shrink is not None else "N/A"
            cov_s     = "Yes" if entry.get('covariance_successful') else "No"
            it        = entry.get('iteration')
            star      = " *" if it == base_best_iter else ""
            print(f"{it:<6} {status_s:<8} {ofv_s:<12} {comp_s:<12} {shrink_s:<10} {cov_s:<5}{star}")

        print(f"{'-'*70}")
        if base_best_iter:
            be = hist_by_iter.get(base_best_iter, {})
            print(f"* Best base model  Iter {base_best_iter} | "
                  f"OFV={be.get('ofv', 'N/A'):.2f}  "
                  f"Composite={be.get('composite_score', float('inf')):.1f}  "
                  f"Shrinkage={be.get('avg_eta_shrinkage', 0) or 0:.1f}%")
        print(f"{'='*70}")

        # ── SCM FORWARD SELECTION RESULTS (Phase 5) ──────────────────────────
        if self.covariate_history:
            confirmed_names = {c['name'] for c in self.scm_confirmed}

            print(f"\n{'='*70}")
            print("SCM FORWARD SELECTION + BACKWARD ELIMINATION RESULTS  (Phase 5)")
            print(f"{'='*70}")
            print(f"{'Rnd':<5} {'Iter':<6} {'Covariate':<18} {'Result':<10} {'ΔOFV':<9} {'OFV':<10} "
                  f"{'Shrink':<10} {'MaxRSE':<10} {'CoV':<5}")
            print(f"{'-'*82}")

            for h in self.covariate_history:
                name   = h['name']
                result = h['result']
                rnd    = h.get('round', '?')
                d_ofv  = h.get('delta_ofv')
                it     = h.get('iteration')
                he     = hist_by_iter.get(it, {})

                icon   = {'ACCEPTED': '[WINNER]', 'TESTED': '[    ]',
                          'REJECTED': '[ REJ ]', 'FAILED': '[ ERR ]',
                          'RETAINED': '[ KEEP ]', 'ELIMINATED': '[ ELIM]'}.get(result, '[  ?  ]')
                d_s    = f"{d_ofv:.2f}" if d_ofv is not None else "N/A"
                ofv_s  = f"{he.get('ofv'):.2f}" if he.get('ofv') is not None else "N/A"

                shrink = he.get('avg_eta_shrinkage')
                sh_s   = (f"{shrink:.1f}%!" if shrink is not None and shrink > 50
                          else (f"{shrink:.1f}%" if shrink is not None else "N/A"))

                rse    = he.get('max_rse')
                rse_s  = (f"{rse:.1f}%!" if rse is not None and rse > 50
                          else (f"{rse:.1f}%" if rse is not None else "N/A"))

                cov_s  = "Yes" if he.get('covariance_successful', h.get('cov_ok')) else "No"
                star   = " *" if name in confirmed_names and result == 'ACCEPTED' else ""

                iter_s = f"{it}" if it is not None else "?"
                print(f"R{rnd:<4} {iter_s:<6} {name:<18} {icon:<10} {d_s:<9} {ofv_s:<10} "
                      f"{sh_s:<10} {rse_s:<10} {cov_s:<5}{star}")

            print(f"{'-'*82}")

            # 라운드별 요약
            if self.scm_eliminated:
                print(f"\nEliminated in backward step (p<0.01 not met on removal):")
                for e in self.scm_eliminated:
                    print(f"  Backward Round {e['round']}: {e['name']} "
                          f"(ΔOFV on removal=+{e['delta_ofv']:.2f}, iter {e['iteration']})")

            if self.scm_confirmed:
                print(f"\nFinal covariates (survived forward selection + backward elimination):")
                for c in self.scm_confirmed:
                    print(f"  Round {c['round']}: {c['name']} (ΔOFV={c['delta_ofv']:.2f}, iter {c['iteration']})")
                print(f"\nFinal SCM OFV: {self.best_ofv:.2f}")
            else:
                print(f"\nNo covariate survived — base model retained")
            print(f"{'='*70}")

        # ── Best model quality assessment ────────────────────────────────────
        best_entry = hist_by_iter.get(self.best_iteration)
        if best_entry:
            print(f"\n{'='*70}")
            print("FINAL MODEL QUALITY")
            print(f"{'='*70}")
            shrink = best_entry.get('avg_eta_shrinkage')
            if shrink is not None:
                grade = ("EXCELLENT" if shrink < 30 else "GOOD" if shrink < 50
                         else "ACCEPTABLE" if shrink < 70 else "CONCERNING" if shrink < 90
                         else "CRITICAL")
                print(f"ETA Shrinkage : {shrink:.1f}%  [{grade}]")
            print(f"Covariance    : {'SUCCESS' if best_entry.get('covariance_successful') else 'FAILED'}")
            ofv = best_entry.get('ofv')
            if ofv is not None:
                print(f"OFV           : {ofv:.2f}")
            print(f"{'='*70}")

        # ── Save final model ─────────────────────────────────────────────────
        final_file = f"{self.output_base}_final.txt"
        best_file  = f"{self.output_base}_iter{self.best_iteration}.txt"
        if os.path.exists(best_file):
            import shutil
            shutil.copy(best_file, final_file)
            print(f"\n[OK] Best model saved to: {final_file}")

        print("=" * 70 + "\n")

        return {
            'total_iterations': self.iteration,
            'best_iteration': self.best_iteration,
            'best_ofv': self.best_ofv,
            'history': self.improvement_history,
            'final_file': final_file if os.path.exists(best_file) else None
        }
