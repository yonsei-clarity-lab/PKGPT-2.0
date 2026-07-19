# Changelog

## 2.0.0-rc1 - 2026-07-19

This release candidate updates the previously published PKGPT implementation. The list below contains new behavior and corrections introduced for the 2.0 workflow; capabilities already present in the earlier public version are not repeated.

### Added

- Five-phase model-development workflow with phase-specific prompts and transition criteria.
- Round-based SCM forward screening against a frozen base model.
- Automatic backward elimination after forward selection.
- Phase 5 structural guards that freeze ADVAN/TRANS, compartment structure, estimation, residual error, and IIV while a covariate is tested.
- Plausibility-based physiological ranges and typical values before initial control-stream generation.
- Dose-unit mismatch detection across mg, mg/kg, mcg, mcg/kg, and g candidates.
- Data-driven weight-normalization consistency checks using AMT/WT and AMT*WT variability.
- Compartment-invariance and ADVAN/TRANS compatibility checks during error recovery.
- Automatic terminal transcript output to `<output_base>_terminal.txt`.
- OpenRouter-based model profiles for Claude, Gemini, and GPT families.
- Optional JSON/YAML preliminary-information input through `--prior-info`.
- Optional user-selected SCM covariates and target parameters.
- Final SCM reporting that distinguishes forward winners, retained effects, rejected candidates, and backward-eliminated effects.

### Changed

- Default Phase 1-4 maximum iteration count increased from 10 to 20.
- Phase 5 continues until all configured candidates are tested.
- Phase 2 recovery priority now favors bounds/initial values, IIV simplification, and residual-error correction before compartment changes.
- Boundary-collapsed estimates are not reused blindly as new initial values.
- Initial THETA generation receives plausibility reference values.
- Covariate medians are calculated from the dataset.
- Covariate model-type selection is centralized in the data loader.
- Weight is evaluated through SCM instead of being forced into the base model.
- Shrinkage decisions consistently use the maximum ETA shrinkage.
- RSE values and AI quality recommendations are propagated to subsequent improvement steps.
- Detailed NONMEM output is supplied to error-diagnosis and improvement logic.
- Phase 5 skips an unused AI quality-evaluation call.

### Fixed

- Corrected phase prompt routing that could silently fall back to Phase 1 logic.
- Corrected early-stop routing so a qualified base model can enter Phase 5.
- Corrected stale Phase 5 base OFV/code handling.
- Prevented covariance-failed candidate models from replacing an SCM base.
- Prevented Phase 5 covariate THETA boundaries from being misclassified as residual-error failures.
- Improved repeated-strategy tracking and recovery after repeated syntax or minimization failures.
- Expanded FOCE-I enforcement to reject unsupported estimation-method substitutions.
- Strengthened Phase 2 completion checks with covariance and boundary status.
- Strengthened Phase 2 and Phase 4 prohibitions against premature covariate insertion.
- Included iteration numbers in SCM result reporting.

### Notes

- PK plausibility references use the selected language model's existing knowledge; no live literature-search integration is included.
- Preliminary information supplied with `--prior-info` is optional and is not automatically verified.
- The codebase-audit observations recorded during development are not listed as release features when no corresponding code change was made.

