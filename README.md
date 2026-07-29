<p align="center">
  <img src="logo.png" alt="PKGPT 2.0 Logo" width="360">
</p>

# PKGPT 2.0 - Pharmacokinetic NONMEM Optimizer

AI-assisted generation and iterative optimization of NONMEM population pharmacokinetic models.

PKGPT analyzes a NONMEM-format dataset, generates an initial control stream, executes NONMEM, evaluates the output, and iteratively proposes model updates. PKGPT 2.0 retains the workflow of the [original PKGPT](https://github.com/Gumgo91/PKGPT) while improving model-development control, safety, reproducibility, and user input.

> **Status:** Research prototype (PKGPT 2.0). Every generated model must be reviewed and validated by a qualified pharmacometrician.

## Overview

PKGPT automatically generates and iteratively improves NONMEM population pharmacokinetic models. The system:

1. **Analyzes** the dataset structure and concentration-time profile
2. **Generates** a complete initial NONMEM control stream
3. **Executes** NONMEM and parses model results
4. **Improves** the model through a five-phase optimization workflow
5. **Performs** forward selection and backward elimination for covariates
6. **Saves** the selected model, iteration history, and terminal transcript

NONMEM remains the estimation engine. PKGPT is intended to support, not replace, pharmacometric judgment.

## Features

- **Automated model generation:** creates complete NONMEM control streams from NONMEM-format datasets
- **Automatic compartment guidance:** evaluates concentration-time profiles and supports data-informed selection between one- and two-compartment models
- **NONMEM-aware parsing:** extracts OFV, parameter estimates, RSE, shrinkage, covariance status, warnings, and errors
- **Recursive optimization:** proposes targeted model changes based on the current NONMEM results
- **Phase-specific development:** separates base-model stabilization, structural diagnosis, overfitting control, IIV optimization, and covariate modeling
- **Stepwise covariate modeling:** performs round-based forward selection followed by automatic backward elimination
- **Flexible covariate control:** supports automatic candidate detection or user-selected covariates and target parameters
- **Preliminary-information input:** accepts optional drug, nonclinical, published-data, population, and covariate context
- **Multi-model access:** supports configured Claude, Gemini, and GPT profiles through OpenRouter
- **Progress and run records:** saves model files, NONMEM listings, SCM decisions, and a complete terminal transcript

## What's New in PKGPT 2.0

PKGPT 2.0 is an updated version of the [original PKGPT by Gumgo91](https://github.com/Gumgo91/PKGPT). The original repository already provided dataset analysis, automatic one- versus two-compartment guidance, complete NONMEM control-stream generation, NONMEM execution, result parsing, recursive AI-guided improvement, progress tracking, and final-model saving. These retained capabilities are not presented as new PKGPT 2.0 functions.

PKGPT 2.0 adds or corrects the following behavior:

### Five-phase optimization

The existing five-phase design is made operational as a controlled sequence:

- Phase-specific prompts are routed according to the actual current phase.
- A qualified base model can transition into Phase 5 instead of stopping before covariate analysis.
- Phase 2 completion requires successful minimization and covariance without important boundary warnings.
- Phase 2 recovery prioritizes initial values and bounds, IIV simplification, and residual-error correction before changing compartment structure.
- Covariate insertion is restricted during Phase 2 and Phase 4 so that base-model problems are not hidden by premature covariate effects.
- Repeated syntax or minimization failures can restore the best known model before another recovery attempt.

### Complete stepwise covariate modeling

Phase 5 now implements a deterministic SCM workflow rather than relying only on general AI-guided covariate suggestions:

- Every forward candidate in a round is tested from the same frozen base model.
- The largest statistically significant OFV improvement is selected.
- Forward selection uses `Delta OFV < -3.84` (1 df).
- Automatic backward elimination follows using an OFV increase of `6.63` (1 df).
- Phase 5 continues until the applicable candidate set has been evaluated, independently of the Phase 1-4 iteration limit.
- The selected forward winner becomes the correctly recorded base for the next round and for backward elimination.
- The final SCM iteration is retained in model history before the final summary is produced.
- Forward and backward results are distinguished in covariate history.
- Candidate, winner, retained, rejected, unsafe, and eliminated outcomes are recorded with their iteration numbers.

During an SCM test, the structural model, ADVAN/TRANS choice, estimation method, residual-error model, and IIV structure are frozen. The generated model is also checked for unintended covariates or target parameters.

### User control of covariate analysis

Automatic covariate detection remains available. Users can optionally restrict Phase 5 to selected dataset covariates and specify which PK parameters, such as `CL` or `V1`, should be tested.

Additional covariate-handling changes include:

- SCM centering medians are calculated from the dataset rather than fixed reference values.
- Candidate effect forms are selected consistently from dataset characteristics.
- A covariate can be evaluated against more than one relevant PK parameter.
- Weight is evaluated through SCM rather than being automatically forced into every base model.
- A user-selected candidate is prioritized for testing but is not automatically accepted; it must satisfy the same SCM criteria.

### Optional preliminary information

The `--prior-info` option allows user-provided context to be included when available:

- Drug name and pharmacological class
- Nonclinical PK information
- Previously published information supplied by the user
- Study population
- Covariates and PK parameters to prioritize

Only populated fields are added to the prompts. The workflow continues to run when this information is not provided, and observed dataset information takes priority when supplied context conflicts with the data.

### PK plausibility and citation context

Plausible PK ranges and typical values are generated before the initial control stream and used as working references for initial estimates and later plausibility review.

- Parameter requests follow the selected structure: `CL` and `V` for one compartment, with `Q` and `V2` added for two compartments.
- Boundary-collapsed parameter estimates are not blindly reused as the next initial values.
- Web-search citation URLs can be returned with the plausibility context.
- Citation URLs are saved for follow-up review, but the software does not automatically verify that a particular sentence or table directly supports each individual PK parameter.

### Stronger model-safety checks

PKGPT 2.0 adds checks around covariance status, parameter boundaries, compartment consistency, ADVAN/TRANS compatibility, repeated failed strategies, and estimation-method substitutions.

For Phase 5, candidate-level numerical safety is assessed before forward or backward decisions are finalized:

- Covariance-failed candidates cannot replace the SCM base.
- OFV below `-50` is treated as implausible for the current safety gate.
- Complete OMEGA collapse is flagged when all extracted OMEGA estimates are below `0.001`.
- Maximum ETA shrinkage above `95%` is treated as unsafe.
- Unsafe forward candidates are rejected even when their OFV appears favorable.
- Unsafe backward-removal models do not justify eliminating the retained covariate.

Shrinkage decisions use maximum ETA shrinkage consistently, and RSE values, AI quality findings, detailed NONMEM output, and critical issues are propagated to subsequent improvement steps.

### Dose-unit safeguards

Dose-scale checks consider mg, mg/kg, mcg, mcg/kg, and g interpretations together with `AMT/WT` and `AMT*WT` variability. When a likely mismatch is detected, the generated model can apply an `F1` conversion. These checks are safeguards and still require confirmation from the protocol and data specification.

### Multi-LLM access and reproducible records

The original public version supported several Gemini profiles. PKGPT 2.0 extends model access through OpenRouter to configured Claude, Gemini, and GPT profiles. The model selected with `--model` is applied consistently to initial generation, iterative improvement, and structural-guard retry calls.

Each run also produces:

- A complete `<output_base>_terminal.txt` transcript
- Iteration-level control streams and NONMEM listings
- Phase transitions and quality metrics
- Forward and backward SCM candidate outcomes
- Final selected covariates and safety-rejection status

The default Phase 1-4 maximum is increased from 10 to 20 iterations. Phase 5 is governed by completion of its candidate-testing procedure rather than that general iteration limit.

The original capabilities for dataset analysis, automatic compartment guidance, control-stream generation, NONMEM execution, result parsing, recursive improvement, and final-model saving remain part of PKGPT 2.0.

## Core Features

### Dataset analysis

PKGPT detects common NONMEM columns, summarizes subjects, observations, doses, and sampling times, and identifies numeric and categorical covariates. Dataset-derived medians can be used as SCM reference values.

### Phase-specific optimization

Model development is organized into five phases:

1. Establish a stable base model
2. Diagnose structural and estimation problems
3. Reduce overfitting
4. Optimize the IIV structure
5. Perform stepwise covariate modeling

### Stepwise covariate modeling

Forward SCM tests candidates against a frozen round base and selects the largest significant OFV improvement (`Delta OFV < -3.84`, 1 df). Backward elimination then evaluates retained effects using an OFV increase of `6.63` (1 df).

Candidate models must also pass numerical safety checks before selection. Covariance failure, implausible OFV, collapsed OMEGA values, or excessive ETA shrinkage can prevent an unsafe candidate from replacing the SCM base.

### Optional preliminary information

The `--prior-info` option accepts JSON or YAML context including:

- Drug name or class
- Nonclinical information
- User-supplied published information
- Study population
- Covariates and PK parameters to prioritize

Only populated fields are used. Preliminary information guides model generation but is not treated as automatically verified evidence.

### Run records

PKGPT saves iteration control streams, NONMEM listings, the selected final model, SCM decisions, and a complete terminal transcript.

## Requirements

- Python 3.8+
- NONMEM 7.x with a working `nmfe` command
- OpenRouter API key

## Installation

```bash
python -m venv .venv
```

Windows:

```bat
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Linux or macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add the OpenRouter API key to `.env`:

```text
OPENROUTER_API_KEY=your_api_key_here
```

Never commit the real `.env` file or an API key.

## Usage

```bash
python pkgpt_optimizer.py <data_file> <output_base> --nmfe <nmfe_command>
```

Windows example:

```bat
python pkgpt_optimizer.py dataset/theo.csv output_theo --nmfe C:\nm75g64\run\nmfe75.bat --model claude-sonnet
```

With preliminary information:

```bat
python pkgpt_optimizer.py dataset/theo.csv output_theo --nmfe C:\nm75g64\run\nmfe75.bat --model gemini-pro --prior-info examples/prior_info.theophylline.json
```

Use `python pkgpt_optimizer.py --help` for all command-line options.

## OpenRouter Model Profiles

| CLI profile | Provider family |
|---|---|
| `claude-sonnet` | Anthropic Claude |
| `claude-opus` | Anthropic Claude |
| `gemini-flash` | Google Gemini |
| `gemini-flash-lite` | Google Gemini |
| `gemini-pro` | Google Gemini |
| `gpt-4.1` | OpenAI GPT |
| `gpt-5.5` | OpenAI GPT |

Model identifiers are configured in `modules/openrouter_client.py`. Availability, access, and usage charges depend on the user's OpenRouter account.

## Preliminary-information Format

```json
{
  "drug": {
    "name": "Example compound",
    "class": "small_molecule"
  },
  "nonclinical_data": {
    "summary": "Optional nonclinical PK summary"
  },
  "published_data": {
    "summary": "Optional published information supplied by the user"
  },
  "population": {
    "group": "healthy_volunteers"
  },
  "covariates": {
    "mode": "user_selected",
    "candidates": [
      {
        "name": "WT",
        "target_parameters": ["CL", "V1"]
      }
    ]
  }
}
```

Use `"mode": "auto"` or omit `covariates` to retain automatic candidate detection. Selected candidates must exist in the dataset and still undergo the normal SCM criteria.

## Dataset Format

Input data should be a CSV file following NONMEM conventions. Common columns include:

- `ID`: subject identifier
- `TIME`: study or sampling time
- `DV`: dependent variable
- `AMT`: dose amount
- `EVID`: event identifier
- `MDV`: missing-DV flag

Optional columns can include `CMT`, `RATE`, and subject-level covariates such as `WT`, `AGE`, `SEX`, and `CRCL`. Column meanings and units must be confirmed from the data specification.

## Output Files

Typical outputs include:

- `<output>_iter0.txt`: initial control stream
- `<output>_iterN.txt`: iteration control stream
- `<output>_iterN.lst`: NONMEM listing
- `<output>_final.txt`: selected final control stream
- `<output>_terminal.txt`: terminal transcript

## Included Example Datasets

- `dataset/theo.csv`
- `dataset/tobramycin.csv`
- `dataset/wafarin.csv`

The `wafarin.csv` filename is retained for compatibility with the supplied project files and represents the warfarin example dataset.

## Limitations

- Automatic structural-model development currently supports one- and two-compartment models. Three-compartment and TMDD models are outside the current implementation.
- Citation URLs are reference links returned from web search. PKGPT does not automatically verify that a specific sentence or table directly supports each PK parameter range.
- Language-model output can be incomplete or incorrect.
- Plausibility context and dose-unit detection require confirmation against source literature, protocols, and data specifications.
- SCM thresholds alone do not establish clinical relevance.
- Final model selection requires expert review, diagnostics, qualification, and validation.

## Security and Data Handling

Do not commit API keys, confidential clinical data, restricted NONMEM files, or generated run outputs containing sensitive information. Review institutional data-handling requirements before sending study-derived information to an external model provider.

## Acknowledgments

PKGPT 2.0 builds on the [original PKGPT implementation](https://github.com/Gumgo91/PKGPT) developed by Hyunseung Kong and Hoyoung Kwack. The applicable original copyright notice is preserved in [`LICENSE`](LICENSE).

## Authors

- **Hoyoung Kwack** - [hoyoung0104@yonsei.ac.kr](mailto:hoyoung0104@yonsei.ac.kr)
- **Park Youngseo** - [selly4577@yonsei.ac.kr](mailto:selly4577@yonsei.ac.kr)
- **Lee Seungwoo** - [seungwu210@yonsei.ac.kr](mailto:seungwu210@yonsei.ac.kr)

## Citation

```text
Kwack H, Park Y, Lee S. PKGPT 2.0: Pharmacokinetic NONMEM Optimizer.
Version 2.0. GitHub repository, 2026.
```

## License

PKGPT 2.0 is distributed under the MIT License. See [`LICENSE`](LICENSE).

## Disclaimer

PKGPT is a research tool that generates and modifies NONMEM code using language models. Users are responsible for reviewing source data, model assumptions, parameterization, convergence, diagnostics, covariate decisions, and generated code before using any result in research, clinical, production, or regulatory work.
