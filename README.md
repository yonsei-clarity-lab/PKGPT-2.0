<p align="center">
  <img src="logo.png" alt="PKGPT Logo" width="360">
</p>

# PKGPT 2.0 - Pharmacokinetic NONMEM Optimizer

AI-assisted generation and iterative optimization of NONMEM population pharmacokinetic models.

PKGPT analyzes a NONMEM-format dataset, generates an initial control stream, executes NONMEM, evaluates the output, and iteratively proposes model updates. PKGPT 2.0 preserves the original workflow while making model development more controlled and reproducible through phase-specific optimization, complete stepwise covariate modeling, stronger model-safety checks, optional user-provided prior information, and OpenRouter-based multi-model access.

> **Status:** Research prototype (`v2.0.0-rc1`). Every generated model must be reviewed and validated by a qualified pharmacometrician.

## Overview

PKGPT is a research-oriented workflow for AI-assisted population PK model development. It connects four components that are usually handled separately:

- **Dataset interpretation:** identifies standard NONMEM columns, observation structure, dosing records, subject counts, and candidate covariates.
- **Model generation:** asks a selected language model to generate a complete NONMEM control stream using the observed data structure and PK plausibility context.
- **NONMEM execution:** runs the configured `nmfe` command and collects the resulting listing and auxiliary files.
- **Iterative model refinement:** parses NONMEM results, identifies convergence and model-quality problems, and requests a targeted update for the current development phase.

The optimizer is not intended to replace pharmacometric judgment. Its purpose is to make model-development steps explicit, reproducible, and easier to inspect while retaining NONMEM as the estimation engine.

## What's New in PKGPT 2.0

PKGPT 2.0 is an update of the [original PKGPT implementation](https://github.com/Gumgo91/PKGPT), not a separate modeling concept. The original sequence - analyze data, generate a NONMEM model, execute NONMEM, interpret the result, and improve the model recursively - remains intact. Version 2.0 strengthens how decisions are made inside that sequence.

| Area | Original PKGPT | PKGPT 2.0 |
|---|---|---|
| Optimization strategy | General recursive improvement over 3-10 iterations | Five phase-specific workflows with explicit transition criteria for structural, estimation, IIV, and covariate decisions |
| Covariate modeling | Covariate relationships could be proposed during model generation or improvement | Complete SCM with frozen-base forward rounds, explicit OFV criteria, and automatic backward elimination |
| Covariate control | Automatic detection of available covariates | Automatic mode remains available, while users can optionally restrict SCM to selected covariates and target parameters |
| Preliminary information | Modeling was primarily guided by the dataset and the language model's existing knowledge | Optional `--prior-info` input can provide drug name or class, nonclinical data, user-supplied published information, study population, and covariates to prioritize |
| Initial values | AI-generated initial parameter estimates | PK plausibility ranges and typical values are supplied as context, and collapsed boundary estimates are not blindly reused |
| Dose units | No dedicated dose-scale safeguard described in the original workflow | Heuristic checks for mg, mg/kg, mcg, mcg/kg, and g, including weight-normalization consistency |
| Model safety | Convergence, OFV, RSE, shrinkage, warnings, and errors guided recursive updates | Adds covariance and boundary gates, compartment-invariance checks, ADVAN/TRANS compatibility checks, and repeated-strategy tracking |
| LLM access | Direct Google Gemini profiles | Unified OpenRouter access to configured Claude, Gemini, and GPT profiles |
| Run records | Iteration control streams, NONMEM listings, history, and final model | Adds an automatic full terminal transcript and detailed SCM outcome reporting |
| Iteration control | A common maximum iteration limit | Phase 1-4 default increases to 20, while Phase 5 continues until applicable SCM candidates are tested |

Functions that already existed in PKGPT - dataset analysis, automatic compartment guidance, complete control-stream generation, NONMEM execution, result parsing, recursive improvement, progress tracking, mock testing, and final-model saving - are retained and are not presented as new 2.0 features.

## Core Features

### Automated dataset analysis

PKGPT summarizes the input dataset before model generation. The analysis includes:

- Detection of common NONMEM columns such as `ID`, `TIME`, `DV`, `AMT`, `EVID`, `MDV`, `CMT`, and `RATE`
- Subject and observation counts
- Dose and sampling-time summaries
- Detection of numeric and categorical covariates
- Dataset-derived medians and reference values for covariate models
- Concentration-time profile information used to support structural-model selection

### Complete NONMEM control-stream generation

The initial-generation prompt requests a complete control stream rather than isolated code fragments. Depending on the data, this can include:

- `$PROBLEM`, `$INPUT`, and `$DATA`
- Appropriate ADVAN/TRANS selection
- `$PK` structural and statistical parameterization
- Initial `$THETA`, `$OMEGA`, and `$SIGMA` values
- Residual-error model
- `$ESTIMATION`, `$COVARIANCE`, and output tables

The generated code is saved before execution so that every proposed model can be reviewed independently.

### NONMEM-aware result interpretation

PKGPT parses model output and tracks information used by subsequent phases:

- Objective Function Value (OFV)
- Minimization and covariance status
- THETA, OMEGA, and SIGMA estimates
- Relative standard errors
- ETA shrinkage
- Boundary and gradient warnings
- Repeated failure patterns
- Model code and iteration history

### Phase-specific model improvement

Instead of sending the same generic improvement request after every run, version 2.0 uses a prompt dedicated to the current model-development problem. This limits unrelated structural changes and makes each iteration easier to interpret.

### Progress and final-model reporting

The console and terminal transcript show phase transitions, iteration-level OFV and quality metrics, SCM candidates, round winners, backward-elimination results, and the selected final control stream.

## Detailed Changes in PKGPT 2.0

The following improvements are new relative to the previously published PKGPT version. Existing capabilities are described in the surrounding sections but are not claimed as new functionality.

### 1. Five-phase model-development workflow

Optimization is now routed through phase-specific prompts and completion criteria:

1. Establish a stable base model
2. Diagnose structural and estimation problems
3. Reduce overfitting
4. Optimize the IIV structure
5. Perform stepwise covariate modeling (SCM)

Phase 2 prioritizes bounds and initial estimates, IIV simplification, and residual-error changes before considering a compartment change. Transition to later phases also requires stronger convergence, covariance, and boundary checks.

### 2. Round-based forward SCM and backward elimination

Phase 5 now performs a full SCM workflow:

- Every candidate in a forward-selection round is compared with the same frozen base model.
- The candidate with the largest significant OFV reduction is selected as the round winner.
- Forward selection uses `p < 0.05` (`Delta OFV < -3.84`, 1 df).
- Backward elimination follows automatically using `p < 0.01` (retained when removal increases OFV by at least `6.63`, 1 df).
- The final report distinguishes selected, retained, rejected, and eliminated effects.
- Phase 5 continues until the configured candidate set is exhausted, independently of the Phase 1-4 iteration limit.

During SCM, the structural model, ADVAN/TRANS choice, estimation method, residual-error model, and IIV structure are frozen. Only the requested covariate effect may change.

### 3. Data-driven covariate handling

- Covariate medians are calculated from the dataset instead of being hard-coded.
- Candidate model type is selected consistently through one data-loader function (for example, power, linear, or categorical forms).
- Covariates are evaluated against both `CL` and `V1` when appropriate instead of being forced onto one parameter.
- Weight is no longer automatically embedded as a fixed allometric effect in the base model. It is evaluated through the same SCM procedure as other covariates.
- Optional user selection can restrict SCM to specified covariates and target parameters.

### 4. PK plausibility context and safer initial values

Before initial model generation, PKGPT asks the selected language model for plausible physiological ranges and typical values for key PK parameters. These values are used as reference context for initial THETA values and later plausibility checks.

This is based on the language model's existing knowledge. PKGPT 2.0 does **not** perform a live literature search and does not treat generated reference values as verified evidence.

Boundary estimates from a failed iteration are not blindly reused as the next initial values. When an estimate collapses at a bound, the phase-specific prompt instructs the model to return to a physiologically plausible reference value.

### 5. Dose-unit mismatch detection

PKGPT compares observed dose statistics with the plausible single-dose reference and evaluates candidate scales such as mg, mg/kg, mcg, mcg/kg, and g. Weight-normalized consistency checks based on `AMT/WT` and `AMT*WT` variability reduce false positive per-kilogram interpretations.

When a likely mismatch is detected, the initial-generation prompt requests an appropriate `F1` conversion. This remains a heuristic and must be checked against the study protocol and dataset documentation.

### 6. Improved quality and recovery logic

- RSE metrics are passed into AI quality evaluation.
- ETA shrinkage decisions use the maximum shrinkage rather than a mixture of mean and maximum values.
- AI recommendations and critical issues are forwarded to the next improvement prompt.
- Detailed NONMEM output is available to the improvement step so boundary and gradient information is not lost.
- Repeated failed strategies are tracked to discourage identical retries.
- Compartment-count invariance and ADVAN/TRANS compatibility checks guard error recovery.
- Repeated syntax or minimization failures can restore the best known code before retrying.
- Phase 5 skips an unused AI quality call, reducing unnecessary latency and API usage.

### 7. OpenRouter multi-model support

Version 2.0 uses a unified OpenRouter client. The current CLI profiles are:

- `claude-sonnet` (default)
- `claude-opus`
- `gemini-flash`
- `gemini-flash-lite`
- `gemini-pro`
- `gpt-4.1`
- `gpt-5.5`

Actual model availability, identifiers, pricing, and access depend on the user's OpenRouter account.

### 8. Optional user-provided preliminary information

The optional `--prior-info` argument accepts JSON, YAML, or YML containing user-supplied context such as:

- Drug name and drug class
- Nonclinical information
- Previously published information supplied by the user
- Study population
- Covariates to prioritize and their target PK parameters

Only populated fields are added to the plausibility and initial-model prompts. The supplied information is treated as context rather than automatically verified truth. If it conflicts with the observed dataset, the data should take priority.

With `covariates.mode` set to `user_selected`, Phase 5 tests only the listed candidates. Selected candidates are not automatically accepted; they still undergo the normal SCM statistical tests.

### 9. Automatic terminal transcript

Each run writes console output and errors to:

```text
<output_base>_terminal.txt
```

The transcript covers normal completion, errors, and keyboard interruption handling, while output remains visible in the terminal.

## Requirements

- Python 3.8+
- NONMEM 7.x and a working `nmfe` command
- An OpenRouter API key
- Windows, Linux, or macOS for Python execution; NONMEM command configuration is environment-specific

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

Edit `.env` and provide your key:

```text
OPENROUTER_API_KEY=your_api_key_here
```

Never commit the real `.env` file or an API key.

## Basic Usage

```bash
python pkgpt_optimizer.py <data_file> <output_base> --nmfe <nmfe_command>
```

Windows example:

```bat
python pkgpt_optimizer.py dataset/theo.csv output_theo --nmfe C:\nm75g64\run\nmfe75.bat --model claude-sonnet
```

Additional examples:

```bash
# Set Phase 1-4 iteration limits
python pkgpt_optimizer.py dataset/tobramycin.csv output_tobra --min-iter 3 --max-iter 20

# Select another configured OpenRouter profile
python pkgpt_optimizer.py dataset/wafarin.csv output_warfarin --model gemini-pro

# Add optional preliminary information
python pkgpt_optimizer.py dataset/theo.csv output_theo --prior-info examples/prior_info.theophylline.json
```

Use `python pkgpt_optimizer.py --help` for the current command-line options.

## Command-line Options

```text
positional arguments:
  data_file             Input CSV dataset in NONMEM format
  output_base           Prefix used for generated files

options:
  --min-iter N          Minimum optimization iterations (default: 3)
  --max-iter N          Maximum Phase 1-4 iterations (default: 20)
  --nmfe CMD            NONMEM execution command (default: nmfe75)
  --model MODEL         OpenRouter model profile
  --api-key KEY         OpenRouter API key; environment variable is preferred
  --prior-info FILE     Optional JSON/YAML preliminary-information file
  --version             Display the PKGPT version
  -h, --help            Display CLI help
```

`--max-iter` limits Phase 1-4 only. Once Phase 5 starts, SCM continues until the applicable forward or backward candidate set has been evaluated.

## Included Dataset Examples

The repository currently includes three CSV datasets used during development and validation:

- `dataset/theo.csv`
- `dataset/tobramycin.csv`
- `dataset/wafarin.csv`

The `wafarin.csv` filename is retained for compatibility with the supplied project files. It represents the warfarin example dataset.

Example commands:

```bat
python pkgpt_optimizer.py dataset/theo.csv theo_run1 --nmfe C:\nm75g64\run\nmfe75.bat --model claude-sonnet
python pkgpt_optimizer.py dataset/tobramycin.csv tobra_run1 --nmfe C:\nm75g64\run\nmfe75.bat --model claude-sonnet
python pkgpt_optimizer.py dataset/wafarin.csv warfarin_run1 --nmfe C:\nm75g64\run\nmfe75.bat --model claude-sonnet
```

## Prior-information Format

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
      },
      {
        "name": "ALBUMIN",
        "target_parameters": ["CL"]
      }
    ]
  }
}
```

Use `"mode": "auto"` or omit `covariates` to retain automatic candidate detection. YAML input requires PyYAML, which is included in `requirements.txt`.

## How It Works

### Step 1: Load and characterize the dataset

PKGPT loads the CSV file, maps standard NONMEM columns, summarizes subjects and observations, identifies candidate covariates, and computes dose and covariate statistics. This metadata is included in later prompts so the language model does not need to infer the entire dataset structure from column names alone.

### Step 2: Build PK plausibility references

The selected OpenRouter model receives the dataset metadata and is asked to identify a likely drug context and plausible ranges for major PK parameters such as clearance and volume. When `--prior-info` is supplied, populated drug, nonclinical, published-data, and population fields are added as user-provided context.

The resulting values are working references, not verified literature values. They are used to:

- Guide initial THETA selection
- Detect implausible estimates
- Avoid carrying boundary-collapsed values into later iterations
- Support dose-unit consistency checks

### Step 3: Generate and run the initial model

PKGPT generates `iter0`, prepares the first executable iteration, and calls NONMEM using the configured `--nmfe` command. The listing file is parsed and added to the model history.

### Step 4: Phase 1 - establish a stable base model

The first phase focuses on obtaining an estimable base model. Typical priorities include valid control-stream syntax, successful minimization, a suitable structural model, and a reasonable residual-error specification.

### Step 5: Phase 2 - diagnose structure and estimation

Phase 2 addresses convergence, covariance, boundary, and structural problems in a controlled order:

1. Correct bounds or initial estimates
2. Simplify poorly identified IIV terms
3. Adjust the residual-error model
4. Change compartment structure only when earlier corrections are insufficient

Covariates are prohibited during this phase so that base-model problems are not hidden by premature covariate effects.

### Step 6: Phase 3 - reduce overfitting

The optimizer evaluates parameter identifiability, shrinkage, unnecessary IIV complexity, and unstable covariance structures. The goal is to obtain a parsimonious model before further optimization.

### Step 7: Phase 4 - optimize IIV

Phase 4 refines the random-effects structure after the structural and residual components are stable. Covariate additions remain prohibited until the model is ready to enter SCM.

### Step 8: Phase 5 - stepwise covariate modeling

Forward SCM is performed in rounds:

1. Freeze the current round base OFV and code.
2. Generate one model for each untested candidate effect.
3. Run each candidate from the same round base.
4. Calculate candidate `Delta OFV = candidate OFV - round base OFV`.
5. Keep candidates with covariance success and `Delta OFV < -3.84`.
6. Select the largest significant OFV reduction as the winner.
7. Start another round from the winning model.

After no additional forward winner remains, backward elimination begins:

1. Remove one confirmed effect at a time from the same full-model base.
2. Calculate the OFV cost of removal.
3. Eliminate an effect when removal costs less than `6.63` OFV units.
4. Repeat until every remaining effect satisfies the retention criterion.

This round-based design prevents candidates in the same round from being compared against different base models.

### Step 9: Save the selected model and reports

The final control stream, terminal transcript, iteration history, and SCM summary are written using the requested output prefix. Generated NONMEM files remain available locally but are ignored by Git through `.gitignore`.

## Available OpenRouter Model Profiles

| CLI profile | Configured provider family | Intended use |
|---|---|---|
| `claude-sonnet` | Anthropic Claude | Default balanced profile |
| `claude-opus` | Anthropic Claude | Complex reasoning and review |
| `gemini-flash` | Google Gemini | Faster iterations |
| `gemini-flash-lite` | Google Gemini | Lower-cost testing |
| `gemini-pro` | Google Gemini | More complex analysis |
| `gpt-4.1` | OpenAI GPT | General model-development tasks |
| `gpt-5.5` | OpenAI GPT | Configured advanced GPT profile |

The profile names map to model identifiers in `modules/openrouter_client.py`. OpenRouter may change model availability or identifiers; review that file and the account's available models before a long run.

## Dataset Format

Input data should be a CSV file following NONMEM conventions.

Common required columns:

- `ID`: subject identifier
- `TIME`: time after dose or study time
- `DV`: dependent variable
- `AMT`: dose amount
- `EVID`: event identifier
- `MDV`: missing-DV flag

Optional columns may include `CMT`, `RATE`, and subject-level covariates such as `WT`, `AGE`, `SEX`, `CRCL`, or laboratory values.

Column meaning and units must be checked before modeling. Automated detection does not replace a data specification.

## Output Files

Typical outputs include:

- `<output>_iter0.txt`: initial control stream
- `<output>_iterN.txt`: control stream for iteration N
- `<output>_iterN.lst`: NONMEM listing output
- `<output>_final.txt`: selected final control stream
- `<output>_terminal.txt`: complete terminal transcript

NONMEM may generate additional table, covariance, XML, and compilation files. These are excluded by `.gitignore`.

## Repository Structure

```text
PKGPT-2.0/
|-- pkgpt_optimizer.py
|-- modules/
|   |-- optimizer.py
|   |-- data_loader.py
|   |-- nonmem_parser.py
|   |-- openrouter_client.py
|   |-- phase_transition_manager.py
|   |-- prior_info.py
|   |-- prompt_templates.py
|   `-- prompts/
|-- dataset/
|-- examples/
|-- requirements.txt
`-- .env.example
```

## Testing Without NONMEM

If the configured NONMEM command cannot be found, PKGPT can generate a mock listing file so that the orchestration and recursive-improvement flow can be demonstrated. Mock output is useful for software testing only.

It cannot validate:

- NONMEM syntax or compilation
- Numerical minimization
- Parameter estimates
- Covariance results
- OFV-based model decisions
- SCM significance

Any pharmacometric conclusion therefore requires a real NONMEM installation and review of the actual output files.

## Troubleshooting

### `OPENROUTER_API_KEY` is not set

Confirm that `.env` exists in the project directory and contains:

```text
OPENROUTER_API_KEY=your_api_key_here
```

The key can also be supplied with `--api-key`, but command-line keys may be retained in shell history and are not recommended for routine use.

### NONMEM command not found

Pass the complete executable or batch-file path:

```bat
python pkgpt_optimizer.py dataset/theo.csv output_theo --nmfe C:\nm75g64\run\nmfe75.bat
```

If a mock output is created, do not interpret it as a successful NONMEM run.

### Dataset columns are not detected correctly

Check that the file:

- Is a readable CSV with a header row
- Uses consistent column names
- Contains numeric values in PK and dosing fields
- Uses `EVID` and `MDV` consistently
- Documents dose, time, and concentration units

### Prior-information file fails to load

The file must use `.json`, `.yaml`, or `.yml`, and the top level must be an object/mapping. JSON requires no optional parser. YAML requires the PyYAML dependency from `requirements.txt`.

### Requested covariate is not found

In `user_selected` mode, candidate names must match dataset columns. PKGPT prints a warning for requested covariates that are absent. An empty applicable candidate set means there is nothing to test in SCM.

### Phase 5 runs longer than `--max-iter`

This is expected in version 2.0. `--max-iter` applies to Phase 1-4, while Phase 5 continues until its candidate-testing process is complete.

### Dose scaling looks incorrect

Review the study protocol, AMT definition, WT units, and concentration units. Dose-unit detection is a safeguard, not an authoritative replacement for source documentation. Inspect the generated `$PK` block and terminal transcript before continuing.

## Advanced Configuration

- Edit `modules/openrouter_client.py` to maintain model-profile mappings.
- Edit `modules/prompts/` to review phase-specific modeling instructions.
- Edit `modules/prompt_templates.py` to review shared initial-generation and evaluation prompts.
- Edit `modules/data_loader.py` to extend dataset-column or covariate-model detection.
- Edit `modules/nonmem_parser.py` to extend NONMEM result extraction.

Changes to prompts or thresholds can alter scientific behavior and should be versioned and validated with representative datasets.

## Contributing

PKGPT is a research codebase. Bug reports and focused pull requests should include:

- A minimal reproducible dataset or synthetic example when sharing is permitted
- The exact command used
- PKGPT version or commit
- Relevant terminal transcript
- NONMEM version
- Expected and observed behavior

Do not include API keys, confidential clinical data, or restricted NONMEM installation files in issues or pull requests.

## License

PKGPT 2.0 is distributed under the MIT License. See [`LICENSE`](LICENSE) for the complete terms and preserved copyright notices for the original and 2.0 contributors.

## Limitations

- Language-model output can be incomplete or incorrect.
- Plausibility references are not a substitute for literature review.
- Dose-unit detection is heuristic and requires protocol confirmation.
- SCM thresholds alone do not establish clinical relevance.
- Results depend on dataset quality, NONMEM configuration, and model availability.
- Regulatory or production use requires independent review, diagnostics, qualification, and validation.

## Authors

- **Hoyoung Kwack** - [hoyoung0104@yonsei.ac.kr](mailto:hoyoung0104@yonsei.ac.kr)
- **Park Youngseo** - [selly4577@yonsei.ac.kr](mailto:selly4577@yonsei.ac.kr)
- **Lee Seungwoo** - [seungwu210@yonsei.ac.kr](mailto:seungwu210@yonsei.ac.kr)

## Citation

If you use PKGPT 2.0 in research, please cite the software repository and the applicable version or commit:

```text
Kwack H, Park Y, Lee S. PKGPT 2.0: Pharmacokinetic NONMEM Optimizer.
Version 2.0.0-rc1. GitHub repository, 2026.
```

## Disclaimer

PKGPT is a research tool that generates and modifies NONMEM code using language models. Users are responsible for reviewing source data, model assumptions, parameterization, convergence, diagnostics, covariate decisions, and all generated code before using any result in research, clinical, production, or regulatory work.
