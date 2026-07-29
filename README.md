<p align="center">
  <img src="logo.png" alt="PKGPT 2.0 Logo" width="360">
</p>

# PKGPT 2.0 - Pharmacokinetic NONMEM Optimizer

AI-assisted generation and iterative optimization of NONMEM population pharmacokinetic models.

PKGPT analyzes a NONMEM-format dataset, generates an initial control stream, executes NONMEM, evaluates the output, and iteratively proposes model updates. PKGPT 2.0 retains the workflow of the [original PKGPT](https://github.com/Gumgo91/PKGPT) while improving model-development control, safety, reproducibility, and user input.

> **Status:** Research prototype (PKGPT 2.0). Every generated model must be reviewed and validated by a qualified pharmacometrician.

## Overview

PKGPT connects four components of population PK model development:

- Dataset interpretation and candidate-covariate detection
- Complete NONMEM control-stream generation
- NONMEM execution and output parsing
- Phase-specific, iterative model refinement

NONMEM remains the estimation engine. PKGPT is intended to support, not replace, pharmacometric judgment.

## What's New in PKGPT 2.0

- Corrected execution of the five-phase optimization workflow
- Round-based forward selection and automatic backward elimination for SCM
- Optional restriction of SCM to user-selected covariates and target parameters
- Optional preliminary information through `--prior-info`
- PK plausibility context with web-search citation URLs
- Safety checks for covariance, parameter boundaries, implausible OFV, OMEGA collapse, and excessive ETA shrinkage
- Dose-unit and weight-normalization consistency checks
- OpenRouter access to configured Claude, Gemini, and GPT model profiles
- Automatic terminal transcripts and detailed iteration/SCM records

Dataset analysis, control-stream generation, NONMEM execution, result parsing, recursive improvement, and final-model saving from the original workflow are retained.

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
