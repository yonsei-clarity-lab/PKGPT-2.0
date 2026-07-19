#!/usr/bin/env python
"""
PKGPT - LLM-assisted Pharmacokinetic NONMEM Optimizer
Recursive optimization of NONMEM control stream files
"""

import sys
import os
import argparse
from dotenv import load_dotenv

# Windows consoles often default to a legacy code page (e.g. cp949 for Korean
# locales) instead of UTF-8. LLM-generated text frequently contains unicode
# punctuation (en/em dashes, multiplication sign, etc.) that cp949 cannot
# encode, which crashes bare print() calls throughout the optimizer (and
# silently empties out try/except-guarded steps like plausibility bounds
# generation). Force UTF-8 on stdout/stderr regardless of the console's
# active code page so no print() call can fail on encoding.
#
# Separately: when stdout/stderr are redirected to a file or pipe (e.g. any
# background/non-interactive run), Python switches from line-buffered to
# full block buffering, so nothing appears in the output file until the
# internal buffer fills or the process exits. line_buffering=True forces a
# flush after every newline so progress is visible in real time.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

# Add modules directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.optimizer import NONMEMOptimizer
from modules.prior_info import load_prior_info


class _Tee:
    """Duplicates every write to multiple streams (e.g. the real console +
    a log file), so a run's full terminal transcript is saved automatically
    instead of having to be copy-pasted out of the console by hand."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()

    def isatty(self):
        return False


def main():
    """Main CLI entry point"""

    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(
        description='PKGPT - Recursive NONMEM Model Optimizer using OpenRouter models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python pkgpt_optimizer.py dataset/theophylline_nonmem.csv output_theo

  # Specify iterations
  python pkgpt_optimizer.py dataset/warfarin_nonmem.csv output_warf --min-iter 5 --max-iter 15

  # Custom NONMEM command
  python pkgpt_optimizer.py data.csv output --nmfe nmfe74

Environment Variables:
  OPENROUTER_API_KEY    OpenRouter API key (required)

The optimizer will:
  1. Analyze your pharmacokinetic dataset
  2. Generate an initial NONMEM control stream
  3. Execute NONMEM and parse results
  4. Recursively improve the model through phase-specific optimization
  5. Save the best model as <output>_final.txt

Output files:
  <output>_iter0.txt       Initial model
  <output>_iter1.txt       First iteration
  <output>_iter1.lst       NONMEM output for iteration 1
  ...
  <output>_final.txt       Best model (copy of best iteration)
        """
    )

    parser.add_argument(
        'data_file',
        help='Path to input CSV dataset (NONMEM format)'
    )

    parser.add_argument(
        'output_base',
        help='Base name for output files (without extension)'
    )

    parser.add_argument(
        '--min-iter',
        type=int,
        default=3,
        metavar='N',
        help='Minimum number of iterations (default: 3)'
    )

    parser.add_argument(
        '--max-iter',
        type=int,
        default=20,
        metavar='N',
        help='Maximum number of iterations for Phase 1-4 (default: 20). Phase 5 (SCM) runs until all covariate candidates are tested regardless of this value.'
    )

    parser.add_argument(
        '--nmfe',
        default='nmfe75',
        metavar='CMD',
        help='NONMEM execution command (default: nmfe75)'
    )

    parser.add_argument(
        '--model',
        choices=['claude-sonnet', 'claude-opus', 'gemini-flash', 'gemini-flash-lite','gemini-pro', 'gpt-4.1', 'gpt-5.5'],
        default='claude-sonnet',
        help='OpenRouter model profile to use (default: claude-sonnet)'
    )

    parser.add_argument(
        '--api-key',
        help='OpenRouter API key (overrides OPENROUTER_API_KEY env var)'
    )

    parser.add_argument(
        '--prior-info',
        metavar='FILE',
        help='Optional JSON/YAML prior-information file. With covariates.mode=user_selected, SCM tests only the listed covariates.'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='PKGPT v2.0.0-rc1'
    )

    args = parser.parse_args()

    # Validation
    if not os.path.exists(args.data_file):
        print(f"Error: Data file not found: {args.data_file}")
        sys.exit(1)

    if args.min_iter < 1:
        print("Error: Minimum iterations must be >= 1")
        sys.exit(1)

    if args.max_iter < args.min_iter:
        print("Error: Maximum iterations must be >= minimum iterations")
        sys.exit(1)

    prior_info = {}
    if args.prior_info:
        try:
            prior_info = load_prior_info(args.prior_info)
        except ValueError as exc:
            print(f"Error: {exc}")
            sys.exit(1)

    # Check for API key
    api_key = args.api_key or os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Please set it in your environment or .env file, or use --api-key")
        print("\nExample:")
        print("  set OPENROUTER_API_KEY=your-api-key-here")
        print("  python pkgpt_optimizer.py data.csv output")
        sys.exit(1)

    # Save the full terminal transcript of this run to a .txt file automatically,
    # so it doesn't have to be copy-pasted out of the console by hand afterward.
    log_path = f"{args.output_base}_terminal.txt"
    log_file = open(log_path, 'w', encoding='utf-8', errors='replace')
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_stdout, log_file)
    sys.stderr = _Tee(orig_stderr, log_file)
    print(f"[LOG] Full terminal output will be saved to: {log_path}")

    try:
        # Create optimizer
        optimizer = NONMEMOptimizer(
            data_file=args.data_file,
            output_base=args.output_base,
            api_key=api_key,
            min_iterations=args.min_iter,
            max_iterations=args.max_iter,
            nmfe_command=args.nmfe,
            model=args.model,
            prior_info=prior_info
        )

        # Run optimization
        results = optimizer.run()

        # Success
        if results['final_file']:
            print(f"\n[OK] Optimization completed successfully!")
            print(f"[OK] Best model saved to: {results['final_file']}")
            return 0
        else:
            print("\n[WARNING] Optimization completed with issues")
            print("[WARNING] Check iteration files for details")
            return 1

    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user")
        return 130

    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        sys.stdout, sys.stderr = orig_stdout, orig_stderr
        log_file.close()


if __name__ == '__main__':
    sys.exit(main())
