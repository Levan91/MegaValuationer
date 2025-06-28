#!/usr/bin/env python3
"""
Automator for vapp.py: launch, lint, type-check, and manage the Streamlit app.
"""
import subprocess
import sys
import os
import argparse

REQUIREMENTS = [
    'streamlit', 'prophet', 'pandas', 'numpy', 'scikit-learn', 'plotly', 'openpyxl', 'st-aggrid', 'statsmodels'
]

VAPP_PATH = os.path.join(os.path.dirname(__file__), 'vapp.py')
REQUIREMENTS_PATH = os.path.join(os.path.dirname(__file__), 'requirements.txt')


def check_dependencies():
    """Check if all required packages are installed."""
    import importlib
    missing = []
    for pkg in REQUIREMENTS:
        try:
            importlib.import_module(pkg.replace('-', '_'))
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\n[!] Missing packages: {', '.join(missing)}")
        print(f"    Install with: pip install {' '.join(missing)}\n")
        return False
    return True


def run_streamlit():
    """Run the Streamlit app."""
    print("[Automator] Launching Streamlit app...")
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', VAPP_PATH])


def lint():
    """Run flake8 linter on vapp.py."""
    print("[Automator] Running flake8 linter...")
    result = subprocess.run(['flake8', VAPP_PATH])
    if result.returncode == 0:
        print("[Automator] No lint errors found.")
    else:
        print("[Automator] Lint errors detected.")


def type_check():
    """Run mypy type checker on vapp.py."""
    print("[Automator] Running mypy type checker...")
    result = subprocess.run(['mypy', VAPP_PATH])
    if result.returncode == 0:
        print("[Automator] No type errors found.")
    else:
        print("[Automator] Type errors detected.")


def main():
    parser = argparse.ArgumentParser(description='Automator for vapp.py')
    parser.add_argument('action', nargs='?', default='help',
                        choices=['run', 'lint', 'typecheck', 'help'],
                        help='Action to perform: run, lint, typecheck, help')
    args = parser.parse_args()

    if args.action == 'help':
        print("""
Automator for vapp.py
---------------------
Usage:
  python automator.py run         # Launch the Streamlit app
  python automator.py lint        # Run flake8 linter on vapp.py
  python automator.py typecheck   # Run mypy type checker on vapp.py
  python automator.py help        # Show this help message

Requirements:
  - All Python dependencies in requirements.txt
  - flake8 (for linting)
  - mypy (for type checking)

Example:
  python automator.py run
        """)
        sys.exit(0)

    if not check_dependencies():
        sys.exit(1)

    if args.action == 'run':
        run_streamlit()
    elif args.action == 'lint':
        lint()
    elif args.action == 'typecheck':
        type_check()

if __name__ == '__main__':
    main() 