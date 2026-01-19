import argparse
import ast

def pydict_type(string):
    try:
        # 1. Fix PowerShell/JSON lowercase booleans to Python case
        string = string.replace('true', 'True').replace('false', 'False')
        # 2. Parse as a Python literal
        return ast.literal_eval(string)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Value must be a dict (e.g., '{{'key': True}}'). Error: {e}")