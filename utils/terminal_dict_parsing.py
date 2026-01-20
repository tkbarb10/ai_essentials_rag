"""Argument parsing utilities for CLI dictionary inputs.

This module provides custom argparse type converters for parsing dictionary
arguments from the command line, with cross-platform boolean handling.
"""

import argparse
import ast


def pydict_type(string):
    """Parse a string argument as a Python dictionary.

    Converts a string representation of a dict to an actual Python dict object.
    Handles PowerShell/JSON-style lowercase booleans (true/false) by converting
    them to Python's True/False before parsing.

    Args:
        string: String representation of a Python dict (e.g., "{'key': True}").

    Returns:
        Parsed Python dictionary.

    Raises:
        argparse.ArgumentTypeError: If the string cannot be parsed as a dict.
    """
    try:
        string = string.replace('true', 'True').replace('false', 'False')
        return ast.literal_eval(string)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Value must be a dict (e.g., '{{'key': True}}'). Error: {e}")