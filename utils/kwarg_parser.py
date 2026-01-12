import ast
from typing import List, Any

def parse_value(kwarg_str: str) -> List[Any] | None:
    """Parse a 'key=value' string and return the parsed key and value.

    The function strips whitespace, splits on the first '=' and attempts to
    parse the value using `ast.literal_eval`. This allows numeric, boolean,
    list, and dict literals (e.g., 123, True, [1, 2], {"a": 1}). If parsing
    fails, the value is returned as a stripped string. Malformed inputs
    (missing '=') are skipped and a warning is printed.

    Args:
        kwarg_str: Input string in the form 'key=value'.

    Returns:
        A two-item list `[key, value]` where `key` is a `str` and `value` is the
        parsed Python object (or a string). Returns `None` if the input is malformed.
    """
    kwarg_str = kwarg_str.strip()
    
    # Check for the equals sign first
    if "=" not in kwarg_str:
        print(f"⚠️  Skipping malformed pair: '{kwarg_str}' (missing '=')")
        return None 

    # Split exactly once at the first equals sign
    key, val_str = kwarg_str.split("=", 1)
    
    # Clean up and parse the value
    try:
        value = ast.literal_eval(val_str.strip())
    except (ValueError, SyntaxError):
        value = val_str.strip()
        
    return [key.strip(), value]
