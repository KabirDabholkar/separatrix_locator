"""
Function composition utilities.
"""

from functools import reduce
from typing import Callable, Any


def compose(*functions: Callable) -> Callable:
    """
    Compose multiple functions from right to left.
    
    Parameters:
    -----------
    *functions : callable
        Functions to compose
        
    Returns:
    --------
    callable
        Composed function
        
    Example:
    --------
    >>> f = compose(lambda x: x + 1, lambda x: x * 2)
    >>> f(3)  # Returns (3 * 2) + 1 = 7
    """
    def _compose(f: Callable, g: Callable) -> Callable:
        return lambda x: f(g(x))
    
    return reduce(_compose, functions, lambda x: x)
