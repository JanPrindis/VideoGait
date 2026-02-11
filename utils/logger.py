"""
This module provides a unified logging interface for the application.

It wraps the `rich` library to provide color-coded console output.
"""
import sys
from rich.console import Console

_console = Console()

def log(module_name: str, message: str, level: str = "info"):
    """
    Logs a message to the console with a standardized format and color.

    Args:
        module_name (str): The name of the module generating the log (e.g., "TRAINER").
        message (str): The log message content.
        level (str): The severity level. Options: 'info' (cyan), 'success' (green), 'warning' (yellow), 'error' (red).
    """
    colors = {
        "info": "cyan",
        "success": "green",
        "warning": "yellow",
        "error": "red"
    }
    color = colors.get(level, "white")
    markup = f"[bold {color}][{module_name}][/] {message}"

    # Check where we are sending the output
    if hasattr(sys.stdout, "log_widget"):
        # GUI: Send raw markup to RichLog(markup=True)
        print(markup)
    else:
        # Standalone - use rich to print
        _console.print(markup)
