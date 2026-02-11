"""
This module provides a unified logging interface for the application.

It wraps the `rich` library to provide color-coded console output.
"""
from rich import print as rprint


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

    rprint(f"[bold {color}][{module_name}][/] {message}")
