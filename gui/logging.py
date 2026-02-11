"""
This module provides a custom logging handler for the Textual UI.

It intercepts stdout and stderr to display logs in a RichLog widget and
parses TQDM progress bars to update a status label instead of spamming the log.
"""
import re
from textual.widgets import RichLog, Label
from textual.app import App

class SmartLogger:
    def __init__(self, log_widget: RichLog, status_widget: Label, app: App, is_error: bool = False):
        """
        Args:
            log_widget (RichLog): The widget to write standard logs to.
            status_widget (Label): The widget to update with progress bar status.
            app (App): The main application instance (needed for thread-safe calls).
            is_error (bool): Whether this logger is handling stderr.
        """
        self.log_widget = log_widget
        self.status_widget = status_widget
        self.app = app
        self.is_error = is_error

        self.module_pattern = re.compile(r"\[.*\]")
        self.percent_pattern = re.compile(r"(\d+)%")

    def _adapt_tqdm_line(self, line: str) -> str:
        """
        Parses a raw TQDM progress line and reformats it to fit within the
        width of the status widget.
        """
        # Current width - padding
        raw_width = getattr(self.status_widget.size, "width", 0)
        width = max(1, raw_width - 2)

        # Find first and last pipe
        first_pipe = line.find('|')
        last_pipe = line.rfind('|')

        if first_pipe == -1 or last_pipe == -1 or first_pipe == last_pipe:
            # Fallback
            return line[:width]

        prefix = line[:first_pipe].rstrip()
        suffix = line[last_pipe+1:].lstrip()

        # Get percentage from progress bar
        percent_match = self.percent_pattern.search(line)
        if percent_match:
            percent = int(percent_match.group(1))
        else:
            percent = 0  # Fallback

        # Available space for progress bar between prefix and suffix
        available = max(3, width - len(prefix) - len(suffix) - 2)  # 2 spaces

        # Length of filled part
        filled_len = int(round(percent / 100 * available))
        empty_len = available - filled_len

        # Construct new bar
        new_bar = '#' * filled_len + ' ' * empty_len

        final_line = f"{prefix} |{new_bar}| {suffix}"
        return final_line[:width]

    def write(self, message: str):
        """
        Writes a message to the UI. Handles TQDM carriage returns (\r) specially
        by updating the status widget instead of the log.
        """
        if not message or not message.strip():
            return

        # TQDM (\r) - Priority for status bar
        if "\r" in message:
            parts = message.split("\r")
            current_status = parts[-1].strip()
            if current_status:
                # from gui.logging import SmartLogger as SL  # pro přístup k _adapt_tqdm_line
                adapted = self._adapt_tqdm_line(current_status)
                self.app.call_from_thread(self.status_widget.update, adapted)
            return

        clean_msg = message.rstrip()

        # FILTERING
        # Allow print only if:
        # - It uses my format
        # - It's from stderr (is_error=True)
        # - Contains error keywords
        is_formatted = self.module_pattern.search(clean_msg)
        is_critical = any(word in clean_msg.lower() for word in ["error", "exception", "traceback", "failed"])

        if is_formatted or self.is_error or is_critical:
            # Wrap stderr output
            if (self.is_error or is_critical) and "[" not in clean_msg:
                clean_msg = f"[bold red]{clean_msg}[/]"

            self.app.call_from_thread(self.log_widget.write, clean_msg)

    def flush(self):
        pass
