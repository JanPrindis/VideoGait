"""
This module defines the main Textual application for the Gait Analysis Dashboard.

It provides a Terminal User Interface (TUI) to configure and run the analysis pipeline,
view execution logs in real-time, and manage input/output paths.
"""
import os
import subprocess
import sys
import platform
from pathlib import Path

import logging
logging.getLogger("asyncio").setLevel(logging.WARNING)

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Header, Footer, Button, Input, Label, RichLog, Select
from textual import work

from main import run_analysis_pipeline
from gui.logging import SmartLogger
from gui.screens import FilePicker, DirPicker
from utils.logger import log


def open_file(path: str):
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":  # macOS
        subprocess.run(["open", path])
    else:  # Linux
        subprocess.run(["xdg-open", path])


class VideoGaitTUI(App):
    """
    The main TUI application class.

    Layout:
    - Sidebar: Inputs for video, config, output path, and run controls.
    - Output Area: Real-time console log and status bar.
    """
    CSS_PATH = "styles.tcss"
    TITLE = "VideoGait Dashboard"
    last_report_path = None

    def compose(self) -> ComposeResult:
        """Constructs the UI layout."""
        yield Header()

        # --- SIDEBAR ---
        with Vertical(id="sidebar"):
            # Video
            yield Label("Input Video:")
            yield Input(placeholder="Select video...", id="input_video")
            yield Button("Browse Video...", id="btn_browse_video")

            # Config
            yield Label("Config File:")
            yield Input(placeholder="Select app config file...", id="input_config")
            yield Button("Browse Config...", id="btn_browse_config")

            # Output override
            yield Label("Output Dir (Optional):")
            yield Input(placeholder="Default: ./results", id="input_output")
            yield Button("Browse Output...", id="btn_browse_output")

            # Name override
            yield Label("Analysis Name:")
            yield Input(placeholder="Auto-generated", id="input_name")

            # Output format
            yield Label("Output Format:")
            yield Select(
                options=[
                    ("PDF Report", "pdf"),
                    ("HTML Static", "html"),
                    ("HTML Interactive", "interactive")
                ],
                value="pdf",
                id="select_output_type",
                allow_blank=False
            )

            # Run
            yield Label("Waiting...", id="step_label")

            yield Button("START ANALYSIS", variant="success", id="btn_run")
            yield Button("OPEN REPORT", variant="primary", id="btn_open_report")

        # --- OUTPUT AREA ---
        with Vertical(id="output-area"):
            yield Label("Execution Log:")
            yield RichLog(id="console_log", markup=True)
            yield Label("Ready", id="status_line", classes="status-box")

        yield Footer()

    # --- BUTTON HANDLERS ---
    def on_button_pressed(self, event: Button.Pressed):
        """Handles button click events to trigger file pickers or start the analysis."""
        bid = event.button.id

        if bid == "btn_browse_video":
            # Filter out only video files
            self.push_screen(
                FilePicker("./videos", allowed_extensions=[".mov", ".mp4", ".avi", ".mkv"]),
                self.set_video_path
            )

        elif bid == "btn_browse_config":
            # Filter out only config files
            start_cfg = Path("./configs/app").resolve()
            if not start_cfg.exists(): start_cfg = Path("./")

            self.push_screen(
                FilePicker(start_cfg, allowed_extensions=[".yaml", ".yml"]),
                self.set_config_path
            )

        elif bid == "btn_browse_output":
            self.push_screen(DirPicker("./"), self.set_output_path)

        elif bid == "btn_run":
            self.run_process()

        if bid == "btn_open_report":
            if self.last_report_path and os.path.exists(self.last_report_path):
                open_file(self.last_report_path)
            else:
                self.notify("Report file not found!", severity="error")

    # --- CALLBACKS ---
    def update_input_end_focused(self, input_id: str, value: str):
        """Updates an Input widget's value and moves focus to the end of the text."""
        inp = self.query_one(input_id, Input)
        inp.value = str(value)
        inp.action_end()

    def set_video_path(self, path: Path):
        """Callback for the video file picker."""
        if path:
            self.update_input_end_focused("#input_video", str(path))

            # Auto-name logic
            name_input = self.query_one("#input_name", Input)
            if not name_input.value:
                name_input.value = path.stem

    def set_config_path(self, path: Path):
        """Callback for the config file picker."""
        if path:
            self.update_input_end_focused("#input_config", str(path))

    def set_output_path(self, path: Path):
        """Callback for the output directory picker."""
        if path:
            self.update_input_end_focused("#input_output", str(path))

    # --- WORKER ---
    @work(thread=True)
    def run_process(self):
        """
        Executes the analysis pipeline in a separate thread.

        It validates inputs, redirects stdout/stderr to the UI log widget,
        and calls the main pipeline function.
        """
        # Get and validate data
        video_path_str = self.query_one("#input_video", Input).value.strip()
        config_path_str = self.query_one("#input_config", Input).value.strip()
        out_override_str = self.query_one("#input_output", Input).value.strip()
        name = self.query_one("#input_name", Input).value.strip() or None
        output_format = self.query_one("#select_output_type", Select).value

        # Widgets
        log_widget = self.query_one("#console_log", RichLog)
        status_widget = self.query_one("#status_line", Label)
        step_lbl = self.query_one("#step_label", Label)
        btn_run = self.query_one("#btn_run", Button)
        btn_open = self.query_one("#btn_open_report", Button)

        self.call_from_thread(status_widget.update, "Running")

        # --- VALIDATION BLOCK ---
        errors = []

        # Video Check
        if not video_path_str:
            errors.append("Video path is empty!")
        else:
            vid_path = Path(video_path_str)
            if not vid_path.exists():
                errors.append(f"Video file not found:\n{vid_path.name}")
            elif not vid_path.is_file():
                errors.append("Video path is not a file!")

        # Config Check
        if not config_path_str:
            errors.append("Config path is empty!")
        else:
            cfg_path = Path(config_path_str)
            if not cfg_path.exists():
                errors.append(f"Config file not found:\n{cfg_path.name}")
            elif not cfg_path.is_file():
                errors.append("Config path is not a file!")

        # Output Check
        if out_override_str:
            out_path = Path(out_override_str)
            try:
                if not out_path.exists() and not out_path.parent.exists():
                    errors.append(f"Output parent directory does not exist:\n{out_path.parent}")
            except Exception:
                errors.append("Invalid output path syntax.")

        # If any errors - stop
        if errors:
            error_msg = "\n".join(errors)
            self.notify("Validation Failed! Check logs.", severity="error", timeout=5)
            self.call_from_thread(log_widget.write, f"[bold red]VALIDATION ERROR:[/]\n{error_msg}")
            return

        # --- UI LOCK ---
        self.call_from_thread(setattr, btn_run, "label", "Running...")
        self.call_from_thread(setattr, btn_run, "disabled", True)
        self.call_from_thread(step_lbl.update, "Initializing...")

        # --- SETUP LOGGING ---
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        logger_out = SmartLogger(log_widget, status_widget, self, is_error=False)
        logger_err = SmartLogger(log_widget, status_widget, self, is_error=True)

        try:
            sys.stdout = logger_out
            sys.stderr = logger_err

            # Hide results button while starting a new analysis
            self.call_from_thread(setattr, btn_open, "display", "none")

            # Clear the output logs before starting a new analysis
            self.call_from_thread(log_widget.clear)

            self.call_from_thread(log_widget.write, f"[bold green]--- STARTING ANALYSIS ---[/]")
            self.call_from_thread(log_widget.write, f"Video: {Path(video_path_str).name}")

            # Run analysis
            result_path = run_analysis_pipeline(
                input_video_path=video_path_str,
                app_config_path=config_path_str,
                analysis_name_override=name,
                output_root_override=out_override_str,
                output_format=output_format
            )

            # Show report button
            if output_format == "pdf":
                ext = "pdf"
            elif output_format == "html" or output_format == "interactive":
                ext = "html"
            else:
                ext = None # Fallback

            suffix = "_interactive" if output_format == "interactive" else ""

            # Only show result button when we know the file type
            if ext:
                report_file = Path(result_path) / f"{name}_report{suffix}.{ext}"

                if report_file.exists():
                    self.last_report_path = str(report_file)
                    self.call_from_thread(setattr, btn_open, "display", "block")
                else:
                    log("GUI", f"Report generated, but could not locate file: {report_file.name}", level="warning")  #
            else:
                log("GUI", f"Preview not supported for '{output_format}'. Visit the results folder to view the output.", level="warning")

            self.call_from_thread(log_widget.write, f"[bold green]Done! Result saved to: {result_path}[/]")
            self.call_from_thread(step_lbl.update, "COMPLETED")
            self.notify("Analysis Finished Successfully!")

        except Exception as e:
            self.call_from_thread(log_widget.write, f"[bold red]CRITICAL ERROR: {e}[/]")
            import traceback
            self.call_from_thread(log_widget.write, traceback.format_exc())
            self.call_from_thread(step_lbl.update, "ERROR")

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            self.call_from_thread(setattr, btn_run, "label", "START ANALYSIS")
            self.call_from_thread(setattr, btn_run, "disabled", False)
            self.call_from_thread(status_widget.update, "Ready")
