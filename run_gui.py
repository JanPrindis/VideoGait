"""
Entry point for the Textual-based GUI (TUI).

Run this script to launch the interactive terminal dashboard for configuring
and running gait analysis pipelines.
"""
from gui.app import VideoGaitTUI

if __name__ == "__main__":
    app = VideoGaitTUI()
    app.run()
