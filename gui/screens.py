"""
This module defines custom modal screens for file and directory selection.

It includes a filtered directory tree and navigation controls.
"""
from typing import Iterable, Optional, List
from pathlib import Path
from textual.screen import ModalScreen
from textual.widgets import DirectoryTree, Label, Button, Input
from textual.containers import Container, Horizontal
from textual.binding import Binding


class FilteredDirectoryTree(DirectoryTree):
    """
    A DirectoryTree widget that filters displayed files based on extensions.
    """
    def __init__(self, path: Path, allowed_extensions: Optional[List[str]] = None, dirs_only: bool = False, **kwargs):
        """
        Args:
            path (Path): Root path.
            allowed_extensions (List[str], optional): List of extensions to show (e.g. ['.json', '.yaml']).
            dirs_only (bool): If True, only directories are shown.
        """
        self.allowed_extensions = allowed_extensions
        self.dirs_only = dirs_only
        super().__init__(path, **kwargs)

    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        """
        Filters the list of paths based on the configuration.
        """
        filtered = []
        for path in paths:
            # Dirs only ignores files
            if self.dirs_only and not path.is_dir():
                continue

            if path.is_dir():
                filtered.append(path)

            # If file - check if it is allowed
            elif self.allowed_extensions:
                if path.suffix.lower() in self.allowed_extensions:
                    filtered.append(path)

            # If no filters - take everything
            else:
                filtered.append(path)

        return filtered


class BaseFileSystemPicker(ModalScreen[Path]):
    """
    Base class for file/folder picker modals. Handles navigation logic and history.
    """

    BINDINGS = [
        Binding("backspace", "go_up", "Go Up"),
    ]

    def __init__(self, start_path=None):
        super().__init__()
        # Start path
        start = Path(start_path).resolve() if start_path else Path.cwd()

        # Path history for Back/Forward buttons
        self.history = [start]
        self.history_index = 0
        self.current_path = start

    def compose_nav_bar(self):
        """Creates the top navigation bar with Back, Forward, Up buttons and path display."""
        with Horizontal(classes="nav-bar"):
            # Nav buttons
            with Horizontal(classes="nav-buttons-group"):
                yield Button("<", id="btn_back", classes="nav-btn", disabled=True)
                yield Button(">", id="btn_fwd", classes="nav-btn", disabled=True)
                yield Button("Up", id="btn_up", classes="nav-btn")

            # Path string
            yield Label(str(self.current_path), id="path_display")

    def on_mount(self):
        # Focus tree on mount
        self.query_one(DirectoryTree).focus()

    def update_path_display(self):
        self.query_one("#path_display", Label).update(str(self.current_path))

        # Enable/Disable Back/Forward buttons
        b_back = self.query_one("#btn_back", Button)
        b_fwd = self.query_one("#btn_fwd", Button)

        b_back.disabled = (self.history_index <= 0)
        b_fwd.disabled = (self.history_index >= len(self.history) - 1)

    def navigate_to(self, path: Path, add_to_history=True):
        if not path.exists() or not path.is_dir():
            self.notify("Cannot access directory", severity="error")
            return

        self.current_path = path
        tree = self.query_one(DirectoryTree)
        tree.path = path

        if add_to_history:
            self.history = self.history[:self.history_index + 1]
            self.history.append(path)
            self.history_index += 1

        self.update_path_display()

    # --- DIRECTORY TREE CALLBACKS ---
    def on_directory_tree_directory_selected(self, event: DirectoryTree.DirectorySelected):
        event.stop()  # Prevent default behaviour
        self.navigate_to(event.path)

    def action_go_up(self):
        self.do_up()

    # --- BUTTON CALLBACKS ---
    def on_button_pressed(self, event: Button.Pressed):
        bid = event.button.id

        if bid == "btn_up":
            self.do_up()

        elif bid == "btn_back":
            if self.history_index > 0:
                self.history_index -= 1
                self.navigate_to(self.history[self.history_index], add_to_history=False)

        elif bid == "btn_fwd":
            if self.history_index < len(self.history) - 1:
                self.history_index += 1
                self.navigate_to(self.history[self.history_index], add_to_history=False)

        elif bid == "btn_new_folder":
            self.app.push_screen(NewFolderModal(self.current_path), self.after_folder_created)

        elif bid == "cancel":
            self.dismiss(None)

        elif bid == "select":
            # Only for DirPicker
            if hasattr(self, 'selected_dir'):  # Fallback safety
                self.dismiss(self.selected_dir)
            else:
                self.dismiss(self.current_path)  # Fallback

    def do_up(self):
        """Navigates to the parent directory."""
        parent = self.current_path.parent

        # Prevent going out of root if not allowed
        if parent != self.current_path:
            self.navigate_to(parent)

    def after_folder_created(self, folder_name: str):
        """Callback after the NewFolderModal closes."""
        if folder_name:
            try:
                new_path = self.current_path / folder_name
                new_path.mkdir(exist_ok=True)
                self.notify(f"Created: {folder_name}")

                # Refresh tree
                self.query_one(DirectoryTree).path = self.current_path

            except Exception as e:
                self.notify(f"Error creating folder: {e}", severity="error")


class FilePicker(BaseFileSystemPicker):
    """
    Modal screen for selecting a single file.
    """
    def __init__(self, start_path=None, allowed_extensions: List[str] = None):
        super().__init__(start_path)
        self.allowed_extensions = allowed_extensions

    def compose(self):
        with Container(classes="modal-container"):
            yield Label(f"Select File", classes="modal-header")

            yield from self.compose_nav_bar()

            yield FilteredDirectoryTree(
                self.current_path,
                allowed_extensions=self.allowed_extensions,
                id="tree"
            )

            # FOOTER
            with Horizontal(classes="modal-footer"):
                yield Button("+ New Folder", id="btn_new_folder", variant="warning", classes="btn-left")
                yield Label("", classes="spacer")
                yield Button("Cancel", variant="error", id="cancel")

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected):
        self.dismiss(event.path)


class DirPicker(BaseFileSystemPicker):
    """
    Modal screen for selecting a directory.
    """
    def compose(self):
        with Container(classes="modal-container"):
            yield Label("Select Output Directory", classes="modal-header")
            yield from self.compose_nav_bar()

            # Show only directories
            yield FilteredDirectoryTree(
                self.current_path,
                dirs_only=True,
                id="tree"
            )

            with Horizontal(classes="modal-footer"):
                yield Button("+ New Folder", id="btn_new_folder", variant="warning", classes="btn-left")
                yield Label("", classes="spacer")
                yield Button("Select Current Folder", variant="success", id="select")
                yield Button("Cancel", variant="error", id="cancel")

    def on_button_pressed(self, event: Button.Pressed):
        # Override for select button
        super().on_button_pressed(event)
        if event.button.id == "select":
            # Return currently opened folder
            self.dismiss(self.current_path)


class NewFolderModal(ModalScreen[str]):
    """
    Small modal dialog to input a name for a new folder.
    """
    def __init__(self, base_path):
        super().__init__()
        self.base_path = base_path

    def compose(self):
        with Container(classes="input-modal"):
            yield Label(f"New folder in:\n{self.base_path}")
            yield Input(placeholder="Folder name...", id="folder_name")

            with Horizontal(classes="modal-footer"):
                yield Button("Create", variant="success", id="create")
                yield Button("Cancel", variant="error", id="cancel")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "create":
            val = self.query_one("#folder_name", Input).value
            self.dismiss(val)

        else:
            self.dismiss(None)
