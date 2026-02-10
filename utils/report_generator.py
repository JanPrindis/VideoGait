"""
This module handles the generation of comprehensive gait analysis reports.

It supports multiple output formats (HTML, PDF, Interactive HTML) and uses Jinja2 templates
to render analysis results, metadata, and visualizations into a polished document.
"""
import os
import json
import datetime
import pathlib
import sys

import yaml

from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML

project_root = pathlib.Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class ReportGenerator:
    """
    Generates reports (HTML, PDF, Interactive) from gait analysis data.
    """
    def __init__(self, app_config: dict, output_dir: str, template_dir: str = "./templates"):
        """
        Initializes the report generator.

        Args:
            app_config (dict): Application configuration dictionary.
            output_dir (str): Directory where reports will be saved.
            template_dir (str, optional): Directory containing Jinja2 templates. Defaults to "./templates".
        """
        self.app_config = app_config
        self.output_dir = output_dir
        self.template_dir = template_dir

        os.makedirs(self.output_dir, exist_ok=True)

        self.env = Environment(
            loader=FileSystemLoader(searchpath=self.template_dir),
            autoescape=select_autoescape(['html', 'xml'])
        )

        self.env.filters['fmt'] = self._format_float_safe
        self._img_cache = {}

        self.analysis_config_meta = self._extract_config_metadata()

    @staticmethod
    def _load_yaml(rel_path):
        """Helper to safely load a YAML file relative to the project root."""
        try:
            full_path = project_root / rel_path
            if full_path.exists():
                with open(full_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
        except Exception as e:
            print(f"[Report Warning] Failed to load config at {rel_path}: {e}")
        return {}

    def _extract_config_metadata(self):
        """
        Extracts metadata about the models used (Pose Detector, Event Detector)
        from the application configuration for display in the report.
        """
        meta = {
            "pose_model": "Unknown",
            "event_method": "Unknown",
            "event_model_details": "-"
        }

        # POSE DETECTOR
        pose_cfg = self.app_config.get("pose_detector", {})
        config_path = pose_cfg.get("config_path")
        if config_path:
            # Load pose detector config
            sub_config = self._load_yaml(config_path)
            model_type = sub_config.get("type", "Unknown")
            model_name = sub_config.get("model_name", "")

            if model_name:
                meta["pose_model"] = f"{model_type} ({model_name})"
            else:
                meta["pose_model"] = model_type

        # EVENT DETECTOR
        event_cfg = self.app_config.get("event_detector", {})
        method = event_cfg.get("method", "Unknown")
        meta["event_method"] = method

        if method == "Heuristic":
            # Heuristic -> method
            meta["event_model_details"] = event_cfg.get("heuristic", {}).get("method", "Default")

        elif method == "NeuralNet":
            # NeuralNet -> experiment_path -> config.yaml -> model.type
            nn_cfg = event_cfg.get("neural_net", {})
            exp_path = nn_cfg.get("experiment_path")
            if exp_path:
                # Load trining config
                train_config_path = os.path.join(exp_path, "config.yaml")
                train_config = self._load_yaml(train_config_path)
                meta["event_model_details"] = train_config.get("model", {}).get("type", "Unknown NN Model")

        return meta

    @staticmethod
    def _format_float_safe(value, precision=2):
        """Jinja2 filter to format floats safely, handling None values."""
        try:
            if value is None:
                return "-"
            return f"{float(value):.{precision}f}"

        except (ValueError, TypeError):
            return "-"

    def _resolve_image_path(self, filename: str, search_dir: str):
        """
        Resolves the path to an image file for inclusion in the report.
        Handles relative paths for HTML and absolute URIs for PDF generation.
        """
        if not filename:
            return None

        full_path = os.path.abspath(os.path.join(search_dir, filename))

        if os.path.exists(full_path):
            abs_output_dir = os.path.abspath(self.output_dir)

            try:
                rel_path = os.path.relpath(full_path, abs_output_dir)
                return rel_path.replace(os.sep, '/')

            except ValueError:
                return pathlib.Path(full_path).as_uri()

        return None

    def _discover_videos(self, video_dir: str):
        """Scans the video directory for valid video files to include in the report."""
        videos = []
        if not video_dir or not os.path.exists(video_dir):
            return videos

        valid_ext = ('.mp4', '.webm', '.mov', '.mkv')
        for f in sorted(os.listdir(video_dir)):
            if f.lower().endswith(valid_ext):
                full_path = os.path.join(video_dir, f)
                nice_name = f.rsplit('.', 1)[0].replace('_', ' ').strip().capitalize()
                try:
                    rel_path = os.path.relpath(full_path, self.output_dir)

                except ValueError:
                    rel_path = full_path

                videos.append({
                    "path": rel_path.replace(os.sep, '/'),
                    "name": nice_name,
                    "filename": f
                })

        return videos

    @staticmethod
    def _group_phases(phases, valid_ranges):
        """
        Groups detected gait phases into clips based on valid analysis ranges.
        Useful for displaying phases per clip in the report.
        """
        if not phases:
            return []

        if not valid_ranges:
            return [{"name": "Full Video", "phases": phases}]

        grouped = []
        for i, (r_start, r_end) in enumerate(valid_ranges):
            in_range = []

            for p in phases:
                if 'frames' in p:
                    p_start, p_end = p['frames'][0], p['frames'][1]
                else:
                    p_start, p_end = p.get('start', 0), p.get('end', 0)

                p_center = (p_start + p_end) / 2

                if r_start <= p_center <= r_end:
                    in_range.append(p)

            if in_range:
                grouped.append({"name": f"Clip {i + 1} (Frames {r_start}-{r_end})", "phases": in_range})

        return grouped

    def export(self,
               analysis_source: str | dict,
               graphs_dir: str = None,
               video_dir: str = None,
               filename_base: str = "gait_report"):
        """
        Generates and saves the report in the configured format.

        Args:
            analysis_source (str | dict): Path to analysis JSON or the data dictionary itself.
            graphs_dir (str, optional): Directory containing generated plots. Defaults to output_dir.
            video_dir (str, optional): Directory containing processed videos. Defaults to output_dir.
            filename_base (str, optional): Base filename for the report. Defaults to "gait_report".
        """

        # Get export type from app config
        output_type = self.app_config.get("visualization", {}).get("text_output_type", "pdf").lower()

        if output_type == "console":
            print("[Report] Skipping file generation (Config: console)")
            return

        # Load Data
        data = {}
        if isinstance(analysis_source, str):
            if os.path.exists(analysis_source):
                with open(analysis_source, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                print(f"[Error] Analysis file not found")
                return
        elif isinstance(analysis_source, dict):
            data = analysis_source

        # Paths & Helpers
        img_search_path = graphs_dir if graphs_dir else self.output_dir
        vid_search_path = video_dir if video_dir else self.output_dir
        valid_ranges = data.get("metadata", {}).get("valid_ranges", [])

        # Process Grouped Phases
        if 'detailed_statistics' in data:
            for side in ['left', 'right']:
                side_data = data['detailed_statistics'].get(side, {})
                for joint, j_data in side_data.items():
                    if 'phases' in j_data:
                        j_data['grouped_phases'] = self._group_phases(j_data['phases'], valid_ranges)

        # Joint Detection
        available_joints = set()
        stats_source = data.get("detailed_statistics", {}).get("left", {}) or data.get("kinematics_stats", {})
        for key in stats_source.keys():
            if "HIP" in key.upper(): available_joints.add("Hip")
            if "KNEE" in key.upper(): available_joints.add("Knee")
            if "ANKLE" in key.upper(): available_joints.add("Ankle")

        sort_order = {"Hip": 1, "Knee": 2, "Ankle": 3}
        sorted_joints = sorted(list(available_joints), key=lambda x: sort_order.get(x, 99))
        all_videos = self._discover_videos(vid_search_path)

        # Context
        context = {
            "d": data,
            "meta": data.get("metadata", {}),
            "config": self.analysis_config_meta,
            "date": datetime.datetime.now().strftime("%d.%m.%Y %H:%M"),
            "has_kinematics": len(sorted_joints) > 0,
            "joints": sorted_joints,
            "videos": all_videos,
            "img": {
                "gantt": self._resolve_image_path("01_gantt_chart.png", img_search_path),
                "pie": self._resolve_image_path("01_phases_pie_triple.png", img_search_path),
                "com": self._resolve_image_path("02_detail_com_height.png", img_search_path),
                "cycle": lambda j: self._resolve_image_path(f"01_avg_cycle_{j.lower()}.png", img_search_path),
                "timeline": lambda s, j: self._resolve_image_path(f"02_detail_{s}_{j.lower()}.png", img_search_path),
                "symmetry": lambda j: self._resolve_image_path(f"03_symmetry_{j.lower()}.png", img_search_path),
                "cyclogram": lambda j1, j2: self._resolve_image_path(f"03_cyclogram_{j1.lower()}_{j2.lower()}.png", img_search_path)
            }
        }

        # --- GENERATION LOGIC ---
        try:
            if output_type == "html":
                # HTML mode
                print("[Report] Rendering HTML Dashboard...")
                template = self.env.get_template("report_html.html")
                html_content = template.render(**context)

                html_path = os.path.join(self.output_dir, f"{filename_base}.html")
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                print(f"[Report] Saved HTML: {html_path}")

            elif output_type == "pdf":
                # PDF mode
                print("[Report] Rendering PDF Report...")
                template = self.env.get_template("report_pdf.html")
                # Remder html file to string
                pdf_html_content = template.render(**context)

                pdf_path = os.path.join(self.output_dir, f"{filename_base}.pdf")

                # Send to WeasyPrint
                HTML(string=pdf_html_content, base_url=self.output_dir).write_pdf(pdf_path)
                print(f"[Report] Saved PDF: {pdf_path}")

            elif output_type == "interactive":

                # Filter out videos - if < 3, use all, if all 4 are loaded, ignore the original
                if len(all_videos) >= 4:
                    keywords = ['inspection', 'kinematics', 'logic', 'overlay']
                    filtered_videos = [v for v in all_videos if any(k in v['filename'].lower() for k in keywords)]
                    if filtered_videos:
                        context['videos'] = filtered_videos[:3]

                print("[Report] Rendering Interactive Dashboard (Plotly)...")
                template = self.env.get_template("report_html_interactive.html")
                html_content = template.render(**context)

                # Save as _interactive.html
                html_path = os.path.join(self.output_dir, f"{filename_base}_interactive.html")
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                print(f"[Report] Saved Interactive HTML: {html_path}")

        except Exception as e:
            print(f"[Error] Report generation failed: {e}")
            import traceback
            traceback.print_exc()
