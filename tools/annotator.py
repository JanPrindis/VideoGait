import tkinter as tk
from tkinter import filedialog

import cv2
import os

from utils.gait_structs import GaitEvent, GaitEventType
from utils.json_serializer import AnnotationSerializer


class GaitAnnotator:
    def __init__(self, save_root="../annotations"):
        self.video_path = self.select_video_file()
        if not self.video_path:
            print("Error, no video selected")
            return

        self.save_root = save_root
        os.makedirs(save_root, exist_ok=True)

        self.left_events = []
        self.right_events = []

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Unable to open video: {self.video_path}")

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.current_frame = 0

        self.last_action_message = "Started."
    @staticmethod
    def select_video_file():
        root = tk.Tk()
        root.withdraw()
        return filedialog.askopenfilename(
            title="Select a video to annotate",
            initialdir=".",
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
        )

    def annotate(self):
        cv2.namedWindow("Gait Annotator", cv2.WINDOW_NORMAL)

        last_frame_idx = -1
        current_image = None

        while True:
            if self.current_frame != last_frame_idx:
                if self.current_frame != last_frame_idx + 1:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                
                ret, frame = self.cap.read()
                if not ret:
                    break
                current_image = frame
                last_frame_idx = self.current_frame

            frame_disp = current_image.copy()
            self._draw_hud(frame_disp)

            cv2.imshow("Gait Annotator", frame_disp)
            key = cv2.waitKeyEx(0)

            if key == 27: # ESC
                print("Saving video")
                self.save_annotations()
                break

            elif key == 3014656: # DEL
                print("Exiting without saving")
                break

            elif key == 2424832:  # left arrow
                self.current_frame = max(0, self.current_frame - 1)

            elif key == 2555904:  # right arrow
                self.current_frame = min(self.frame_count - 1, self.current_frame + 1)

            elif key == ord('a'):
                self.left_events.append(GaitEvent(self.current_frame, GaitEventType.HEEL_STRIKE))
                msg = f"[L] Heel strike @ {self.current_frame}"
                print(msg)
                self.last_action_message = msg

            elif key == ord('q'):
                self.left_events.append(GaitEvent(self.current_frame, GaitEventType.TOE_OFF))
                msg = f"[L] Toe off @ {self.current_frame}"
                print(msg)
                self.last_action_message = msg

            elif key == ord('s'):
                self.right_events.append(GaitEvent(self.current_frame, GaitEventType.HEEL_STRIKE))
                msg = f"[R] Heel strike @ {self.current_frame}"
                print(msg)
                self.last_action_message = msg

            elif key == ord('w'):
                self.right_events.append(GaitEvent(self.current_frame, GaitEventType.TOE_OFF))
                msg = f"[R] Toe off @ {self.current_frame}"
                print(msg)
                self.last_action_message = msg

            elif key == ord('z'):
                if len(self.left_events) == 0:
                    continue
                e = self.left_events.pop()
                msg = f"[L] Removed: {e.event_type.name} @ {e.frame}"
                print(msg)
                self.last_action_message = msg

            elif key == ord('x'):
                if len(self.right_events) == 0:
                    continue
                e = self.right_events.pop()
                msg = f"[R] Removed: {e.event_type.name} @ {e.frame}"
                print(msg)
                self.last_action_message = msg

        self.cap.release()
        cv2.destroyAllWindows()

    def _draw_hud(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        y_pos = 40
        x_pos = 20

        def draw_text(text, y, color, scale=font_scale, th=thickness):
            cv2.putText(frame, text, (x_pos, y), font, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
            cv2.putText(frame, text, (x_pos, y), font, scale, color, th, cv2.LINE_AA)

        # Frame counter
        frame_text = f"Frame: {self.current_frame} / {self.frame_count}"
        draw_text(frame_text, y_pos, (50, 255, 50))
        y_pos += 35

        # Event counters
        left_hs = sum(1 for e in self.left_events if e.event_type == GaitEventType.HEEL_STRIKE)
        left_to = sum(1 for e in self.left_events if e.event_type == GaitEventType.TOE_OFF)
        right_hs = sum(1 for e in self.right_events if e.event_type == GaitEventType.HEEL_STRIKE)
        right_to = sum(1 for e in self.right_events if e.event_type == GaitEventType.TOE_OFF)

        left_text = f"Left : HS={left_hs}, TO={left_to}"
        right_text = f"Right: HS={right_hs}, TO={right_to}"
        draw_text(left_text, y_pos, (255, 200, 50))
        y_pos += 30
        draw_text(right_text, y_pos, (50, 150, 255))
        y_pos += 40

        # Last action
        action_text = f"Last: {self.last_action_message}"
        draw_text(action_text, y_pos, (230, 230, 230), scale=0.6, th=1)

    def save_annotations(self):
        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        output_file = f"{base_name}.json"
        output_path = os.path.join(self.save_root, output_file)

        serializer = AnnotationSerializer(output_path=self.save_root, output_file_name=output_file, fps=self.fps)
        for e in self.left_events:
            serializer.add_event(leg="left", event=e)

        for e in self.right_events:
            serializer.add_event(leg="right", event=e)

        serializer.save()
        print(f"Output saved as: {output_path}")


if __name__ == "__main__":
    annotator = GaitAnnotator()
    if annotator.video_path:
        annotator.annotate()
