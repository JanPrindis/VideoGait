import tkinter as tk
from tkinter import filedialog

import cv2
import os

from gaitStructs import GaitEvent, GaitEventType
from utils.jsonSerializer import AnnotationSerializer


class GaitAnnotator:
    def __init__(self, save_root="./annotations"):
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

        while True:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_disp = frame.copy()
            cv2.putText(frame_disp, f"Frame: {self.current_frame}/{self.frame_count}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Gait Annotator", frame_disp)
            key = cv2.waitKeyEx(0)

            if key == 27: # ESC
                print("Saving video")
                self.save_annotations()
                break

            elif key == 2424832:  # left arrow
                self.current_frame = max(0, self.current_frame - 1)

            elif key == 2555904:  # right arrow
                self.current_frame = min(self.frame_count - 1, self.current_frame + 1)

            elif key == ord('a'):
                self.left_events.append(GaitEvent(self.current_frame, GaitEventType.HEEL_STRIKE))
                print(f"[L] Heel strike @ {self.current_frame}")

            elif key == ord('q'):
                self.left_events.append(GaitEvent(self.current_frame, GaitEventType.TOE_OFF))
                print(f"[L] Toe off @ {self.current_frame}")

            elif key == ord('s'):
                self.right_events.append(GaitEvent(self.current_frame, GaitEventType.HEEL_STRIKE))
                print(f"[R] Heel strike @ {self.current_frame}")

            elif key == ord('w'):
                self.right_events.append(GaitEvent(self.current_frame, GaitEventType.TOE_OFF))
                print(f"[R] Toe off @ {self.current_frame}")

            elif key == ord('z'):
                if len(self.left_events) == 0:
                    continue
                e = self.left_events.pop()
                print(f"[L] Removed last event: {e}")

            elif key == ord('x'):
                if len(self.right_events) == 0:
                    continue
                e = self.right_events.pop()
                print(f"[R] Removed last event: {e}")

        self.cap.release()
        cv2.destroyAllWindows()

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
