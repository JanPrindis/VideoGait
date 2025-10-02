import os
import cv2

from rich.progress import Progress, BarColumn, MofNCompleteColumn, TimeRemainingColumn, TextColumn, TimeElapsedColumn
from Skeletons.skeletons import SkeletonDefinition
from utils.jsonSerializer import KeypointSerializer

class Visualizer:
    def __init__(self, skeleton_definition: SkeletonDefinition):
        self.skeleton = skeleton_definition
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(
                style = "red",
                complete_style = "bold blue",
                finished_style = "bold green"
            ),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
        )

    def draw_skeleton(self, image, frame_data, confidence_threshold):
        if not frame_data:
            return image

        left_keypoint_color = self.skeleton.colors["left_side_keypoint"]
        right_keypoint_color = self.skeleton.colors["right_side_keypoint"]
        left_link_color = self.skeleton.colors["left_side_link"]
        right_link_color = self.skeleton.colors["right_side_link"]

        keypoints = frame_data["keypoints"]

        for link_name, (start, end) in self.skeleton.links.items():
            start_point = keypoints[start * 3:start * 3 + 3]  # x, y, confidence
            end_point = keypoints[end * 3:end * 3 + 3]

            if start_point[2] >= confidence_threshold and end_point[2] >= confidence_threshold:
                start_x, start_y = start_point[0], start_point[1]
                end_x, end_y = end_point[0], end_point[1]

                color = right_link_color if start % 2 != 0 and end % 2 != 0 else left_link_color
                cv2.line(image, (int(start_x), int(start_y)), (int(end_x), int(end_y)), color, 2, cv2.LINE_AA)

        for i in range(0, len(keypoints), 3):
            x = keypoints[i]
            y = keypoints[i + 1]
            confidence = keypoints[i + 2]

            if confidence >= confidence_threshold and (i // 3) in self.skeleton.keypoints.__members__.values():
                color = right_keypoint_color if i // 3 % 2 != 0 else left_keypoint_color
                border_color = right_link_color if i // 3 % 2 != 0 else left_link_color

                cv2.circle(image, (int(x), int(y)), 5, color, -1, cv2.LINE_AA)
                cv2.circle(image, (int(x), int(y)), 5, border_color, 1, cv2.LINE_AA)

        return image

    def visualize(
            self,
            original_video: str,
            visualizer_output_path: str,
            visualizer_output_file: str,
            detector_output_path: str,
            detector_output_file: str,
            confidence_threshold=0.5):

        output_path_full = os.path.join(visualizer_output_path, visualizer_output_file)
        os.makedirs(os.path.dirname(output_path_full), exist_ok=True)

        cap = cv2.VideoCapture(original_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(output_path_full, -1, fps, (width, height))

        keypoints = KeypointSerializer.load(
            os.path.join(detector_output_path, detector_output_file)
        )

        with Progress() as progress:
            task = progress.add_task("Creating video...", total=total_frames)
            current_frame = 0
            current_keypoint_index = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Check matching frame
                if len(keypoints) > current_keypoint_index and f'{current_frame}.jpg' == keypoints[current_keypoint_index]["image_id"]:
                    frame = self.draw_skeleton(frame, keypoints[current_keypoint_index], confidence_threshold)
                    current_keypoint_index += 1

                cv2.putText(frame, f'{current_frame}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
                out.write(frame)

                current_frame += 1
                progress.advance(task_id=task, advance=1)

            progress.stop()

        cap.release()
        out.release()
        return
