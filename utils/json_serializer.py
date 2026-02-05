import json
import os
from dataclasses import asdict

from utils.gait_structs import GaitEvent, GaitEventType


class KeypointSerializer:
    def __init__(self, output_path: str, output_file_name: str):
        self.output_path = output_path
        self.output_file_name = output_file_name
        self.data = []
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def add_frame(self, frame_number, keypoints):
        self.data.append({
            'image_id':  f'{frame_number}.jpg',
            'category_id': 1, # Human
            'keypoints': keypoints,
            'score': 0,
        })

    def save(self):
        with open(self.output_path + "/" + self.output_file_name, 'w') as f:
            json.dump(self.data, f, indent=4)

    @staticmethod
    def load(input_file_path: str):
        with open(input_file_path, 'r') as f:
            raw_data = json.load(f)

        data = []
        for entry in raw_data:
            parsed_entry = {
                'image_id': entry['image_id'],
                'keypoints': entry['keypoints']
            }
            data.append(parsed_entry)

        return data

class AnalysisSerializer:
    @staticmethod
    def save(output_path: str, output_file_name: str, data: object):
        output_path = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path + "/" + output_file_name, 'w') as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def load(input_file_path: str):
        with open(input_file_path, 'r') as f:
            raw_data = json.load(f)

        return raw_data


class AnnotationSerializer:
    def __init__(self, output_path: str, output_file_name: str, fps: int = 30):
        self.output_path = output_path
        self.output_file_name = output_file_name
        self.data = {
            "metadata": {
                "fps": fps
            },
            "annotations": {
                "left": [],
                "right": []
            }
        }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def add_event(self, leg: str, event: GaitEvent):
        if leg not in self.data["annotations"]:
            raise ValueError("Invalid leg name. Must be 'left' or 'right'.")
        self.data["annotations"][leg].append(asdict(event))

    def set_fps(self, fps: int):
        self.data["metadata"]["fps"] = fps

    def save(self):
        with open(os.path.join(self.output_path, self.output_file_name), 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=4)

    @staticmethod
    def load(input_file_path: str):
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Parse the annotations into GaitEvent objects
        if "annotations" in data:
            for leg in ["left", "right"]:
                if leg in data["annotations"]:
                    event_list = [
                        GaitEvent(
                            frame=event_data["frame"],
                            event_type=GaitEventType(event_data["event_type"])
                        )
                        for event_data in data["annotations"][leg]
                    ]
                    data["annotations"][leg] = event_list
        return data
