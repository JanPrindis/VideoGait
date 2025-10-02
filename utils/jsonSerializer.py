import json
import os

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
