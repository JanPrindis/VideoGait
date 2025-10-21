from venv import create

import cv2
import glob

from Skeletons.halpe_skeleton import HALPE_SKELETON
from utils.visualizer import Visualizer
from utils.data import create_folder_if_not_exists

video_folder = "silhouette"
result_folder = "sil"

create_folder_if_not_exists(f"./videoResults/{result_folder}")

files = glob.glob(f"./results/{result_folder}/*.json")
files.sort()

for file_path in files:
    file_name = file_path.split("\\")[-1].split(".")[0]

    vis = Visualizer(skeleton_definition=HALPE_SKELETON)
    vis.visualize(
        original_video=f"D:/Semestralka/Health_Gait/generated_videos/{video_folder}/{file_name}.mp4",
        visualizer_output_path=f"./videoResults/{result_folder}",
        visualizer_output_file=f"{file_name}.mp4",
        detector_output_path=f"./results/{result_folder}",
        detector_output_file=f"{file_name}.json",
        confidence_threshold=0.6
    )

#file_name = "patient_00"

# vis = Visualizer(skeleton_definition=HALPE_SKELETON)
# vis.visualize(
#     original_video="./videos/patient_00_i.mp4",
#     visualizer_output_path="./videoResults",
#     visualizer_output_file=f"{file_name}.mp4",
#     detector_output_path="./results",
#     detector_output_file=f"{file_name}.json",
#     confidence_threshold=0.6
# )

