import cv2

from Skeletons.halpe_skeleton import HALPE_SKELETON
from utils.visualizer import Visualizer

file_name = "result"

vis = Visualizer(skeleton_definition=HALPE_SKELETON)
vis.visualize(
    original_video="./videos/logitech-1920-60-8.avi",
    visualizer_output_path="./videoResults",
    visualizer_output_file=f"{file_name}.mp4",
    detector_output_path="./inferResults",
    detector_output_file=f"{file_name}.json",
    confidence_threshold=0.6
)

