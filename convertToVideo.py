import cv2
import glob

from utils.data import create_folder_if_not_exists

import numpy as np
from tqdm import tqdm

root_folder_path = "D:/Semestralka/Health_Gait/"
output_folder_path = "D:/Semestralka/Health_Gait/generated_videos/segmentation"
# output_folder_path = "D:/Semestralka/Health_Gait/generated_videos/silhouette"
create_folder_if_not_exists(output_folder_path)

patients = glob.glob(f"{root_folder_path}/semantic_segmentation/PA*")
# patients = glob.glob(f"{root_folder_path}/silhouette/PA*")
patients.sort()

fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

progress = tqdm(total=len(patients), desc="Creating videos")

missing_list = []

for i, patient_path in enumerate(patients):
    patient_r = glob.glob(f"{patient_path}/UGS/WoJ_1_DensePose/*.png")
    patient_l = glob.glob(f"{patient_path}/UGS/WoJ_2_DensePose/*.png")

    # patient_r = glob.glob(f"{patient_path}/UGS/WoJ_1_YOLOV8/*.jpg")
    # patient_l = glob.glob(f"{patient_path}/UGS/WoJ_2_YOLOV8/*.jpg")

    patient_r.sort()
    patient_l.sort()

    if not patient_r or not patient_l:
        missing_list.append(f"No images found for patient {patient_path}. Skipping.")
        continue

    frame_size = cv2.imread(patient_r[0]).shape[1], cv2.imread(patient_r[0]).shape[0]

    writer = cv2.VideoWriter(
        f"{output_folder_path}/patient_{i:03d}.mp4",
        fourcc,
        fps,
        frame_size
    )

    all_frames = []
    all_frames += patient_r
    all_frames += patient_l

    for frame in all_frames:
        img = cv2.imread(frame)
        # mask = np.any(img > 0, axis=-1)
        # out = np.zeros_like(img)
        # out[mask] = [255, 255, 255]
        # writer.write(out)
        writer.write(img)

    writer.release()
    progress.update(1)

progress.close()

print("No images found for the following patients:")
for msg in missing_list:
    print(msg)

