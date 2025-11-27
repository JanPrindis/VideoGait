import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Project-specific imports ---
from gaitDetectNN.models.gaitLSTM import GaitLSTM
from gaitDetectNN.inference.predictor import Predictor
from utils.preprocessing import generate_features
from Skeletons.halpe_skeleton import HALPE_SKELETON

# TODO: Load model and parameters from config file
# TODO: Separate validation and inference pipelines? - One accepting ground truth, other not
# TODO: This will be internal functions - will return gait-event structures
# TODO: Validation pipeline - measure accuracy of detection compared to ground truth - separate file?

# =============================================================================
# Configuration
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "../checkpoints/lstm_best.pth"

KEYPOINTS_USED = [
    "HIP", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX", "LEFT_SHOULDER", "RIGHT_SHOULDER"
]
KINEMATICS_USED = [
    "HIP", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
]
ANGLES_USED = [
    ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"), ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
    ("LEFT_ANKLE", "LEFT_HEEL", "LEFT_FOOT_INDEX"), ("RIGHT_ANKLE", "RIGHT_HEEL", "RIGHT_FOOT_INDEX"),
    ("LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"), ("RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE")
]
DISTANCES_USED = [
    ("HIP", "LEFT_HEEL"), ("HIP", "RIGHT_HEEL"), ("HIP", "LEFT_FOOT_INDEX"), ("HIP", "RIGHT_FOOT_INDEX"),
    ("LEFT_KNEE", "RIGHT_KNEE")
]

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# TEST_VIDEO_KP_PATH = os.path.join(PROJECT_ROOT, "dataset/PROCESSED/60/KEYPOINTS/PD_006_MD.json")
TEST_VIDEO_KP_PATH = os.path.join(PROJECT_ROOT, "results/test.json")

# TEST_VIDEO_ANN_PATH = os.path.join(PROJECT_ROOT, "annotations/60/PD_006_MD.json")
TEST_VIDEO_ANN_PATH = os.path.join(PROJECT_ROOT, "results/result.json")
# TEST_VIDEO_ANN_PATH = None

# =============================================================================
# Prepare Data
# =============================================================================
print(f"Preparing data for {os.path.basename(TEST_VIDEO_KP_PATH)}...")

# Define the arguments for our unified feature generator
preprocess_args = {
    "skeleton_definition": HALPE_SKELETON,
    "confidence_threshold": 0.5,
    "exclude_ratio": 0.1,
    "keypoints": KEYPOINTS_USED,
    "kinematics_keypoints": KINEMATICS_USED,
    "angle_triplets": ANGLES_USED,
    "distance_pairs": DISTANCES_USED
}

# If you have annotations, the function will use them for FPS and labels.
# Otherwise, you must provide the frame rate manually.
if TEST_VIDEO_ANN_PATH and os.path.exists(TEST_VIDEO_ANN_PATH):
    feature_matrices, label_matrices, global_ranges = generate_features(
        keypoints_path=TEST_VIDEO_KP_PATH,
        annotations_path=TEST_VIDEO_ANN_PATH,
        **preprocess_args
    )
else:
    # For a new, un-annotated video, you must provide the frame rate.
    # Example of getting it dynamically:
    # cap = cv2.VideoCapture("path/to/original/video.mp4")
    # video_fps = cap.get(cv2.CAP_PROP_FPS)
    # cap.release()
    video_fps = 60  # Manually setting for this example
    feature_matrices, label_matrices, global_ranges = generate_features(
        keypoints_path=TEST_VIDEO_KP_PATH,
        frame_rate=video_fps,  # Pass FPS directly
        **preprocess_args
    )

if not feature_matrices:
    raise ValueError("Preprocessing did not produce any valid clips from the video.")

# --- Dynamically get the number of features ---
NUM_FEATURES = feature_matrices[0].shape[1]
print(f"Model expects {NUM_FEATURES} features.")

# =============================================================================
# Load Model and Create Predictor
# =============================================================================
print("Loading model...")
# First, instantiate the model architecture with the correct number of features
model = GaitLSTM(input_size=NUM_FEATURES)

# Then, load the learned weights from your checkpoint file
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

# Now, create the generic predictor with your loaded model
predictor = Predictor(model, DEVICE)

# =============================================================================
# Run Inference and Visualize for Each Clip
# =============================================================================
total_frames = 0
if global_ranges:
    total_frames = max(end for start, end in global_ranges) + 1

# Initialize empty arrays to hold all predictions and labels
all_predictions = np.zeros((total_frames, 4))
all_labels = np.zeros((total_frames, 4)) if label_matrices else None


# Run inference and populate the master arrays
for idx, test_clip_features in enumerate(feature_matrices):
    print(f"\n--- Running inference on Clip {idx + 1}/{len(feature_matrices)} ---")
    predicted_probs = predictor.predict_on_clip(test_clip_features)

    # Get the global frame range for this clip
    start_frame, end_frame = global_ranges[idx]

    # Place the predictions into the master array at the correct global timeline
    all_predictions[start_frame: end_frame + 1] = predicted_probs

    # Do the same for labels if they exist
    if all_labels is not None:
        all_labels[start_frame: end_frame + 1] = label_matrices[idx]

# TODO: Maybe keep as a debug visualization (like verbose param or something)
# Now, plot the combined results
print("\nPlotting combined results for the entire video...")
fig, axs = plt.subplots(4, 1, figsize=(18, 10), sharex=True)
fig.suptitle(f"Full Video Predictions: {os.path.basename(TEST_VIDEO_KP_PATH)}", fontsize=16)
event_names = ["Left Heel Strike", "Left Toe Off", "Right Heel Strike", "Right Toe Off"]
threshold = 0.5

for i in range(4):
    # Plot the master predictions array
    axs[i].plot(all_predictions[:, i], label='Model Prediction', color='red', linestyle='--')

    # If we have ground truth, plot that too
    if all_labels is not None:
        axs[i].plot(all_labels[:, i], label='Ground Truth', color='blue', linewidth=2.5, alpha=0.8)

    axs[i].axhline(y=threshold, color='gray', linestyle=':', label=f'Threshold ({threshold})')
    axs[i].set_title(event_names[i], fontsize=14)
    axs[i].set_ylim(-0.1, 1.1)
    axs[i].legend()
    axs[i].grid(True, which='both', linestyle='--', linewidth=0.5)

plt.xlabel("Frame Number in Original Video", fontsize=12)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
