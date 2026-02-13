import os
import argparse
import sys
import platform
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch

# Silence Deprecated warnings
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

current_dir = Path(__file__).parent.resolve()
if str(current_dir) in sys.path:
    sys.path.remove(str(current_dir))
sys.path.insert(0, str(current_dir))

from detectors import POSE_DETECTORS
from ..base import BaseDetector
from utils.json_serializer import KeypointSerializer
from utils.logger import log

# --- ALPHAPOSE IMPORTS ---
# Hack to handle 'utils' name collision between project root and AlphaPose
# We temporarily remove the project's 'utils' from sys.modules so AlphaPose can load its own.
utils_modules = {}
for k in list(sys.modules.keys()):
    if k == "utils" or k.startswith("utils."):
        utils_modules[k] = sys.modules.pop(k)

try:
    # Pre-load AlphaPose's internal utils to prevent it from finding the project root 'utils'
    # We map 'utils' -> 'trackers.utils' to satisfy imports in tracker_api.py
    try:
        import trackers.utils.utils
        sys.modules["utils"] = sys.modules["trackers.utils"]
        sys.modules["utils.utils"] = sys.modules["trackers.utils.utils"]
    except ImportError:
        # If we can't find the internal utils, we proceed and hope for the best
        pass

    from alphapose.models import builder
    from alphapose.utils.config import update_config
    from alphapose.utils.detector import DetectionLoader
    from alphapose.utils.transforms import flip, flip_heatmap, get_func_heatmap_to_coord
    from trackers.tracker_api import Tracker
    from trackers.tracker_cfg import cfg as tcfg
    from trackers import track
except ImportError as e:
    log("AlphaPose", f"Import failed: {e}", level="error")
    raise e
finally:
    # Restore project utils to sys.modules
    sys.modules.update(utils_modules)


@POSE_DETECTORS.register
class AlphaPose(BaseDetector):
    def __init__(self, config: dict):
        super().__init__(config)

        # SETUP PATHS
        self.base_dir = Path(__file__).parent.resolve()
        cfg_name = self.config.get('config_file', 'halpe_26_fast_res50_256x192.yaml')
        ckpt_name = self.config.get('checkpoint_file', 'halpe26_fast_res50_256x192.pth')

        self.cfg_path = self.base_dir / "configs" / cfg_name
        self.checkpoint_path = self.base_dir / "pretrained_models" / ckpt_name

        log("AlphaPose", f"Config: {self.cfg_path}", level="info")
        log("AlphaPose", f"Checkpoint: {self.checkpoint_path}", level="info")

        # LOAD CONFIG
        self.ap_cfg = update_config(str(self.cfg_path))

        # HARDWARE & PARAMS
        gpus = self.config.get('gpus', '0')
        self.gpus = [int(i) for i in gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.posebatch = self.config.get('posebatch', 64)
        self.detbatch = self.config.get('detbatch', 5)
        self.detector_name = self.config.get('detector', 'yolox')
        self.qsize = self.config.get('qsize', 1024)

        # Windows fix
        self.sp = (platform.system() == 'Windows')
        if not self.sp:
            try:
                torch.multiprocessing.set_start_method('forkserver', force=True)
                torch.multiprocessing.set_sharing_strategy('file_system')
            except RuntimeError:
                pass

        # INIT TRANSFORMS (from writer.py)
        self.heatmap_to_coord = get_func_heatmap_to_coord(self.ap_cfg)
        self.norm_type = self.ap_cfg.LOSS.get('NORM_TYPE', None)
        self.hm_size = self.ap_cfg.DATA_PRESET.HEATMAP_SIZE

        # Determine Eval Joints (from writer.py update())
        self.eval_joints = list(range(26))  # Default for Halpe26

    def detect(self, video_path, output_path):
        v_path = Path(video_path).resolve()
        save_dir = Path(output_path).resolve()

        save_dir.mkdir(parents=True, exist_ok=True)
        json_filename = f"{v_path.stem}.json"
        serializer = KeypointSerializer(str(save_dir), json_filename)

        original_cwd = os.getcwd()
        os.chdir(str(self.base_dir))

        det_loader = None

        try:
            # ARGS SETUP
            args = argparse.Namespace(
                video=str(v_path),
                cfg=str(self.cfg_path),
                checkpoint=str(self.checkpoint_path),
                gpus=self.gpus,
                device=self.device,
                posebatch=self.posebatch,
                detbatch=self.detbatch,
                sp=self.sp,
                detector=self.detector_name,
                qsize=self.qsize,
                save_video=False,
                vis=False, vis_fast=False, save_img=False, pose_flow=False,
                pose_track=True,
                tracking=True,
                format='', eval=False, flip=True, debug=False,
                min_box_area=0
            )

            # INIT MODULES
            from detector.apis import get_detector
            det_loader = DetectionLoader(
                input_source=str(v_path),
                detector=get_detector(args),
                cfg=self.ap_cfg,
                opt=args,
                batchSize=args.detbatch,
                mode='video',
                queueSize=args.qsize
            )
            det_loader.start()

            pose_model = builder.build_sppe(self.ap_cfg.MODEL, preset_cfg=self.ap_cfg.DATA_PRESET)
            pose_model.load_state_dict(torch.load(str(self.checkpoint_path), map_location=self.device))
            pose_model.to(self.device)
            pose_model.eval()

            tracker = Tracker(tcfg, args)
            pose_dataset = builder.retrieve_dataset(self.ap_cfg.DATASET.TRAIN)

            data_len = det_loader.length
            pbar = tqdm(total=data_len, desc=f"AlphaPose: {v_path.name}", unit="frame")
            frame_idx = 0

            # MAIN LOOP
            for i in range(data_len):
                with torch.no_grad():
                    (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()

                    if orig_img is None:
                        break

                    if boxes is None or boxes.nelement() == 0:
                        pbar.update(1)
                        frame_idx += 1
                        continue

                    # Pose Inference
                    inps = inps.to(self.device)
                    datalen = inps.size(0)
                    leftover = 0
                    if (datalen) % self.posebatch:
                        leftover = 1
                    num_batches = datalen // self.posebatch + leftover

                    hm = []
                    for j in range(num_batches):
                        inps_j = inps[j * self.posebatch: min((j + 1) * self.posebatch, datalen)]
                        if args.flip:
                            inps_j = torch.cat((inps_j, flip(inps_j)))

                        hm_j = pose_model(inps_j)

                        if args.flip:
                            hm_j_flip = flip_heatmap(hm_j[int(len(hm_j) / 2):], pose_dataset.joint_pairs, shift=True)
                            hm_j = (hm_j[0:int(len(hm_j) / 2)] + hm_j_flip) / 2

                        hm.append(hm_j)

                    hm = torch.cat(hm)

                    # --- TRACKING ---
                    boxes, scores, ids, hm, cropped_boxes = track(
                        tracker, args, orig_img, inps, boxes, hm, cropped_boxes, im_name, scores
                    )

                    # --- PROCESSING (from writer.py) ---
                    hm = hm.cpu()

                    # Update Eval Joints based on heatmap shape (from writer.py)
                    num_hm_joints = hm.size(1)
                    if num_hm_joints == 136:
                        self.eval_joints = list(range(136))
                    elif num_hm_joints == 26:
                        self.eval_joints = list(range(26))
                    elif num_hm_joints == 133:
                        self.eval_joints = list(range(133))
                    elif num_hm_joints == 68:
                        self.eval_joints = list(range(68))
                    elif num_hm_joints == 21:
                        self.eval_joints = list(range(21))

                    pose_coords = []
                    pose_scores = []

                    for k in range(hm.shape[0]):
                        bbox = cropped_boxes[k].tolist()

                        # Calling heatmap_to_coord exactly as in writer.py
                        pose_coord, pose_score = self.heatmap_to_coord(
                            hm[k][self.eval_joints],
                            bbox,
                            hm_shape=self.hm_size,
                            norm_type=self.norm_type
                        )

                        pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
                        pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))

                    # Stack results
                    preds_img = torch.cat(pose_coords)
                    preds_scores = torch.cat(pose_scores)

                    # --- SCORING & SELECTION ---
                    # We use writer.py's scoring logic:

                    valid_people = []
                    for k in range(len(scores)):
                        # Scores from Keypoint estimation (N_Joints, 1)
                        kp_score = preds_scores[k]
                        # Score from Detector/Tracker
                        box_score = scores[k].item()

                        # Official AlphaPose Proposal Score Formula:
                        # mean(kp_score) + box_score + 1.25 * max(kp_score)
                        proposal_score = torch.mean(kp_score) + box_score + 1.25 * torch.max(kp_score)
                        proposal_score = proposal_score.item()

                        valid_people.append({
                            'kps': preds_img[k].numpy(),
                            'scs': preds_scores[k].numpy(),
                            'rank': proposal_score
                        })

                    # Select best person
                    if valid_people:
                        best_person = max(valid_people, key=lambda x: x['rank'])

                        final_kps = best_person['kps']
                        final_scs = best_person['scs']

                        flat_kps = []
                        for j in range(len(final_kps)):
                            # We keep raw confidence for analysis.py to decide.
                            flat_kps.extend([
                                float(final_kps[j][0]),
                                float(final_kps[j][1]),
                                float(final_scs[j][0])
                            ])

                        serializer.add_frame(frame_idx, flat_kps)

                    pbar.update(1)
                    frame_idx += 1

            pbar.close()

        except Exception as e:
            log("AlphaPose", f"Critical Error: {e}", level="error")
            import traceback
            traceback.print_exc()
        finally:
            if det_loader:
                det_loader.stop()
                if not self.sp:
                    det_loader.terminate()

            serializer.save()
            os.chdir(original_cwd)
