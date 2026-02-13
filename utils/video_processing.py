import math
import os
import shutil
import subprocess
import sys
import cv2
import torch
import warnings
import numpy as np

from matplotlib.pyplot import title
from torch.nn import functional as F
from tqdm import tqdm
from utils.logger import log

# old numpy compatibility (np.float -> float...)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    for alias, dtype in [('float', float), ('int', int), ('bool', bool), ('object', object), ('complex', complex)]:
        if not hasattr(np, alias):
            setattr(np, alias, dtype)


# Video reader replacement
def _read_video_frames(video_path):
    """
    Generator that yields frames from a video file.

    Args:
        video_path (str): Path to the video file.

    Yields:
        np.ndarray: Video frame in RGB format.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cap.release()


def _pad_image(img, padding, fp16=False):
    """
    Pads the image tensor.

    Args:
        img (torch.Tensor): Input image tensor.
        padding (tuple): Padding values (left, right, top, bottom).
        fp16 (bool): Whether to convert to half precision.

    Returns:
        torch.Tensor: Padded image tensor.
    """
    img = F.pad(img, padding)
    return img.half() if fp16 else img


def _make_inference(model, I0, I1, n, scale):
    """
    Recursively interpolates frames between I0 and I1.

    Args:
        model: RIFE model instance.
        I0 (torch.Tensor): Start frame.
        I1 (torch.Tensor): End frame.
        n (int): Number of intermediate frames to generate.
        scale (float): Scaling factor.

    Returns:
        list[torch.Tensor]: List of interpolated frames.
    """
    if n == 1:
        mid = model.inference(I0, I1, scale)
        return [mid]

    middle = model.inference(I0, I1, scale)
    first_half = _make_inference(model, I0, middle, n=n // 2, scale=scale)
    second_half = _make_inference(model, middle, I1, n=n // 2, scale=scale)

    if n % 2:
        return [*first_half, middle, *second_half]
    else:
        return [*first_half, *second_half]


def RIFE_interpolate(
    video: str,
    output: str = None,
    exp: int = 1,
    fps: float = None,
    scale: float = 1.0,
    fp16: bool = False,
    ext: str = "mp4"
):
    """
    Video interpolation using RIFE HDv3.

    Args:
        video (str): Path to input video.
        output (str, optional): Output file path. Defaults to None (auto based on input).
        exp (int, optional): Interpolation exponent (2^exp frames). Defaults to 1.
        fps (float, optional): Override output FPS. Defaults to None.
        scale (float, optional): Image scaling factor. Defaults to 1.0.
        fp16 (bool, optional): Enable FP16 inference. Defaults to False.
        ext (str, optional): Output file extension. Defaults to "mp4".
    """

    # If output isn't provided, derive from input
    if output is None:
        base, _ = os.path.splitext(video)
        output = f"{base}_RIFE.{ext}"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if fp16:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

    # Load model
    from rife.train_log.RIFE_HDv3 import Model
    model = Model()
    model.load_model("rife/train_log", -1)
    model.eval()
    model.device()

    # Video info
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video}")
    fps_orig = cap.get(cv2.CAP_PROP_FPS)
    tot_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps is None:
        fps = fps_orig * (2 ** exp)

    # print(f"Video: {args.video} | {tot_frame} frames | {fps_orig:.2f} FPS -> {args.fps:.2f} FPS | Resolution: {width}x{height}")

    # Video writer
    out_path = output if output else os.path.splitext(video)[0] + f"_interp.{ext}"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Padding
    tmp = max(32, int(32 / scale))
    ph = ((height - 1) // tmp + 1) * tmp
    pw = ((width - 1) // tmp + 1) * tmp
    padding = (0, pw - width, 0, ph - height)

    # Video generator
    videogen = _read_video_frames(video)
    last_frame_np = next(videogen)
    last_frame = torch.from_numpy(np.transpose(last_frame_np, (2,0,1))).to(device).unsqueeze(0).float() / 255.
    last_frame = _pad_image(last_frame, padding, fp16)

    pbar = tqdm(desc=f"{fps_orig} -> {fps}", total=tot_frame-1, leave=False)

    # Frame processing loop
    for frame_np in videogen:
        frame = torch.from_numpy(np.transpose(frame_np, (2,0,1))).to(device).unsqueeze(0).float() / 255.
        frame = _pad_image(frame, padding, fp16)
        interp_frames = _make_inference(model, last_frame, frame, n=(2 ** exp - 1), scale=scale)

        # Write frame
        out_frame = (last_frame[0] * 255).byte().cpu().numpy().transpose(1,2,0)[:height, :width]
        writer.write(cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR))

        # Write interpolated frames
        for mid in interp_frames:
            mid_np = (mid[0] * 255).byte().cpu().numpy().transpose(1,2,0)[:height, :width]
            writer.write(cv2.cvtColor(mid_np, cv2.COLOR_RGB2BGR))

        last_frame = frame
        pbar.update(1)

    # Write last frame
    out_frame = (last_frame[0] * 255).byte().cpu().numpy().transpose(1,2,0)[:height, :width]
    writer.write(cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR))

    writer.release()
    pbar.close()
    #print(f"Finished. Output saved to {out_path}")


def interpolate_minterpolate(input_video, output_video, target_fps=120):
    """
    Interpolates a video to a target FPS using FFmpeg's minterpolate filter.

    Args:
        input_video (str): Path to the input video file.
        output_video (str): Path to the output video file.
        target_fps (int): The desired output FPS.
    """
    command = [
        'ffmpeg',
        '-i', input_video,
        '-vf', f'minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir',
        '-c:v', 'libx264',
        '-crf', '18',
        '-preset', 'slow',
        output_video
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        log("VIDEO", f"Error interpolating {input_video}:\n{e.stderr}", level="error")


def get_video_fps(video_path):
    """
    Retrieves the framerate of a video file.

    Args:
        video_path (str): Path to the video file.

    Returns:
        float: The framerate (FPS).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def smart_interpolate(input_path, output_path, target_fps):
    """
    Intelligently interpolates a video to a target FPS using a hybrid strategy.

    It attempts to use RIFE (Deep Learning) for 2^n interpolation factors where possible,
    and falls back to FFmpeg's minterpolate for other ratios or as a pre-step.

    Args:
        input_path (str): Path to input video.
        output_path (str): Path to output video.
        target_fps (float): Desired output framerate.
    """
    orig_fps = get_video_fps(input_path)

    # Float error tolerance
    EPS = 0.1

    log("SMART_INTERP", f"Input: {orig_fps:.2f} FPS | Target: {target_fps} FPS", level="info")

    # SAME FPS -> COPY
    if abs(orig_fps - target_fps) < EPS:
        log("SMART_INTERP", "FPS match, copying file.", level="info")
        shutil.copy(input_path, output_path)
        return

    # Is RIFE able to reach target FPS? (target = source * 2^N)
    def is_rife_compatible(src, tgt):
        if src > tgt: return False  # We cannot downsample
        ratio = tgt / src
        log_ratio = math.log2(ratio)
        return abs(log_ratio - round(log_ratio)) < EPS

    # DIRECT RIFE
    if is_rife_compatible(orig_fps, target_fps):
        exp = int(round(math.log2(target_fps / orig_fps)))
        log("SMART_INTERP", f"Direct RIFE compatible (2^{exp}x). Executing...", level="info")
        RIFE_interpolate(video=input_path, output=output_path, exp=exp, fps=target_fps)
        return

    # HYBRID PATH (minterpolate -> RIFE)
    # Base FPS values where RIFE can take over
    rife_bases = [30.0, 60.0, 120.0]

    best_base = None
    min_diff = float('inf')

    for base in rife_bases:
        # Check if base is target or RIFE can reach the target FPS
        if abs(base - target_fps) < EPS or is_rife_compatible(base, target_fps):

            # Find base with the smallest difference
            diff = abs(orig_fps - base)

            if diff < min_diff:
                min_diff = diff
                best_base = base

    if best_base is not None:
        log("SMART_INTERP", f"Hybrid Strategy: {orig_fps} -> minterpolate({best_base}) -> RIFE({target_fps})", level="info")

        # Temp file path
        temp_file = output_path.replace(".mp4", f"_temp_{int(best_base)}.mp4")

        try:
            # Minterpolate to the closest base
            if abs(orig_fps - best_base) > EPS:
                log("SMART_INTERP", f"  [Step 1] Minterpolate to {best_base} FPS...", level="info")
                interpolate_minterpolate(input_path, temp_file, target_fps=int(best_base))
                current_input = temp_file
            else:
                current_input = input_path

            # RIFE to target
            if abs(best_base - target_fps) > EPS:
                exp = int(round(math.log2(target_fps / best_base)))
                log("SMART_INTERP", f"  [Step 2] RIFE 2^{exp}x to {target_fps} FPS...", level="info")
                RIFE_interpolate(video=current_input, output=output_path, exp=exp, fps=target_fps)
            else:
                # If base is our target framerate, rename temp to final
                if os.path.exists(temp_file):
                    shutil.move(temp_file, output_path)
                else:
                    shutil.copy(input_path, output_path)

        finally:
            # Cleanup
            if os.path.exists(temp_file):
                os.remove(temp_file)
        return

    # FALLBACK
    log("SMART_INTERP", f"No clean path found. Brute-forcing minterpolate to {target_fps}.", level="warning")
    interpolate_minterpolate(input_path, output_path, target_fps=target_fps)
