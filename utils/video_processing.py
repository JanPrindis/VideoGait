import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
import cv2
import torch
import warnings
import numpy as np

from torch.nn import functional as F
from tqdm import tqdm
from utils.logger import log

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# old numpy compatibility (np.float -> float...)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    for alias, dtype in [('float', float), ('int', int), ('bool', bool), ('object', object), ('complex', complex)]:
        if not hasattr(np, alias):
            setattr(np, alias, dtype)


class FFmpegPipeWriter:
    """
    Writes video directly to FFmpeg subprocess via stdin pipe.
    Bypasses OpenCV VideoWriter completely to avoid DLL/Codec issues.
    """

    def __init__(self, filename, fps, width, height):
        """
        Initializes the FFmpeg pipe writer.

        Args:
            filename (str): Output video filename.
            fps (float): Frames per second.
            width (int): Video width.
            height (int): Video height.
        """
        self.filename = str(filename)
        self.width = int(width)
        self.height = int(height)

        # H264/AV1 requires even resolution value
        vf_filters = []
        target_w = self.width
        target_h = self.height

        if target_w % 2 != 0:
            target_w -= 1
            vf_filters.append(f"crop={target_w}:{self.height}:0:0")

        if target_h % 2 != 0:
            target_h -= 1
            vf_filters.append(f"crop={target_w}:{target_h}:0:0")

        vf_arg = ",".join(vf_filters)

        # Detect best available encoder
        encoder_config = _get_best_encoder_config()
        codec = encoder_config['codec']

        # Command
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{self.width}x{self.height}',
            '-pix_fmt', 'bgr24',  # OpenCV default format
            '-r', str(fps),
            '-i', '-',  # Input from STDIN (Pipe)

            '-c:v', codec,
            '-pix_fmt', 'yuv420p',  # Web compatibility
        ]

        if vf_arg:
            cmd.extend(['-vf', vf_arg])

        # Extend with encoder params
        cmd.extend(encoder_config['params'])
        cmd.append(self.filename)

        # Run
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,  # To prevent deadlocks
            stderr=subprocess.DEVNULL   # --------------------
        )
        self.active = True

    def write(self, frame):
        """
        Writes a single frame to the FFmpeg pipe.

        Args:
            frame (np.ndarray): Image frame (BGR format).
        """
        if not self.active: return
        try:
            self.process.stdin.write(frame.tobytes())   # Raw video
        except (IOError, BrokenPipeError) as e:
            self.release()
            raise RuntimeError(f"FFmpeg pipe broke while writing frame: {e}")

    def release(self):
        """Closes the pipe and waits for the process to finish."""
        if self.active:
            if self.process.stdin:
                self.process.stdin.close()
            self.process.wait()
            self.active = False

    def is_opened(self):
        """Checks if the FFmpeg process is still running."""
        return self.active and self.process.poll() is None


# Encoder cache
_ffmpeg_encoders = None


def _get_best_encoder_config():
    """
    Detects available FFmpeg encoders and returns the best configuration.

    Priority:
    1. AV1 NVENC (HW)
    2. AV1 SVT (SW)
    3. H264 NVENC (HW)
    4. H264 LIBX264 (SW)
    5. MPEG4 (Backup)

    Returns:
        dict: Dictionary containing 'codec' and 'params'.
    """
    global _ffmpeg_encoders
    if _ffmpeg_encoders is None:
        try:
            res = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True)
            _ffmpeg_encoders = res.stdout
        except:
            _ffmpeg_encoders = ""

    # 1. AV1 NVIDIA (Hardware)
    if 'av1_nvenc' in _ffmpeg_encoders:
        return {'codec': 'av1_nvenc', 'params': ['-rc', 'vbr', '-cq', '30', '-preset', 'p4']}

    # 2. AV1 Software (SVT-AV1)
    if 'libsvtav1' in _ffmpeg_encoders:
        return {'codec': 'libsvtav1', 'params': ['-crf', '35', '-preset', '8', '-g', '240']}

    # 3. H.264 NVIDIA (Hardware)
    if 'h264_nvenc' in _ffmpeg_encoders:
        return {'codec': 'h264_nvenc', 'params': ['-rc', 'vbr', '-cq', '23', '-preset', 'p4']}

    # 4. H.264 Software (libx264)
    if 'libx264' in _ffmpeg_encoders:
        return {'codec': 'libx264', 'params': ['-crf', '23', '-preset', 'fast']}

    # 5. MPEG-4 (Last Resort Backup)
    return {'codec': 'mpeg4', 'params': ['-q:v', '5']}


def create_video_writer(path, fps, width, height):
    """
    Factory function that returns an FFmpegPipeWriter instance.

    Args:
        path (str): Output video path.
        fps (float): Frames per second.
        width (int): Video width.
        height (int): Video height.

    Returns:
        FFmpegPipeWriter: The initialized writer.

    Raises:
        RuntimeError: If FFmpeg is not installed or found in PATH.
    """
    # Check if ffmpeg exists
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return FFmpegPipeWriter(path, fps, width, height)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("FFmpeg is missing! Please install FFmpeg and add it to PATH.")


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
    model_dir = PROJECT_ROOT / "rife" / "train_log"

    if not model_dir.exists():
        raise FileNotFoundError(f"RIFE model directory not found at: {model_dir}")

    model.load_model(str(model_dir), -1)
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

    writer = create_video_writer(out_path, fps, width, height)

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
        input_video (str): Path to input video.
        output_video (str): Path to output video.
        target_fps (int, optional): Target framerate. Defaults to 120.
    """

    # Get input video length for progress bar
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        log("VIDEO", f"Cannot open video for metadata: {input_video}", level="error")
        return

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Calculate expected frame number (InputFrames * (Target / Input))
    if input_fps <= 0: input_fps = 30  # Div by zero fallback
    expected_output_frames = int(input_frames * (target_fps / input_fps))

    # Get the best available encoder
    config = _get_best_encoder_config()

    # Command
    command = [
        'ffmpeg', '-y',
        '-i', input_video,
        '-vf', f'minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=obmc:me_mode=bidir',
        '-c:v', config['codec'],
    ]
    command.extend(config['params'])
    command.append(output_video)

    # Run and read output
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,  # Text mode
            encoding='utf-8',
            errors='replace'
        )

        # Regex for searching frame numbers "frame=  123"
        frame_pattern = re.compile(r"frame=\s*(\d+)")

        # TQDM Progress Bar
        pbar = tqdm(total=expected_output_frames, desc=f"Minterpolate {input_fps:.1f}->{target_fps}", unit="fr",
                    leave=False)

        # Read output and try to find frame numbers
        while True:
            line = process.stderr.readline()
            if not line and process.poll() is not None:
                break  # End

            if line:
                match = frame_pattern.search(line)
                if match:
                    current_frame = int(match.group(1))

                    # Update progress bar with current frame
                    pbar.n = current_frame
                    pbar.refresh()

        pbar.close()

        # Check return code
        if process.returncode != 0:
            err_msg = f"FFmpeg error in {input_video} (Code: {process.returncode})"
            log("VIDEO", err_msg, level="error")
            raise RuntimeError(err_msg)

    except Exception as e:
        log("VIDEO", f"Error interpolating {input_video}: {e}", level="error")
        raise


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
