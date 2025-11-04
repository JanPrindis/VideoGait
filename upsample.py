import os
import sys
import cv2
import torch
import numpy as np

from matplotlib.pyplot import title
from torch.nn import functional as F
from tqdm import tqdm
from rife.model.pytorch_msssim import ssim_matlab

# To fix rife imports, because it is no longer project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rife"))

# old numpy compatibility (np.float -> float...)
for alias, dtype in [('float', float), ('int', int), ('bool', bool), ('object', object), ('complex', complex)]:
    if not hasattr(np, alias):
        setattr(np, alias, dtype)

# Video reader replacement
def _read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()

# Padding helper
def _pad_image(img, padding, fp16=False):
    img = F.pad(img, padding)
    return img.half() if fp16 else img

# Frame interpolation
def _make_inference(model, I0, I1, n, scale):
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
    Video interpolation using RIFE HDv3

    Parameters
    ----------
    video : str
        Path to input video
    output : str, optional
        Output file path, default: None (auto based on input)
    exp : int, optional
        Interpolation exponent, default: 1
    fps : float, optional
        Override output FPS, default: None
    scale : float, optional
        Image scaling factor, default: 1.0
    fp16 : bool, optional
        Enable FP16 inference, default: False
    ext : str, optional
        Output file extension, default: "mp4"
    """

    # If output isn't provided, derive from input
    if output is None:
        import os
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

        # Skip almost identical frames
        I0_small = F.interpolate(last_frame, (32,32), mode='bilinear', align_corners=False)
        I1_small = F.interpolate(frame, (32,32), mode='bilinear', align_corners=False)
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

        if ssim > 0.996:
            interp_frames = []
        else:
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

