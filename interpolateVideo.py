import os
import sys
import cv2
import torch
import numpy as np
import argparse
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
def read_video_frames(video_path):
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
def pad_image(img, padding, fp16=False):
    img = F.pad(img, padding)
    return img.half() if fp16 else img

# Frame interpolation
def make_inference(model, I0, I1, n, scale):
    if n == 1:
        mid = model.inference(I0, I1, scale)
        return [mid]
    middle = model.inference(I0, I1, scale)
    first_half = make_inference(model, I0, middle, n=n//2, scale=scale)
    second_half = make_inference(model, middle, I1, n=n//2, scale=scale)
    if n % 2:
        return [*first_half, middle, *second_half]
    else:
        return [*first_half, *second_half]

# MAIN
def main():
    parser = argparse.ArgumentParser(description="Video interpolation using RIFE HDv3")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--exp", type=int, default=1)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--ext", type=str, default="mp4")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if args.fp16:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

    # Load model
    from rife.train_log.RIFE_HDv3 import Model
    model = Model()
    model.load_model("rife/train_log", -1)
    model.eval()
    model.device()
    print("Loaded RIFE HDv3 model.")

    # Video info
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {args.video}")
    fps_orig = cap.get(cv2.CAP_PROP_FPS)
    tot_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if args.fps is None:
        args.fps = fps_orig * (2 ** args.exp)

    print(f"Video: {args.video} | {tot_frame} frames | {fps_orig:.2f} FPS → {args.fps:.2f} FPS | Resolution: {width}x{height}")

    # Video writer
    out_path = args.output if args.output else os.path.splitext(args.video)[0] + f"_interp.{args.ext}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, args.fps, (width, height))

    # Padding
    tmp = max(32, int(32 / args.scale))
    ph = ((height - 1) // tmp + 1) * tmp
    pw = ((width - 1) // tmp + 1) * tmp
    padding = (0, pw - width, 0, ph - height)

    # Video generator
    videogen = read_video_frames(args.video)
    last_frame_np = next(videogen)
    last_frame = torch.from_numpy(np.transpose(last_frame_np, (2,0,1))).to(device).unsqueeze(0).float() / 255.
    last_frame = pad_image(last_frame, padding, args.fp16)

    pbar = tqdm(total=tot_frame-1)

    # Frame processing loop
    for frame_np in videogen:
        frame = torch.from_numpy(np.transpose(frame_np, (2,0,1))).to(device).unsqueeze(0).float() / 255.
        frame = pad_image(frame, padding, args.fp16)

        # Optional: skip almost identical frames
        I0_small = F.interpolate(last_frame, (32,32), mode='bilinear', align_corners=False)
        I1_small = F.interpolate(frame, (32,32), mode='bilinear', align_corners=False)
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

        if ssim > 0.996:
            interp_frames = []
        else:
            interp_frames = make_inference(model, last_frame, frame, n=(2**args.exp-1), scale=args.scale)

        # Write last frame
        out_frame = (last_frame[0] * 255).byte().cpu().numpy().transpose(1,2,0)[:height, :width]
        writer.write(cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR))

        # Write interpolated frames
        for mid in interp_frames:
            mid_np = (mid[0] * 255).byte().cpu().numpy().transpose(1,2,0)[:height, :width]
            writer.write(cv2.cvtColor(mid_np, cv2.COLOR_RGB2BGR))

        last_frame = frame
        pbar.update(1)

    # Write last frame again
    out_frame = (last_frame[0] * 255).byte().cpu().numpy().transpose(1,2,0)[:height, :width]
    writer.write(cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR))

    writer.release()
    pbar.close()
    print(f"Finished. Output saved to {out_path}")

if __name__ == "__main__":
    main()
