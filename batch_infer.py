#!/usr/bin/env python3
"""
Batch inference script for LingBot-Depth.
Processes all recordings in mcap/pull/ with multi-GPU, multi-worker parallelism.

Usage:
    python batch_infer.py --data_root /home/zoyo/mcap/pull --num_workers_per_gpu 4
"""

import os
import argparse
import time
from pathlib import Path
import multiprocessing as mp
from multiprocessing import Process, Queue

import numpy as np
import cv2
import h5py


def discover_tasks(data_root: str):
    """Discover all (recording_dir, camera, num_frames) tasks."""
    data_root = Path(data_root)
    tasks = []
    for rec_dir in sorted(data_root.iterdir()):
        if not rec_dir.is_dir():
            continue
        rgb_dir = rec_dir / "rgb"
        depth_dir = rec_dir / "depth"
        h5_path = rec_dir / "trajectory_valid.h5"
        if not (rgb_dir.exists() and depth_dir.exists() and h5_path.exists()):
            continue
        for cam in sorted(os.listdir(rgb_dir)):
            cam_rgb = rgb_dir / cam
            cam_depth = depth_dir / cam
            if not (cam_rgb.is_dir() and cam_depth.is_dir()):
                continue
            frames = sorted([f for f in os.listdir(cam_rgb) if f.endswith(('.jpg', '.png'))])
            if len(frames) == 0:
                continue
            tasks.append((str(rec_dir), cam, len(frames)))
    return tasks


def worker_fn(physical_gpu_id: int, task_queue: Queue, model_path: str, depth_scale: float):
    """Worker process: restrict to one physical GPU via env var, load model on cuda:0."""
    # Each worker only sees its assigned GPU as cuda:0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)

    import torch
    from mdm.model.v2 import MDMModel

    device = torch.device("cuda:0")
    pid = os.getpid()
    print(f"[Worker pid={pid} phys_GPU={physical_gpu_id}] Loading model...", flush=True)
    model = MDMModel.from_pretrained(model_path).to(device)
    print(f"[Worker pid={pid} phys_GPU={physical_gpu_id}] Model loaded.", flush=True)

    while True:
        item = task_queue.get()
        if item is None:
            break

        rec_dir, cam_name, num_frames = item
        rec_name = Path(rec_dir).name
        print(f"[Worker pid={pid} phys_GPU={physical_gpu_id}] Processing {rec_name}/{cam_name} ({num_frames} frames)", flush=True)
        t0 = time.time()

        rgb_dir = Path(rec_dir) / "rgb" / cam_name
        depth_dir = Path(rec_dir) / "depth" / cam_name
        h5_path = Path(rec_dir) / "trajectory_valid.h5"
        out_dir = Path(rec_dir) / "depth_lingbot" / cam_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load intrinsics once (constant across frames)
        with h5py.File(str(h5_path), "r") as f:
            intrinsics_raw = f[f"observation/camera/intrinsics/{cam_name}"][0].copy()

        rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.jpg', '.png'))])

        for frame_file in rgb_files:
            frame_stem = Path(frame_file).stem
            out_path = out_dir / f"{frame_stem}.npy"
            if out_path.exists():
                continue

            img_bgr = cv2.imread(str(rgb_dir / frame_file))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            image = torch.tensor(img_rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)

            depth_path = depth_dir / f"{frame_stem}.npy"
            if not depth_path.exists():
                continue
            depth_raw = np.load(str(depth_path)).astype(np.float32) / depth_scale
            depth = torch.tensor(depth_raw, dtype=torch.float32, device=device).unsqueeze(0)

            intr = intrinsics_raw.copy()
            intr[0] /= w
            intr[1] /= h
            intrinsics = torch.tensor(intr, dtype=torch.float32, device=device).unsqueeze(0)

            output = model.infer(image, depth_in=depth, intrinsics=intrinsics)
            depth_refined = output["depth"].squeeze(0).cpu().numpy()
            np.save(str(out_path), depth_refined.astype(np.float32))

        elapsed = time.time() - t0
        fps = num_frames / elapsed if elapsed > 0 else 0
        print(f"[Worker pid={pid} phys_GPU={physical_gpu_id}] Done {rec_name}/{cam_name}: {num_frames} frames in {elapsed:.1f}s ({fps:.1f} fps)", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Batch LingBot-Depth inference")
    parser.add_argument("--data_root", type=str, default="/home/zoyo/mcap/pull")
    parser.add_argument("--model_path", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "checkpoints/lingbot-depth-v0.5/model.pt"))
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs (default: all)")
    parser.add_argument("--num_workers_per_gpu", type=int, default=4, help="Workers per GPU")
    parser.add_argument("--depth_scale", type=float, default=1000.0, help="raw_value / scale = meters")
    args = parser.parse_args()

    import torch
    if args.num_gpus is None:
        args.num_gpus = torch.cuda.device_count()
    total_workers = args.num_gpus * args.num_workers_per_gpu

    print(f"GPUs: {args.num_gpus}, Workers/GPU: {args.num_workers_per_gpu}, Total workers: {total_workers}")

    tasks = discover_tasks(args.data_root)
    total_frames = sum(t[2] for t in tasks)
    print(f"Found {len(tasks)} recording-camera pairs, {total_frames} total frames")

    if len(tasks) == 0:
        print("No tasks found, exiting.")
        return

    task_queue = Queue()
    for t in tasks:
        task_queue.put(t)
    for _ in range(total_workers):
        task_queue.put(None)

    t0 = time.time()
    workers = []
    for gpu_id in range(args.num_gpus):
        for _ in range(args.num_workers_per_gpu):
            p = Process(target=worker_fn, args=(gpu_id, task_queue, args.model_path, args.depth_scale))
            p.start()
            workers.append(p)

    for p in workers:
        p.join()

    elapsed = time.time() - t0
    print(f"\nAll done! {total_frames} frames processed in {elapsed:.1f}s ({total_frames/elapsed:.1f} fps overall)")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
