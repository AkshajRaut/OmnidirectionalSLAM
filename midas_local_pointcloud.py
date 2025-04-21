import cv2
import torch
import numpy as np
import open3d as o3d
import sys
import os

# Add MiDaS path
sys.path.append("/home/ubuntu/Downloads/MiDaS")
from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose

# Load model
def load_midas_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DPTDepthModel(
        path=model_path,
        backbone="vitl16_384",
        non_negative=True
    )
    model.eval().to(device)

    transform = Compose([
        Resize(384, 384, resize_target=None, keep_aspect_ratio=True,
               ensure_multiple_of=32, resize_method="minimal", image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        PrepareForNet()
    ])

    return model, transform, device

# Depth estimation
def estimate_depth(img, model, transform, device):
    img_input = transform({"image": img})["image"]
    with torch.no_grad():
        sample = torch.from_numpy(img_input).unsqueeze(0).to(device)
        prediction = model.forward(sample)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()
        depth_map = prediction.cpu().numpy()
    return depth_map

# Convert depth to point cloud
def depth_to_point_cloud(depth, img, fx=500, fy=500, cx=None, cy=None):
    if cx is None: cx = depth.shape[1] / 2
    if cy is None: cy = depth.shape[0] / 2

    points = []
    colors = []

    for v in range(depth.shape[0]):
        for u in range(depth.shape[1]):
            z = depth[v, u]
            if z == 0: continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points.append([x, y, z])
            colors.append(img[v, u] / 255.0)

    return np.array(points), np.array(colors)

# Main process
def process_video(video_path, model, transform, device):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    all_points = []
    all_colors = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        print(f"[INFO] Processing frame {frame_idx}")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        depth = estimate_depth(frame_rgb, model, transform, device)
        points, colors = depth_to_point_cloud(depth, frame_rgb)
        all_points.append(points)
        all_colors.append(colors)

        frame_idx += 1
        if frame_idx > 30:  
            break

    cap.release()
    print("[INFO] Video processing complete.")
    
    # Combine all and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    model_path = "/home/ubuntu/Downloads/MiDaS/dpt_large-midas-2f21e586.pt"
    model, transform, device = load_midas_model(model_path)
    process_video("equirectangular.mp4", model, transform, device)