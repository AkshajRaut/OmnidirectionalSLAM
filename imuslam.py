import cv2
import numpy as np
import pandas as pd
import open3d as o3d

# Load IMU data
imu_df = pd.read_csv("imu_20250320_155131_11_003.csv")

def quaternion_to_rotation_matrix(w, x, y, z):
    R = np.array([
        [1 - 2*y**2 - 2*z**2,     2*x*y - 2*z*w,         2*x*z + 2*y*w],
        [2*x*y + 2*z*w,           1 - 2*x**2 - 2*z**2,   2*y*z - 2*x*w],
        [2*x*z - 2*y*w,           2*y*z + 2*x*w,         1 - 2*x**2 - 2*y**2]
    ])
    return R

def visualize_trajectory(trajectory):
    traj_pts = np.array(trajectory)
    traj_pcd = o3d.geometry.PointCloud()
    traj_pcd.points = o3d.utility.Vector3dVector(traj_pts)
    traj_pcd.paint_uniform_color([1, 0, 0])
    
    line_set = o3d.geometry.LineSet()
    if len(traj_pts) >= 2:
        lines = [[i, i+1] for i in range(len(traj_pts) - 1)]
        line_set.points = o3d.utility.Vector3dVector(traj_pts)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in lines])
    
    o3d.visualization.draw_geometries([traj_pcd, line_set])

def run_slam_with_imu(video_path, video_fps=30):
    cap = cv2.VideoCapture(video_path)
    orb = cv2.ORB_create(2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    pose = np.eye(4)
    trajectory = [pose[:3, 3].tolist()]
    prev_kp, prev_des, prev_frame = None, None, None
    frame_idx = 0
    frame_interval_ms = 1000 / video_fps

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)

        if prev_des is not None:
            matches = bf.match(prev_des, des)
            matches = sorted(matches, key=lambda x: x.distance)
            if len(matches) > 8:
                pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches])
                pts2 = np.float32([kp[m.trainIdx].pt for m in matches])
                E, mask = cv2.findEssentialMat(pts1, pts2, method=cv2.RANSAC, threshold=1.0)
                if E is not None:
                    _, R_vo, t, _ = cv2.recoverPose(E, pts1, pts2)
                    T_vo = np.eye(4)
                    T_vo[:3, :3] = R_vo
                    T_vo[:3, 3] = t.squeeze()

                    # IMU FUSION START
                    frame_timestamp = frame_idx * frame_interval_ms
                    imu_row = imu_df.iloc[(imu_df['timestamp_ms'] - frame_timestamp).abs().argmin()]
                    R_imu = quaternion_to_rotation_matrix(
                        imu_row['stab_quat_w'],
                        imu_row['stab_quat_x'],
                        imu_row['stab_quat_y'],
                        imu_row['stab_quat_z']
                    )
                    T_vo[:3, :3] = R_imu  # Fuse orientation from IMU
                    # IMU FUSION END

                    pose = pose @ np.linalg.inv(T_vo)
                    trajectory.append(pose[:3, 3].tolist())

        prev_kp, prev_des = kp, des
        prev_frame = gray
        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    visualize_trajectory(trajectory)

if __name__ == "__main__":
    run_slam_with_imu("equirectangular.mp4")