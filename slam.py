import cv2
import numpy as np
import open3d as o3d

def build_equirect_remap(h_in, w_in, h_out, w_out, fov=90):
    K = np.array([[w_out / (2 * np.tan(np.radians(fov / 2))), 0, w_out / 2],
                  [0, w_out / (2 * np.tan(np.radians(fov / 2))), h_out / 2],
                  [0, 0, 1]])
    map_x = np.zeros((h_out, w_out), dtype=np.float32)
    map_y = np.zeros((h_out, w_out), dtype=np.float32)
    for y in range(h_out):
        for x in range(w_out):
            xn = (x - w_out / 2) / K[0, 0]
            yn = (y - h_out / 2) / K[1, 1]
            zn = 1.0
            vec = np.array([xn, yn, zn])
            vec /= np.linalg.norm(vec)
            lat = np.arcsin(vec[1])
            lon = np.arctan2(vec[0], vec[2])
            u = (lon + np.pi) / (2 * np.pi) * w_in
            v = (np.pi / 2 - lat) / np.pi * h_in
            map_x[y, x] = u
            map_y[y, x] = v
    return map_x, map_y

def equirect_to_perspective_fast(img, map_x, map_y):
    persp = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return cv2.flip(persp, 0)

def visualize_slam(trajectory, landmarks):
    traj_pts = np.array(trajectory)
    landmark_pts = np.array(landmarks)

    traj_pcd = o3d.geometry.PointCloud()
    traj_pcd.points = o3d.utility.Vector3dVector(traj_pts)
    traj_pcd.paint_uniform_color([1, 0, 0])

    map_pcd = o3d.geometry.PointCloud()
    if len(landmark_pts) > 0:
        landmark_pcd_raw = o3d.geometry.PointCloud()
        landmark_pcd_raw.points = o3d.utility.Vector3dVector(landmark_pts)
        landmark_pcd_down = landmark_pcd_raw.voxel_down_sample(voxel_size=0.1)
        landmark_pcd_down.paint_uniform_color([0, 1, 0])
        map_pcd = landmark_pcd_down

    line_set = o3d.geometry.LineSet()
    if len(traj_pts) >= 2:
        lines = [[i, i + 1] for i in range(len(traj_pts) - 1)]
        line_set.points = o3d.utility.Vector3dVector(traj_pts)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in lines])

    frustums = []
    for pt in traj_pts:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
        frame.translate(pt)
        frustums.append(frame)

    start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    start_sphere.paint_uniform_color([0, 1, 0])
    start_sphere.translate(traj_pts[0])

    end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    end_sphere.paint_uniform_color([0, 0, 1])
    end_sphere.translate(traj_pts[-1])

    o3d.visualization.draw_geometries([traj_pcd, map_pcd, line_set, start_sphere, end_sphere] + frustums)

def run_slam(video_path):
    cap = cv2.VideoCapture(video_path)
    orb = cv2.ORB_create(2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    pose = np.eye(4)
    trajectory = [pose[:3, 3].tolist()]
    all_landmarks = []
    MAP_H, MAP_W = 180, 320
    map_x, map_y = None, None
    prev_kp, prev_des, prev_frame = None, None, None
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % 4 != 0:
            continue
        print(f"[INFO] Processing frame {frame_idx}")
        if map_x is None or map_y is None:
            h_in, w_in = frame.shape[:2]
            map_x, map_y = build_equirect_remap(h_in, w_in, MAP_H, MAP_W)
        persp = equirect_to_perspective_fast(frame, map_x, map_y)
        gray = cv2.cvtColor(persp, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        vis = cv2.drawKeypoints(gray, kp, None, color=(0, 255, 0))
        cv2.imshow("ORB Features", vis)
        if prev_des is not None:
            matches = bf.match(prev_des, des)
            matches = sorted(matches, key=lambda x: x.distance)
            if len(matches) > 12:
                pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches])
                pts2 = np.float32([kp[m.trainIdx].pt for m in matches])
                K = np.array([[300, 0, MAP_W / 2],
                              [0, 300, MAP_H / 2],
                              [0,   0,        1]])
                E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=1.0)
                if E is not None:
                    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
                    T = np.eye(4)
                    T[:3, :3] = R
                    T[:3, 3] = t.squeeze()
                    delta_translation = np.linalg.norm(t)
                    if delta_translation > 0.05:
                        pose = pose @ np.linalg.inv(T)
                        trajectory.append(pose[:3, 3].tolist())
                        P0 = K @ np.eye(3, 4)
                        P1 = K @ np.hstack((R, t))
                        pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, None)
                        pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, None)
                        pts_4d = cv2.triangulatePoints(P0, P1, pts1.T, pts2.T)
                        pts_4d /= pts_4d[3]
                        pts_3d = pts_4d[:3].T
                        for pt in pts_3d:
                            z = pt[2]
                            if 0 < z < 30:
                                all_landmarks.append(pt.tolist())
                        match_vis = cv2.drawMatches(prev_frame, prev_kp, gray, kp, matches[:30], None, flags=2)
                        cv2.imshow("ORB Matches", match_vis)
        prev_kp, prev_des = kp, des
        prev_frame = gray.copy()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Finished processing. Launching viewer...")
    visualize_slam(trajectory, all_landmarks)

if __name__ == '__main__':
    run_slam("equirectangular.mp4")

