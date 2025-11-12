"""
End-to-end pipeline:
 - Read depth frames from ROS2 bag folder (.db3 + metadata.yaml) using rosbags AnyReader
 - ROI crop, depth -> point cloud
 - RANSAC plane fit per frame, compute normal, angle, visible area
 - Save per-frame CSV, identify largest visible face
 - Estimate rotation axis from normals (PCA/SVD)
 - Visualize & save 3D axis + 2D projected circle fit

Usage:
    python full_pipeline.py
    python full_pipeline.py --no-plots
"""
## Import libraries
import os
import math
import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from rosbags.highlevel import AnyReader
from sklearn.linear_model import RANSACRegressor
from scipy.spatial import ConvexHull

## Automatic path setup
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BAG_FOLDER = SCRIPT_DIR / "depth_data"
RESULTS_DIR = SCRIPT_DIR / "Results"
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR = RESULTS_DIR / "ransac_plots"
PLOTS_DIR.mkdir(exist_ok=True)

## Configuration
BAG_FOLDER = DEFAULT_BAG_FOLDER
TOPIC_NAME = '/depth'
PROCESS_FRAMES = 7
CROP_RATIO = 0.40
RANSAC_THRESHOLD = 0.02
MIN_POINTS_PER_FRAME = 100
DEPTH_CLIP = (0.1, 5.0)
DOWNSAMPLE_FOR_PLOT = 40

## All outputs go in ./Results
OUTPUT_CSV = RESULTS_DIR / "ransac_results.csv"
OUTPUT_CSV_WITH_AXIS = RESULTS_DIR / "ransac_results_with_axis.csv"
LARGEST_TEXT = RESULTS_DIR / "largest_visible_face.txt"
ROTATION_AXIS_TXT = RESULTS_DIR / "rotation_axis.txt"
ROTATION_AXIS_IMG = RESULTS_DIR / "rotation_axis_3d.png"
CIRCLE_IMG = RESULTS_DIR / "normals_projected_circle.png"
CIRCLE_RESULTS_TXT = RESULTS_DIR / "circle_fit_results.txt"

## Utility functions
def list_bag_topics(bag_path: Path):
    with AnyReader([bag_path]) as reader:
        topics = [(c.topic, c.msgtype) for c in reader.connections]
    return topics

def read_depth_frames(bag_path: Path, topic_name: str):
    frames = []
    timestamps = []
    with AnyReader([bag_path]) as reader:
        print("Available topics in bag:")
        for c in reader.connections:
            print(f" - {c.topic} [{c.msgtype}]")
        conns = [x for x in reader.connections if x.topic == topic_name]
        if not conns:
            print(f"\n Topic '{topic_name}' not found in bag. Choose one from the above and re-run.")
            return [], []
        for conn, t, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            dtype = np.dtype(np.uint16 if getattr(msg, 'encoding', '') == '16UC1' else np.float32)
            depth = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
            if np.nanmax(depth) > 50:
                depth = depth / 1000.0  # mm → m
            frames.append(depth)
            timestamps.append(t)
    return frames, timestamps

def depth_to_pointcloud(depth, fx=None, fy=None, cx=None, cy=None, depth_min=0.1, depth_max=5.0, smooth=True):
    h, w = depth.shape
    if fx is None or fy is None:
        fx = fy = w
    if cx is None or cy is None:
        cx, cy = w / 2.0, h / 2.0
    depth = depth.astype(np.float32)
    depth[depth <= 0] = np.nan
    depth = np.clip(depth, depth_min, depth_max)
    if smooth:
        depth = cv2.medianBlur(depth, 5)
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    pts = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    mask = ~np.isnan(pts).any(axis=1)
    return pts[mask]

def fit_plane_ransac(points, residual_threshold=0.02):
    if len(points) < 3:
        return None, None, np.empty((0,3))
    X = points[:, :2]
    y = points[:, 2]
    model = RANSACRegressor(residual_threshold=residual_threshold, max_trials=1000)
    model.fit(X, y)
    a, b = model.estimator_.coef_
    c = model.estimator_.intercept_
    normal = np.array([a, b, -1.0])
    normal /= np.linalg.norm(normal)
    inliers = points[model.inlier_mask_]
    return normal, c, inliers

def compute_visible_area(points):
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points[:, :2])
        return float(hull.volume)
    except Exception:
        return 0.0

def draw_bounding_box(ax, inliers, color='cyan', linewidth=1.5):
    if len(inliers) < 4:
        return
    try:
        hull = ConvexHull(inliers[:, :2])
        hull_pts = inliers[hull.vertices]
        z_mean = float(np.nanmean(inliers[:,2]))
        for i in range(len(hull_pts)):
            p1 = hull_pts[i]; p2 = hull_pts[(i+1) % len(hull_pts)]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [z_mean, z_mean], color=color, linewidth=linewidth)
    except Exception as e:
        print("Bounding box skipped (ConvexHull error):", e)


## Task 1: detect planes, visualize frames, save metrics
def run_task1(frames, show_plots=True):
    if len(frames) == 0:
        raise RuntimeError("No frames to process.")

    h, w = frames[0].shape
    h1 = int(h * (0.5 - CROP_RATIO / 2))
    h2 = int(h * (0.5 + CROP_RATIO / 2))
    w1 = int(w * (0.5 - CROP_RATIO / 2))
    w2 = int(w * (0.5 + CROP_RATIO / 2))

    results = []

    # --- Create folder for frame plots ---
    PLOTS_DIR = RESULTS_DIR / "ransac_plots"
    PLOTS_DIR.mkdir(exist_ok=True)

    for i, depth in enumerate(frames[:PROCESS_FRAMES]):
        cropped = depth[h1:h2, w1:w2]
        points = depth_to_pointcloud(
            cropped,
            depth_min=DEPTH_CLIP[0],
            depth_max=DEPTH_CLIP[1],
            smooth=True
        )
        if len(points) < MIN_POINTS_PER_FRAME:
            print(f"Skipping Frame {i+1}: only {len(points)} points")
            continue

        normal, c, inliers = fit_plane_ransac(points, residual_threshold=RANSAC_THRESHOLD)
        if normal is None or len(inliers) == 0:
            print(f"RANSAC failed for Frame {i+1}")
            continue

        cam_axis = np.array([0.0, 0.0, 1.0])
        angle = np.degrees(np.arccos(np.clip(np.dot(normal, cam_axis), -1.0, 1.0)))
        if angle > 90.0:
            angle = 180.0 - angle
        area = compute_visible_area(inliers)

        results.append({
            "Frame": i + 1,
            "Normal_X": float(normal[0]),
            "Normal_Y": float(normal[1]),
            "Normal_Z": float(normal[2]),
            "Angle_deg": round(float(angle), 3),
            "Visible_Area_m2": round(float(area), 6),
            "InlierCount": int(len(inliers))
        })

        # --- Generate and save individual 3D plot per frame ---
        if show_plots:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection='3d')

            sample = points[::max(1, DOWNSAMPLE_FOR_PLOT)]
            ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=0.5, color='gray', alpha=0.3)
            ax.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2], s=2.0, color='red', label='Inliers')

            z_mean = float(np.nanmean(inliers[:, 2]))
            ax.quiver(0, 0, z_mean, normal[0], normal[1], normal[2],
                      length=0.25, color='blue', label='Normal')

            draw_bounding_box(ax, inliers)

            ax.set_title(f"Frame {i+1}\nAngle={angle:.1f}°", fontsize=10)
            ax.set_xlabel('Y [m]')
            ax.set_ylabel('X [m]')
            ax.set_zlabel('Z [m]')
            ax.view_init(elev=20, azim=0)
            ax.set_box_aspect([1, 1, 1])
            ax.legend(fontsize=6, loc='upper left')

            # Save figure
            frame_plot_path = PLOTS_DIR / f"frame_{i+1:02d}.png"
            plt.tight_layout()
            plt.savefig(frame_plot_path, dpi=200, bbox_inches='tight')
            print(f"✅ Saved plot for Frame {i+1} → '{frame_plot_path}'")
            plt.show(block=True)
            plt.close(fig)

    df = pd.DataFrame(results)
    return df


def summarize_and_save_task1(df):
    # Accept None as empty
    if df is None:
        df = pd.DataFrame()
    if df.empty:
        print("No detections to save.")
        pd.DataFrame().to_csv(OUTPUT_CSV, index=False)
        return df, None

    idx_max = df['Visible_Area_m2'].idxmax()
    largest_row = df.loc[idx_max]
    largest_frame = int(largest_row['Frame'])
    largest_area = float(largest_row['Visible_Area_m2'])
    largest_angle = float(largest_row['Angle_deg'])

    print("\n=== Largest visible face summary ===")
    print(f"Frame: {largest_frame}, Area: {largest_area:.6f} m^2, Angle: {largest_angle:.3f}°")

    with open(LARGEST_TEXT, "w") as f:
        f.write("Largest visible face summary\n")
        f.write(f"Frame: {largest_frame}\n")
        f.write(f"Visible Area (m^2): {largest_area:.6f}\n")
        f.write(f"Normal Angle (deg): {largest_angle:.3f}\n")

    df['IsLargest'] = False
    df.at[idx_max, 'IsLargest'] = True
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved per-frame results to '{OUTPUT_CSV}' and largest info to '{LARGEST_TEXT}'")
    return df, int(largest_frame)


## Task 2: estimate rotation axis from normals and visualize
def estimate_rotation_axis_from_normals(normals):
    normals = np.asarray(normals)
    if normals.shape[0] < 3:
        raise ValueError("Need at least 3 normals to estimate axis.")
    # align normals to same hemisphere using mean
    mean_n = normals.mean(axis=0)
    for i in range(normals.shape[0]):
        if np.dot(normals[i], mean_n) < 0:
            normals[i] = -normals[i]
    mean_n = normals.mean(axis=0)
    N_centered = normals - mean_n
    U, S, Vt = np.linalg.svd(N_centered, full_matrices=False)
    axis = Vt[-1, :]
    axis /= np.linalg.norm(axis)
    if axis[2] < 0:
        axis = -axis
    return axis, mean_n, S

def visualize_axis_3d(normals, axis, mean_n, savepath=ROTATION_AXIS_IMG, show_plot=True):
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    for i, n in enumerate(normals):
        ax.quiver(0,0,0, n[0], n[1], n[2], color='blue', length=1.0, normalize=True, alpha=0.7)
        ax.text(n[0]*1.05, n[1]*1.05, n[2]*1.05, f'F{i+1}', color='navy', fontsize=9)
    ax.scatter(mean_n[0], mean_n[1], mean_n[2], color='orange', s=60, label='Mean normals')
    ax.quiver(0,0,0, axis[0], axis[1], axis[2], color='red', length=1.3, linewidth=2.5, label='Rotation axis')
    ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([-1,1])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_box_aspect([1,1,1])
    ax.legend()
    ax.set_title('Rotation Axis and Frame Normals (Camera frame)')
    plt.tight_layout()
    fig.savefig(savepath, dpi=200, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close(fig)
    print(f"Saved 3D axis visualization to '{savepath}'")

def project_normals_to_plane(normals, axis):
    axis = np.asarray(axis) / np.linalg.norm(axis)
    arbitrary = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(axis, arbitrary)) > 0.9:
        arbitrary = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, arbitrary); u /= np.linalg.norm(u)
    v = np.cross(axis, u); v /= np.linalg.norm(v)
    proj = []
    for n in normals:
        p = n - np.dot(n, axis) * axis
        proj.append([np.dot(p, u), np.dot(p, v)])
    return np.array(proj), u, v

def fit_circle_least_squares(x, y):
    A = np.column_stack([2*x, 2*y, np.ones_like(x)])
    b = x**2 + y**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    xc, yc, c = sol
    r = math.sqrt(max(0.0, xc*xc + yc*yc + c))
    return float(xc), float(yc), float(r)

def project_and_fit_circle(normals, axis, save_img=CIRCLE_IMG, save_txt=CIRCLE_RESULTS_TXT, show_plot=True):
    proj, uvec, vvec = project_normals_to_plane(normals, axis)
    xs = proj[:,0]; ys = proj[:,1]
    xc, yc, r = fit_circle_least_squares(xs, ys)
    residuals = np.sqrt((xs-xc)**2 + (ys-yc)**2) - r

    # plot
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(xs, ys, c='tab:blue', s=60, label='Projected normals')
    for i,(px,py) in enumerate(zip(xs, ys)):
        ax.text(px*1.05, py*1.05, f'F{i+1}', fontsize=9, color='navy')
    theta = np.linspace(0,2*math.pi,240)
    ax.plot(xc + r*np.cos(theta), yc + r*np.sin(theta), color='red', linewidth=1.8, label=f'Fitted circle r={r:.4f}')
    ax.scatter([xc],[yc], color='orange', s=60, label='Circle center')
    ax.set_aspect('equal','box'); ax.grid(True)
    ax.set_xlabel('u'); ax.set_ylabel('v')
    ax.legend(loc='upper left'); ax.set_title('Normals projected onto plane ⟂ axis (2D)')
    fig.savefig(save_img, dpi=200, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close(fig)

    # save diagnostics
    with open(save_txt, 'w') as f:
        f.write("2D Projection / Circle Fit Results\n")
        f.write(f"Circle center (u,v): {xc:.6f}, {yc:.6f}\n")
        f.write(f"Radius: {r:.6f}\n")
        f.write("Residuals per frame: " + np.array2string(np.round(residuals,6)) + "\n")
        f.write(f"Mean abs residual: {np.mean(np.abs(residuals)):.6f}\n")

    print(f"Saved 2D circle image to '{save_img}' and diagnostics to '{save_txt}'") 
    return xc, yc, r, residuals


## Runner (calling all functions in order)
def main(no_plots=False):
    # 1) read frames
    if not BAG_FOLDER.exists():
        raise RuntimeError(f"Bag folder '{BAG_FOLDER}' does not exist.")
    frames, timestamps = read_depth_frames(BAG_FOLDER, TOPIC_NAME)
    print(f"\n Extracted {len(frames)} frames.")
    if len(frames) == 0:
        return

    # 2) Task 1: plane detection and per-frame metrics + visualization
    df = run_task1(frames, show_plots=not no_plots)
    df, largest_frame = summarize_and_save_task1(df)

    # 3) Task 2: rotation axis estimation & visuals
    if df.empty or not {'Normal_X','Normal_Y','Normal_Z'}.issubset(df.columns):
        print("No normals available for Task 2. Exiting.")
        return

    normals = df[['Normal_X','Normal_Y','Normal_Z']].to_numpy()
    try:
        axis_vec, mean_n, svd_vals = estimate_rotation_axis_from_normals(normals)
        print("\nEstimated rotation axis (unit):", np.round(axis_vec, 4))
        print("SVD singular values:", np.round(svd_vals,4))
        # save axis numeric
        with open(ROTATION_AXIS_TXT, 'w') as f:
            f.write("Estimated rotation axis (camera frame)\n")
            f.write("Axis (unit): " + np.array2string(axis_vec, precision=6) + "\n")
            f.write("Mean normals: " + np.array2string(mean_n, precision=6) + "\n")
            f.write("SVD singular values: " + np.array2string(svd_vals, precision=6) + "\n")
        # 3D visualize and save
        visualize_axis_3d(normals, axis_vec, mean_n, savepath=ROTATION_AXIS_IMG, show_plot=not no_plots)
        # 2D projection + circle fit
        xc, yc, r, residuals = project_and_fit_circle(normals, axis_vec, show_plot=not no_plots)
        # save augmented csv
        df['RotationAxis_X'] = float(axis_vec[0])
        df['RotationAxis_Y'] = float(axis_vec[1])
        df['RotationAxis_Z'] = float(axis_vec[2])
        df.to_csv(OUTPUT_CSV_WITH_AXIS, index=False)
        print(f"Saved augmented CSV '{OUTPUT_CSV_WITH_AXIS}'")
    except Exception as e:
        print("Rotation axis estimation failed:", e)

    # -------------------------
    # 5.3 Temporal Validation (Optional)
    # -------------------------
    try:
        if len(frames) > 1 and "Angle_deg" in df.columns:
            print("\n Performing temporal validation...")

            # Prepare temporal data
            if len(timestamps) == len(df):
                # Convert nanoseconds to seconds
                t_rel = (np.array(timestamps) - timestamps[0]) * 1e-9
                x_vals = t_rel
                x_label = "Time [s]"
            else:
                # fallback to frame index if timestamps unavailable
                x_vals = np.arange(1, len(df) + 1)
                x_label = "Frame Index"

            # Plot angle over time
            plt.figure(figsize=(8, 4))
            plt.plot(x_vals, df["Angle_deg"], marker='o', color='tab:blue', linewidth=2)
            plt.title("Temporal Validation: Plane Angle Across Frames")
            plt.xlabel(x_label)
            plt.ylabel("Plane Angle (°)")
            plt.grid(True)
            plt.tight_layout()

            # Save to /Results/
            temporal_plot_path = RESULTS_DIR / "temporal_angle_validation.png"
            plt.savefig(temporal_plot_path, dpi=200, bbox_inches="tight")
            print(f"Saved temporal validation plot to '{temporal_plot_path}'")

            if not no_plots:
                plt.show()

            plt.close()
        else:
            print("Temporal validation skipped (insufficient frames or missing angle data).")

    except Exception as e:
        print(f"Temporal validation failed: {e}")


## CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Full pipeline for plane detection and rotation axis estimation")
    ap.add_argument("--bag", type=str, default=str(DEFAULT_BAG_FOLDER), help="Path to bag folder (metadata.yaml + .db3)")
    ap.add_argument("--topic", type=str, default=TOPIC_NAME, help="Depth topic name")
    ap.add_argument("--no-plots", action="store_true", help="Do not show interactive plots (save-only mode)")
    args = ap.parse_args()

    BAG_FOLDER = Path(args.bag)
    TOPIC_NAME = args.topic

    if not BAG_FOLDER.exists():
        raise RuntimeError(f"Bag folder '{BAG_FOLDER}' not found. Place 'depth_data' in the same directory as this script.")
    
main(no_plots=args.no_plots)