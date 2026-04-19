#!/usr/bin/env python3
from __future__ import annotations
"""
colmap_pipeline.py

Subscribes to:
    /mono_cam/image_raw          (sensor_msgs/msg/Image)
    /mono_cam/camera_info        (sensor_msgs/msg/CameraInfo)
    /fmu/out/vehicle_odometry    (px4_msgs/msg/VehicleOdometry)

Pipeline:
    1. Records images to colmap_dataset/images/ at a >0.2m baseline.
    2. Stores exact camera poses from PX4 Odometry.
    3. On shutdown (Ctrl+C), generates COLMAP cameras.txt and images.txt priors.
    4. Automatically launches COLMAP to extract, match, and triangulate using priors.

Dependencies:
    pip install "numpy<2" opencv-python
    sudo apt install colmap
"""

import os
import sys
import shutil
import signal
import subprocess
import argparse
import sqlite3
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    from sensor_msgs.msg import Image, CameraInfo
    from px4_msgs.msg import VehicleOdometry
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
except ImportError:
    rclpy = None
    QoSProfile = ReliabilityPolicy = HistoryPolicy = DurabilityPolicy = None
    Image = CameraInfo = VehicleOdometry = None
    CvBridge = None
    ROS_AVAILABLE = False

    class _DummyLogger:
        def info(self, *_args, **_kwargs):
            pass

        def warn(self, *_args, **_kwargs):
            pass

        def fatal(self, *_args, **_kwargs):
            pass

    class Node:
        def __init__(self, *_args, **_kwargs):
            self._logger = _DummyLogger()

        def get_logger(self):
            return self._logger

        def create_subscription(self, *_args, **_kwargs):
            return None

        def create_timer(self, *_args, **_kwargs):
            return None

        def destroy_node(self):
            return None

@dataclass
class ColmapFrame:
    image_name: str
    position: np.ndarray        # [x, y, z] NED
    orientation_q: np.ndarray   # [w, x, y, z] PX4 Body to NED


# Body-frame (X-fwd, Y-right, Z-down) → OpenCV camera (X-right, Y-down, Z-fwd)
_R_BODY_TO_CAM = np.array([
    [0,  1,  0],
    [0,  0,  1],
    [1,  0,  0],
], dtype=np.float64)

def quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,  2*x*y - 2*z*w,      2*x*z + 2*y*w     ],
        [2*x*y + 2*z*w,      1 - 2*x*x - 2*z*z,  2*y*z - 2*x*w     ],
        [2*x*z - 2*y*w,      2*y*z + 2*x*w,      1 - 2*x*x - 2*y*y ],
    ], dtype=np.float64)


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to quaternion [w, x, y, z]."""
    trace = np.trace(R)
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    
    # Normalize to ensure unit length 
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def remap_prior_image_ids_to_db(model_dir: str, db_path: str) -> bool:
    """Rewrite prior images.txt IDs to match COLMAP database IMAGE_ID assignment."""
    images_txt = os.path.join(model_dir, "images.txt")

    if not os.path.exists(images_txt) or not os.path.exists(db_path):
        return False

    with open(images_txt, "r") as f:
        lines = f.readlines()

    pose_lines = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) >= 10:
            pose_lines.append((idx, parts))

    if not pose_lines:
        return False

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT image_id, name FROM images")
        db_rows = cursor.fetchall()
    finally:
        conn.close()

    db_id_by_name = {name: image_id for image_id, name in db_rows}

    remapped_entries = []
    for _, parts in pose_lines:
        name = parts[9]
        if name not in db_id_by_name:
            return False
        remapped_entries.append((db_id_by_name[name], parts))

    remapped_entries.sort(key=lambda x: x[0])

    header_lines = [ln for ln in lines if ln.strip().startswith("#")]
    new_lines = list(header_lines)
    for db_image_id, parts in remapped_entries:
        parts[0] = str(db_image_id)
        new_lines.append(" ".join(parts) + "\n")
        new_lines.append("\n")

    with open(images_txt, "w") as f:
        f.writelines(new_lines)

    return True


def has_colmap_model(model_path: str) -> bool:
    bin_files = ["cameras.bin", "images.bin", "points3D.bin"]
    txt_files = ["cameras.txt", "images.txt", "points3D.txt"]
    has_bin = all(os.path.exists(os.path.join(model_path, f)) for f in bin_files)
    has_txt = all(os.path.exists(os.path.join(model_path, f)) for f in txt_files)
    return has_bin or has_txt


def points3d_count_from_bin(model_path: str) -> Optional[int]:
    points_bin = os.path.join(model_path, "points3D.bin")
    if not os.path.exists(points_bin):
        return None

    # COLMAP binary points3D starts with uint64 count.
    with open(points_bin, "rb") as f:
        header = f.read(8)
    if len(header) < 8:
        return None
    return int.from_bytes(header, byteorder="little", signed=False)


def database_match_stats(db_path: str) -> tuple[int, int]:
    """Return (matches_count, two_view_geometries_count) from the COLMAP database."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM matches")
        matches_count = int(cursor.fetchone()[0])
        cursor.execute("SELECT COUNT(*) FROM two_view_geometries")
        two_view_count = int(cursor.fetchone()[0])
    finally:
        conn.close()
    return matches_count, two_view_count


def run_colmap_matcher(colmap_bin: str, db_path: str, use_gpu: str, matcher: str, image_path: str) -> None:
    if matcher == "custom":
        match_list_path = os.path.join(os.path.dirname(db_path), "custom_matches.txt")
        if not os.path.exists(match_list_path):
            raise FileNotFoundError(match_list_path)

        print(">> Running Matches Importer with custom pairs...")
        subprocess.run([
            colmap_bin, "matches_importer",
            "--database_path", db_path,
            "--match_list_path", match_list_path,
            "--match_type", "pairs",
            "--SiftMatching.use_gpu", use_gpu,
        ], check=True)
        return

    if matcher == "exhaustive":
        print(">> Running Exhaustive Matcher...")
        subprocess.run([
            colmap_bin, "exhaustive_matcher",
            "--database_path", db_path,
            "--SiftMatching.use_gpu", use_gpu,
        ], check=True)
        return

    raise ValueError(f"Unknown matcher: {matcher}")


def copy_model_files(src_model: str, dst_model: str) -> bool:
    if not has_colmap_model(src_model):
        return False

    os.makedirs(dst_model, exist_ok=True)
    copied_any = False
    for name in ["cameras.bin", "images.bin", "points3D.bin", "cameras.txt", "images.txt", "points3D.txt"]:
        src = os.path.join(src_model, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_model, name))
            copied_any = True
    return copied_any


def canonicalize_model_to_zero(sparse_dir: str) -> Optional[str]:
    root_model = sparse_dir
    zero_model = os.path.join(sparse_dir, "0")

    if has_colmap_model(zero_model):
        return zero_model

    if has_colmap_model(root_model) and copy_model_files(root_model, zero_model):
        return zero_model

    return None


def prompt_before_triangulation() -> bool:
    try:
        answer = input("Triangulation fallback is about to run. Proceed? [y/N]: ").strip().lower()
    except EOFError:
        return False

    return answer in {"y", "yes"}


class ColmapCollectorNode(Node):
    COLLECT_INTERVAL_SEC = 1.0
    MIN_BASELINE_M = 0.2

    def __init__(self, dataset_dir: Optional[str] = None, clean_dataset: bool = True):
        super().__init__('colmap_collector')
        self._bridge = CvBridge()

        self._dataset_dir = os.path.abspath(dataset_dir or "colmap_dataset")
        self._images_dir = os.path.join(self._dataset_dir, "images")
        self._model_dir = os.path.join(self._dataset_dir, "sparse", "model")
        
        # Fresh dataset only when explicitly collecting new data.
        if clean_dataset and os.path.exists(self._dataset_dir):
            self.get_logger().info(f'Cleaning old dataset: {self._dataset_dir}')
            shutil.rmtree(self._dataset_dir)
            
        os.makedirs(self._images_dir, exist_ok=True)
        os.makedirs(self._model_dir, exist_ok=True)

        self._frames: List[ColmapFrame] = []
        self._latest_image: Optional[np.ndarray] = None
        self._camera_K: Optional[np.ndarray] = None
        self._img_width = 0
        self._img_height = 0
        self._latest_position = np.zeros(3)
        self._latest_orientation = np.array([1.0, 0.0, 0.0, 0.0]) # w, x, y, z
        self._img_counter = 0

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.create_subscription(Image, '/mono_cam/image_raw', self._image_cb, 10)
        self.create_subscription(CameraInfo, '/mono_cam/camera_info', self._caminfo_cb, 10)
        self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self._odom_cb, px4_qos)

        self.create_timer(self.COLLECT_INTERVAL_SEC, self._capture_tick)

        self.get_logger().info('📸 Colmap data collector ready! Fly the drone, then press Ctrl+C to trigger matching.')

    def _image_cb(self, msg: Image):
        try:
            self._latest_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self._img_width = msg.width
            self._img_height = msg.height
        except AttributeError as e:
            if "_ARRAY_API" in str(e):
                self.get_logger().fatal("cv_bridge failed due to NumPy 2.x incompatibility! Run: pip install 'numpy<2'")
                sys.exit(1)

    def _caminfo_cb(self, msg: CameraInfo):
        if self._camera_K is None:
            self._camera_K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def _odom_cb(self, msg: VehicleOdometry):
        self._latest_position = np.array([msg.position[0], msg.position[1], msg.position[2]], dtype=np.float64)
        self._latest_orientation = np.array([msg.q[0], msg.q[1], msg.q[2], msg.q[3]], dtype=np.float64)

    def _capture_tick(self):
        if self._latest_image is None or self._camera_K is None:
            return

        pos = self._latest_position.copy()

        if self._frames:
            baseline = np.linalg.norm(pos - self._frames[-1].position)
            if baseline < self.MIN_BASELINE_M:
                return

        self._img_counter += 1
        img_name = f"img_{self._img_counter:04d}.jpg"
        img_path = os.path.join(self._images_dir, img_name)
        
        cv2.imwrite(img_path, self._latest_image)

        frame = ColmapFrame(
            image_name=img_name,
            position=pos,
            orientation_q=self._latest_orientation.copy()
        )
        self._frames.append(frame)
        
        self.get_logger().info(f'Saved {img_name} | Total valid: {len(self._frames)} frames')

    def generate_colmap_model(self):
        """Converts collected data into Colmap priors on shutdown."""
        if len(self._frames) == 0:
            self.get_logger().warn("No frames collected. Skipping Colmap.")
            return False

        cam_path = os.path.join(self._model_dir, "cameras.txt")
        img_path = os.path.join(self._model_dir, "images.txt")
        pts_path = os.path.join(self._model_dir, "points3D.txt")

        # Create points3D.txt (empty initially)
        with open(pts_path, "w") as f:
            pass

        # Create cameras.txt
        with open(cam_path, "w") as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            # PINHOLE requires: fx, fy, cx, cy
            K = self._camera_K
            fx, fy = K[0,0], K[1,1]
            cx, cy = K[0,2], K[1,2]
            f.write(f"1 PINHOLE {self._img_width} {self._img_height} {fx} {fy} {cx} {cy}\n")

        # Create images.txt
        with open(img_path, "w") as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

            for i, frame in enumerate(self._frames):
                img_id = i + 1
                
                # PX4: q is w,x,y,z from Body -> NED
                R_body2ned = quat_to_rot(frame.orientation_q)
                
                # Colmap: R_cam is World(NED) -> Camera
                R_cam = _R_BODY_TO_CAM @ R_body2ned.T
                
                # T_cam = -R_cam @ P_world
                T_cam = -R_cam @ frame.position.reshape(3, 1)
                
                # Convert R_cam (3x3) to Quat (Qw, Qx, Qy, Qz)  
                qw, qx, qy, qz = rot_to_quat(R_cam)
                tx, ty, tz = T_cam.flatten()

                f.write(f"{img_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {frame.image_name}\n")
                f.write("\n")

        # Create custom_matches.txt for efficient Loop Closure
        match_path = os.path.join(self._dataset_dir, "custom_matches.txt")
        num_pairs = 0
        with open(match_path, "w") as f:
            for i in range(len(self._frames)):
                for j in range(i + 1, len(self._frames)):
                    f1 = self._frames[i]
                    f2 = self._frames[j]
                    
                    # 1. Distance check (only match if within 15 meters physical distance)
                    dist = np.linalg.norm(f1.position - f2.position)
                    if dist < 15.0:
                        temporal_neighbor = (j - i) <= 5

                        # 2. View angle check (dot product of camera forward vectors)
                        R1 = quat_to_rot(f1.orientation_q)
                        R2 = quat_to_rot(f2.orientation_q)
                        
                        # PX4 body forward is X [1, 0, 0]
                        v1 = R1[:, 0]
                        v2 = R2[:, 0]
                        
                        # Cosine similarity (dot product of unit vectors)
                        dot = np.dot(v1, v2)
                        
                        # Always connect nearby temporal neighbors; otherwise allow wider view-angle gating.
                        if temporal_neighbor or dot > -0.3:
                            f.write(f"{f1.image_name} {f2.image_name}\n")
                            num_pairs += 1

        self.get_logger().info(f"✅ Generated exact PX4 camera priors in {self._model_dir}")
        self.get_logger().info(f"✅ Generated {num_pairs} custom spatial match pairs for O(N) Loop Closure")
        return True


def run_dense_reconstruction(colmap_bin: str, dataset_dir: str, sparse_model_dir: str) -> Optional[str]:
    dense_dir = os.path.join(dataset_dir, "dense")

    if os.path.isdir(dense_dir):
        shutil.rmtree(dense_dir)
    os.makedirs(dense_dir, exist_ok=True)

    print(">> Running Image Undistorter...")
    subprocess.run([
        colmap_bin, "image_undistorter",
        "--image_path", os.path.join(dataset_dir, "images"),
        "--input_path", sparse_model_dir,
        "--output_path", dense_dir,
        "--output_type", "COLMAP",
    ], check=True)

    print(">> Running Patch Match Stereo...")
    subprocess.run([
        colmap_bin, "patch_match_stereo",
        "--workspace_path", dense_dir,
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true",
        "--PatchMatchStereo.max_image_size", "1600",
        "--PatchMatchStereo.num_iterations", "3",
    ], check=True)

    print(">> Running Stereo Fusion...")
    fused_path = os.path.join(dense_dir, "fused.ply")
    subprocess.run([
        colmap_bin, "stereo_fusion",
        "--workspace_path", dense_dir,
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", fused_path,
        # Memory-safe settings for 8 GB VRAM / limited RAM systems.
        "--StereoFusion.min_num_pixels", "5",
        "--StereoFusion.max_image_size", "2000",
        "--StereoFusion.cache_size", "4",
    ], check=True)

    print(f">> Dense reconstruction saved to {fused_path}")
    return fused_path

def resolve_colmap_binary() -> str:
    """Resolve COLMAP binary in a portable order for host and container runs."""
    env_colmap_bin = os.environ.get("COLMAP_BIN", "").strip()
    if env_colmap_bin:
        return env_colmap_bin

    local_cuda_build = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "colmap_cuda_src",
            "build_cuda",
            "src",
            "colmap",
            "exe",
            "colmap",
        )
    )
    candidates = [
        local_cuda_build,
        "/opt/colmap/bin/colmap",
        "/usr/local/bin/colmap",
        "/usr/bin/colmap",
    ]

    for candidate in candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    path_colmap = shutil.which("colmap")
    if path_colmap:
        return path_colmap

    raise FileNotFoundError("COLMAP binary not found. Set COLMAP_BIN to the executable path.")

def run_colmap(dataset_dir: str):
    print("\n🚀 Starting Automated Colmap Pipeline...\n")
    dataset_dir = os.path.abspath(dataset_dir)
    db_path = os.path.join(dataset_dir, "db.db")
    img_dir = os.path.join(dataset_dir, "images")
    sparse_dir = os.path.join(dataset_dir, "sparse")
    model_dir = os.path.join(sparse_dir, "model")
    canonical_out_dir = os.path.join(sparse_dir, "0")
    triangulated_out_dir = os.path.join(sparse_dir, "triangulated")
    match_list_path = os.path.join(dataset_dir, "custom_matches.txt")
    colmap_bin = resolve_colmap_binary()
    use_gpu = "1"

    os.makedirs(sparse_dir, exist_ok=True)

    # Keep priors in sparse/model and let mapper use them as the initial model.
    # Clear old numeric reconstruction folders (e.g. sparse/0) so COLMAP rebuilds from the PX4 seed.
    for entry in os.listdir(sparse_dir):
        entry_path = os.path.join(sparse_dir, entry)
        if entry.isdigit() and os.path.isdir(entry_path):
            shutil.rmtree(entry_path)

    if os.path.isdir(triangulated_out_dir):
        shutil.rmtree(triangulated_out_dir)

    try:
        # Auto-fallback to CPU when COLMAP is built without CUDA support.
        try:
            help_proc = subprocess.run(
                [colmap_bin, "help"],
                check=True,
                capture_output=True,
                text=True,
            )
            if "without CUDA" in help_proc.stdout:
                use_gpu = "0"
        except Exception:
            pass

        print(f">> COLMAP GPU mode: {use_gpu}")

        # 1. Wipe old DB and recreate
        if os.path.exists(db_path):
            os.remove(db_path)

        print(">> Creating fresh COLMAP database...")
        subprocess.run([
            colmap_bin, "database_creator",
            "--database_path", db_path,
        ], check=True)

        # 2. Feature Extraction
        print(">> Running Feature Extractor...")
        subprocess.run([
            colmap_bin, "feature_extractor",
            "--database_path", db_path,
            "--image_path", img_dir,
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.max_num_features", "8000",
            "--SiftExtraction.use_gpu", use_gpu
        ], check=True)

        # 3. Try custom pair matching first; fall back to exhaustive matching if it did not verify any pairs.
        if not os.path.exists(match_list_path):
            print(f"\n❌ Missing match list file: {match_list_path}")
            return

        run_colmap_matcher(colmap_bin, db_path, use_gpu, "custom", img_dir)

        matches_count, two_view_count = database_match_stats(db_path)
        print(f">> Match stats after custom matcher: matches={matches_count}, two_view_geometries={two_view_count}")

        if two_view_count == 0:
            print(">> Custom matcher produced no verified two-view geometries; falling back to exhaustive matching...")
            run_colmap_matcher(colmap_bin, db_path, use_gpu, "exhaustive", img_dir)
            matches_count, two_view_count = database_match_stats(db_path)
            print(f">> Match stats after exhaustive matcher: matches={matches_count}, two_view_geometries={two_view_count}")

        # 3b. Ensure prior image IDs match COLMAP DB IDs before loading input model.
        if not remap_prior_image_ids_to_db(model_dir, db_path):
            print("\n❌ Failed to remap prior IMAGE_IDs against database image IDs.")
            return

        # 4. Incremental mapper seeded from PX4 priors.
        print(">> Running Mapper...")
        subprocess.run([
            colmap_bin, "mapper",
            "--database_path", db_path,
            "--image_path", img_dir,
            # "--input_path", model_dir,
            "--output_path", sparse_dir,
            "--Mapper.tri_ignore_two_view_tracks", "0",
        ], check=True)

        mapper_out_0 = os.path.join(sparse_dir, "0")
        mapper_out_root = sparse_dir

        # Depending on COLMAP version and input/output layout, output may be sparse/0
        # or written directly into sparse/. Canonicalize to sparse/0.
        if not has_colmap_model(mapper_out_0) and has_colmap_model(mapper_out_root):
            copy_model_files(mapper_out_root, mapper_out_0)

        if not has_colmap_model(mapper_out_0):
            print("\n❌ Colmap failed to map any models.")
            return

        model_out = mapper_out_0

        point_count = points3d_count_from_bin(model_out)
        if point_count == 0:
            if not prompt_before_triangulation():
                print("\n⏸️ Triangulation skipped by user. Final sparse model remains in:")
                print(f"  {model_out}")
                return

            print("\n⚠️ Incremental mapper produced 0 points. Running point_triangulator fallback...")
            os.makedirs(triangulated_out_dir, exist_ok=True)
            subprocess.run([
                colmap_bin, "point_triangulator",
                "--database_path", db_path,
                "--image_path", img_dir,
                "--input_path", model_out,
                "--output_path", triangulated_out_dir,
                "--clear_points", "1",
                "--Mapper.fix_existing_images", "1",
                "--Mapper.tri_ignore_two_view_tracks", "0",
            ], check=True)

            if has_colmap_model(triangulated_out_dir):
                copy_model_files(triangulated_out_dir, model_out)
                point_count = points3d_count_from_bin(model_out)

        if point_count == 0:
            print(f"\n⚠️ No 3D points were triangulated. Final model (single canonical set) is in:\n  {model_out}")
        else:
            print(f"\n✅ SUCCESS! Colmap sparse reconstruction saved to:\n  {model_out}\n   Points: {point_count}")
        print("To view it, run: colmap gui and click File -> Import Model")

        if point_count and point_count > 0:
            run_dense_reconstruction(colmap_bin, dataset_dir, model_out)

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Colmap failed while running: {colmap_bin}\nError: {e}")
    except FileNotFoundError:
        print("\n❌ COLMAP binary not found. Set COLMAP_BIN to the executable path.")


def run_existing_colmap(dataset_dir: str):
    dataset_dir = os.path.abspath(dataset_dir)
    prior_model_dir = os.path.join(dataset_dir, "sparse", "model")

    if not os.path.exists(dataset_dir):
        print(f"\n❌ Dataset directory does not exist: {dataset_dir}")
        return

    if not os.path.exists(prior_model_dir):
        print(f"\n❌ Missing prior model directory: {prior_model_dir}")
        return

    run_colmap(dataset_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="colmap_dataset", help="Dataset root directory")
    parser.add_argument(
        "--reconstruct-existing",
        action="store_true",
        help="Run COLMAP on the existing dataset without starting ROS collection",
    )
    args = parser.parse_args()

    if args.reconstruct_existing:
        run_existing_colmap(args.dataset_dir)
        return

    if not ROS_AVAILABLE:
        print("\n❌ ROS packages are not available in this shell. Use --reconstruct-existing to run COLMAP on the saved dataset.")
        return

    rclpy.init()
    node = ColmapCollectorNode(dataset_dir=args.dataset_dir, clean_dataset=True)
    
    _shutdown_handled = False

    def handle_shutdown(sig, frame):
        nonlocal _shutdown_handled
        if _shutdown_handled: return
        _shutdown_handled = True
        
        print("\n[Ctrl+C Detected] Stopping data collection...")
        generated_model = node.generate_colmap_model()
        node.destroy_node()
        rclpy.try_shutdown()

        if not generated_model:
            sys.exit(0)

        # Kick off colmap
        run_colmap(args.dataset_dir)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_shutdown)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
