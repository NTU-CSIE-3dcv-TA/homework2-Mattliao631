import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import os

def create_cube_points(n_points_per_edge=10):
    """
    Create a dense point cloud of the cube surface.
    More points = better visual quality.
    """
    points = []
    colors = []
    
    # Define 6 faces with different colors
    faces = [
        # Face, color (RGB)
        ([[0,0,0], [1,0,0], [1,1,0], [0,1,0]], [255, 0, 0]),    # Front - Red
        ([[0,0,1], [1,0,1], [1,1,1], [0,1,1]], [0, 255, 0]),    # Back - Green
        ([[0,0,0], [0,0,1], [0,1,1], [0,1,0]], [0, 0, 255]),    # Left - Blue
        ([[1,0,0], [1,0,1], [1,1,1], [1,1,0]], [255, 255, 0]),  # Right - Yellow
        ([[0,0,0], [1,0,0], [1,0,1], [0,0,1]], [255, 0, 255]),  # Bottom - Magenta
        ([[0,1,0], [1,1,0], [1,1,1], [0,1,1]], [0, 255, 255]),  # Top - Cyan
    ]
    
    for face_corners, color in faces:
        # Create grid of points on this face
        for i in range(n_points_per_edge):
            for j in range(n_points_per_edge):
                u = i / (n_points_per_edge - 1)
                v = j / (n_points_per_edge - 1)
                
                # Bilinear interpolation
                p0, p1, p2, p3 = face_corners
                point = (1-u)*(1-v)*np.array(p0) + u*(1-v)*np.array(p1) + \
                        u*v*np.array(p2) + (1-u)*v*np.array(p3)
                
                points.append(point)
                colors.append(color)
    
    return np.array(points), np.array(colors)


def painter_algorithm(points_3d, colors, depths):
    """
    Sort points by depth (painter's algorithm).
    Draw furthest points first.
    """
    # Sort by depth (furthest first)
    sorted_indices = np.argsort(-depths)  # Negative for descending order
    return points_3d[sorted_indices], colors[sorted_indices]


def draw_cube_on_image(image, rvec, tvec, cube_vertices, cube_transform_mat, camera_matrix, dist_coeffs):
    """
    Draw the virtual cube on the image using the painter's algorithm.
    
    Args:
        image: Input image
        rvec: Camera rotation vector (world to camera)
        tvec: Camera translation vector (world to camera)
        cube_vertices: Cube vertices in local coordinates
        cube_transform_mat: 3x4 transformation matrix (scale, rotation, translation of cube)
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        
    Returns:
        Image with cube drawn on it
    """
    img_with_cube = image.copy()
    
    # Generate dense cube points
    cube_points_local, cube_colors = create_cube_points(n_points_per_edge=30)
    
    # Apply cube transformation (scale, rotate, translate in world)
    # cube_transform_mat is [3x4]: [scale*R | t]
    cube_points_local_homo = np.hstack([cube_points_local, np.ones((len(cube_points_local), 1))])
    cube_points_world = (cube_transform_mat @ cube_points_local_homo.T).T
    
    # Transform world points to camera coordinates
    R_cam, _ = cv2.Rodrigues(rvec)
    t_cam = tvec.flatten()
    
    cube_points_cam = (R_cam @ cube_points_world.T).T + t_cam
    
    # Get depths (z-coordinates in camera frame)
    depths = cube_points_cam[:, 2]
    
    # Filter points behind camera
    valid_mask = depths > 0
    cube_points_cam = cube_points_cam[valid_mask]
    cube_colors_filtered = cube_colors[valid_mask]
    depths = depths[valid_mask]
    
    if len(cube_points_cam) == 0:
        return img_with_cube
    
    # Apply painter's algorithm (sort by depth)
    cube_points_sorted, colors_sorted = painter_algorithm(
        cube_points_cam, cube_colors_filtered, depths
    )
    
    # Project to image plane
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    u = fx * cube_points_sorted[:, 0] / cube_points_sorted[:, 2] + cx
    v = fy * cube_points_sorted[:, 1] / cube_points_sorted[:, 2] + cy
    
    # Draw points on image
    for i in range(len(u)):
        x, y = int(round(u[i])), int(round(v[i]))
        
        # Check if point is within image bounds
        if 0 <= x < img_with_cube.shape[1] and 0 <= y < img_with_cube.shape[0]:
            color = tuple(int(c) for c in colors_sorted[i])
            # Draw a small circle for each point (size 2-3 pixels)
            cv2.circle(img_with_cube, (x, y), 6, color, -1)
    
    return img_with_cube


def generate_ar_video(image_ids, images_df, pose_results, cube_transform_mat, 
                      output_video='ar_output.mp4', fps=30):
    """
    Generate AR video with virtual cube overlay.
    
    Args:
        image_ids: List of image IDs to process
        images_df: DataFrame with image info
        pose_results: List of (rvec, tvec) for each image
        cube_transform_mat: Cube transformation matrix
        output_video: Output video filename
        fps: Frames per second
    """
    # Camera parameters
    camera_matrix = np.array([[1868.27, 0, 540], [0, 1869.18, 960], [0, 0, 1]])
    dist_coeffs = np.array([0.0847023, -0.192929, -0.000201144, -0.000725352])
    
    # Get cube vertices (not really needed for point-based rendering, but kept for compatibility)
    cube_vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ])
    
    # Initialize video writer
    first_image_name = images_df.loc[images_df["IMAGE_ID"] == image_ids[0]]["NAME"].values[0]
    first_image = cv2.imread(f"data/frames/{first_image_name}")
    height, width = first_image.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    print(f"Generating AR video: {output_video}")
    print(f"Resolution: {width}x{height}, FPS: {fps}")
    
    for idx, (image_id, (rvec, tvec)) in enumerate(tqdm(zip(image_ids, pose_results), total=len(image_ids))):
        # Load image
        image_name = images_df.loc[images_df["IMAGE_ID"] == image_id]["NAME"].values[0]
        image = cv2.imread(f"data/frames/{image_name}")
        
        if image is None:
            print(f"Warning: Could not load image {image_name}")
            continue
        
        # Draw cube on image
        image_with_cube = draw_cube_on_image(
            image, rvec, tvec, cube_vertices, 
            cube_transform_mat, camera_matrix, dist_coeffs
        )
        
        # Write frame
        out.write(image_with_cube)
    
    out.release()
    print(f"✓ Video saved to: {output_video}")


if __name__ == "__main__":
    # Load data
    images_df = pd.read_pickle("data/images.pkl")
    
    # Load cube transformation (run transform_cube.py first to generate this!)
    if not os.path.exists("cube_transform_mat.npy"):
        print("ERROR: cube_transform_mat.npy not found!")
        print("Run transform_cube.py first to position the cube.")
        exit(1)
    
    cube_transform_mat = np.load("cube_transform_mat.npy")
    print(f"Loaded cube transformation matrix:")
    print(cube_transform_mat)
    
    # Load your pose estimation results
    # You need to save these from your main script!
    # Example: np.save('pose_results.npy', list(zip(r_list, t_list)))
    
    if not os.path.exists("pose_results.npy"):
        print("ERROR: pose_results.npy not found!")
        print("Run your pose estimation script first and save results.")
        exit(1)
    
    pose_data = np.load("pose_results.npy", allow_pickle=True).item()
    image_ids = pose_data['image_ids']
    pose_results = list(zip(pose_data['rvecs'], pose_data['tvecs']))
    
    print(f"Loaded {len(pose_results)} camera poses")
    
    # Generate AR video
    generate_ar_video(
        image_ids, 
        images_df, 
        pose_results, 
        cube_transform_mat,
        output_video='ar_cube_video.mp4',
        fps=15
    )