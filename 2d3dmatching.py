from scipy.spatial.transform import Rotation as R
import open3d as o3d
import pandas as pd
import numpy as np
import random
import cv2
import time

from tqdm import tqdm

np.random.seed(148) # do not change this seed
random.seed(1428) # do not change this seed

def average(x):
    return list(np.mean(x,axis=0))

def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc

def solve_p3p(points_3d, points_2d, camera_matrix):
    """
    Solve P3P problem to find camera pose from 3 point correspondences.
    
    Args:
        points_3d: (3, 3) array of 3D points in world coordinates
        points_2d: (3, 2) array of 2D points in image coordinates
        camera_matrix: (3, 3) camera intrinsic matrix
        
    Returns:
        list of (R, t) tuples, where R is 3x3 rotation matrix and t is 3x1 translation vector
    """
    # Normalize 2D points using camera intrinsics
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    # Convert 2D points to normalized camera coordinates (bearing vectors)
    bearing_vectors = np.zeros((3, 3))
    for i in range(3):
        x = (points_2d[i, 0] - cx) / fx
        y = (points_2d[i, 1] - cy) / fy
        bearing_vectors[i] = np.array([x, y, 1.0])
        # Normalize to unit vector
        bearing_vectors[i] = bearing_vectors[i] / np.linalg.norm(bearing_vectors[i])
    
    # Use OpenCV's P3P solver (it's too complex to implement the geometry from scratch)
    # But we're still implementing the RANSAC loop ourselves!
    success, rvec, tvec = cv2.solveP3P(
        points_3d.reshape(3, 1, 3),
        points_2d.reshape(3, 1, 2),
        camera_matrix,
        None,
        flags=cv2.SOLVEPNP_P3P
    )
    
    if not success:
        return []
    
    # cv2.solveP3P can return multiple solutions
    solutions = []
    if rvec is not None:
        # Handle case where multiple solutions are returned
        if len(rvec.shape) == 3:
            for i in range(rvec.shape[0]):
                R_mat, _ = cv2.Rodrigues(rvec[i])
                t_vec = tvec[i].flatten()
                solutions.append((R_mat, t_vec))
        else:
            R_mat, _ = cv2.Rodrigues(rvec)
            t_vec = tvec.flatten()
            solutions.append((R_mat, t_vec))
    
    return solutions


def refine_pose_with_inliers(points_3d, points_2d, R, t, camera_matrix, dist_coeffs):
    """
    Refine the camera pose using all inlier points with iterative optimization.
    
    Args:
        points_3d: (N, 3) array of 3D inlier points
        points_2d: (N, 2) array of 2D inlier points
        R: Initial 3x3 rotation matrix
        t: Initial 3x1 translation vector
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        
    Returns:
        Refined rotation vector and translation vector
    """
    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1)
    
    # Use iterative refinement with all inliers
    rvec, tvec = cv2.solvePnPRefineLM(
        points_3d.reshape(-1, 1, 3),
        points_2d.reshape(-1, 1, 2),
        camera_matrix,
        dist_coeffs,
        rvec,
        tvec
    )
    
    return rvec, tvec


def compute_reprojection_error(points_3d, points_2d, R, t, camera_matrix, dist_coeffs):
    """
    Compute reprojection error for given pose.
    
    Args:
        points_3d: (N, 3) array of 3D points
        points_2d: (N, 2) array of 2D points
        R: 3x3 rotation matrix
        t: 3x1 translation vector
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        
    Returns:
        Array of reprojection errors for each point
    """
    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1)
    
    # Project 3D points to image plane
    projected_points, _ = cv2.projectPoints(
        points_3d.reshape(-1, 1, 3),
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs
    )
    
    projected_points = projected_points.reshape(-1, 2)
    
    # Compute Euclidean distance between projected and observed points
    errors = np.linalg.norm(projected_points - points_2d, axis=1)
    
    return errors


def pnp_ransac(points_3d, points_2d, camera_matrix, dist_coeffs, 
               reprojection_threshold=8.0, max_iterations=1000, confidence=0.99):
    """
    P3P + RANSAC implementation for robust camera pose estimation.
    
    Args:
        points_3d: (N, 3) array of 3D points in world coordinates
        points_2d: (N, 2) array of corresponding 2D points in image coordinates
        camera_matrix: (3, 3) camera intrinsic matrix
        dist_coeffs: (4,) or (5,) array of distortion coefficients
        reprojection_threshold: Maximum reprojection error for inliers (pixels)
        max_iterations: Maximum RANSAC iterations
        confidence: Desired confidence level (0.99 = 99%)
        
    Returns:
        success: Boolean indicating if pose was found
        rvec: Rotation vector (3x1)
        tvec: Translation vector (3x1)
        inlier_indices: Indices of inlier correspondences
    """
    n_points = len(points_3d)
    
    if n_points < 3:
        return False, None, None, None
    
    best_inliers = []
    best_R = None
    best_t = None
    best_num_inliers = 0
    
    # Adaptive RANSAC: adjust iterations based on inlier ratio
    iterations = 0
    
    while iterations < max_iterations:
        # Step 1: Randomly sample 3 points
        sample_indices = np.random.choice(n_points, 3, replace=False)
        sample_3d = points_3d[sample_indices]
        sample_2d = points_2d[sample_indices]
        
        # Step 2: Solve P3P for the 3 sampled points
        solutions = solve_p3p(sample_3d, sample_2d, camera_matrix)
        
        if not solutions:
            iterations += 1
            continue
        
        # Step 3: Evaluate each solution (P3P can return multiple solutions)
        for R, t in solutions:
            # Compute reprojection errors for all points
            errors = compute_reprojection_error(
                points_3d, points_2d, R, t, camera_matrix, dist_coeffs
            )
            
            # Step 4: Count inliers
            inlier_mask = errors < reprojection_threshold
            inlier_indices = np.where(inlier_mask)[0]
            num_inliers = len(inlier_indices)
            
            # Step 5: Update best model if this is better
            if num_inliers > best_num_inliers:
                best_num_inliers = num_inliers
                best_R = R
                best_t = t
                best_inliers = inlier_indices
                
                # Adaptive RANSAC: update max_iterations based on inlier ratio
                inlier_ratio = num_inliers / n_points
                if inlier_ratio > 0:
                    # Formula: N = log(1-p) / log(1-w^s)
                    # where p=confidence, w=inlier_ratio, s=sample_size (3 for P3P)
                    num_iterations_needed = np.log(1 - confidence) / np.log(1 - inlier_ratio**3)
                    max_iterations = min(max_iterations, int(num_iterations_needed) + 1)
        
        iterations += 1
    
    # Check if we found a valid solution
    if best_num_inliers < 4:  # Need at least 4 inliers for a stable solution
        return False, None, None, None
    
    # Step 6: Refine pose using all inliers
    inlier_3d = points_3d[best_inliers]
    inlier_2d = points_2d[best_inliers]
    
    rvec_refined, tvec_refined = refine_pose_with_inliers(
        inlier_3d, inlier_2d, best_R, best_t, camera_matrix, dist_coeffs
    )
    
    return True, rvec_refined, tvec_refined, best_inliers


def pnpsolver_custom(query, model, cameraMatrix=0, distortion=0):
    """
    Custom PnP solver using descriptor matching + P3P + RANSAC.
    This is the DROP-IN REPLACEMENT for your original pnpsolver.
    
    Args:
        query: tuple of (kp_query, desc_query)
        model: tuple of (kp_model, desc_model)
        
    Returns:
        retval: Success flag
        rvec: Rotation vector
        tvec: Translation vector
        inliers: Indices of inlier matches
    """
    kp_query, desc_query = query
    kp_model, desc_model = model
    
    # Camera parameters
    cameraMatrix = np.array([[1868.27, 0, 540], [0, 1869.18, 960], [0, 0, 1]])
    distCoeffs = np.array([0.0847023, -0.192929, -0.000201144, -0.000725352])
    
    # Step 1: Descriptor Matching using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(desc_query, desc_model, k=2)
    
    # Step 2: Apply Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < 4:
        return False, None, None, None
    
    # Step 3: Extract matched keypoints
    query_pts = np.array([kp_query[m.queryIdx] for m in good_matches])
    model_pts = np.array([kp_model[m.trainIdx] for m in good_matches])
    
    # Step 4: Apply our custom P3P + RANSAC
    retval, rvec, tvec, inlier_indices = pnp_ransac(
        model_pts,  # 3D points
        query_pts,  # 2D points
        cameraMatrix,
        distCoeffs,
        reprojection_threshold=8.0,
        max_iterations=1000,
        confidence=0.99
    )
    
    if not retval:
        return False, None, None, None
    
    # Convert inlier_indices to the format expected by the caller
    # (indices into the good_matches array)
    inliers = inlier_indices.reshape(-1, 1) if inlier_indices is not None else None
    
    return retval, rvec, tvec, inliers
def pnpsolver(query, model, cameraMatrix=0, distortion=0):
    kp_query, desc_query = query
    kp_model, desc_model = model
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])
    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])

    # Descriptor Matching using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(desc_query, desc_model, k=2)
    
    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:  # Ratio threshold
                good_matches.append(m)
    
    # Need at least 4 points for PnP
    if len(good_matches) < 4:
        return None, None, None, None
    
    # Extract matched keypoints
    query_pts = np.array([kp_query[m.queryIdx] for m in good_matches])
    model_pts = np.array([kp_model[m.trainIdx] for m in good_matches])
    
    # Solve PnP using RANSAC for robustness
    retval, rvec, tvec, inliers = cv2.solvePnPRansac(
        model_pts,  # 3D points in world coordinates
        query_pts,  # 2D points in image coordinates
        cameraMatrix,
        distCoeffs,
        reprojectionError=8.0,
        confidence=0.99,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    return retval, rvec, tvec, inliers
# def pnpsolver(query,model,cameraMatrix=0,distortion=0):
#     kp_query, desc_query = query
#     kp_model, desc_model = model
#     cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])
#     distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])

#     # TODO: solve PnP problem using OpenCV
#     # Hint: you may use "Descriptors Matching and ratio test" first
#     return None, None, None, None

def rotation_error(R1, R2):
    # R1 and R2 are quaternions in format [QX, QY, QZ, QW] or [x, y, z, w]
    # Convert quaternions to rotation matrices
    
    # Handle the input format - could be 2D array with shape (1, 4) or 1D array
    if R1.ndim == 2:
        R1 = R1.flatten()
    if R2.ndim == 2:
        R2 = R2.flatten()
    
    # scipy expects quaternion in [x, y, z, w] format
    # Your data has [QX, QY, QZ, QW] which is the same format
    rot1 = R.from_quat(R1)
    rot2 = R.from_quat(R2)
    
    # Calculate relative rotation: R_rel = R2^T * R1
    # This gives us the rotation difference between the two poses
    rot_diff = rot2.inv() * rot1
    
    # Convert to angle-axis representation and get the angle
    # The magnitude of the rotation vector is the rotation angle in radians
    rot_vec = rot_diff.as_rotvec()
    angle_rad = np.linalg.norm(rot_vec)
    
    # Convert to degrees
    angle_deg = np.rad2deg(angle_rad)
    
    return angle_deg

# def rotation_error(R1, R2):
#     #TODO: calculate rotation error
#     return None

def translation_error(t1, t2):
    # Handle 2D arrays (shape (1,3)) by flattening
    if t1.ndim == 2:
        t1 = t1.flatten()
    if t2.ndim == 2:
        t2 = t2.flatten()
    
    # Calculate the Euclidean distance between the two translation vectors
    return np.linalg.norm(t1 - t2)

# def translation_error(t1, t2):
#     #TODO: calculate translation error
#     return None

def visualization(Camera2World_Transform_Matrixs, points3D_df):
    
    # Load point cloud
    xyz = np.vstack(points3D_df['XYZ'])
    rgb = np.vstack(points3D_df['RGB'])/255
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Camera Poses and Point Cloud")
    vis.add_geometry(pcd)
    
    # Add coordinate axes at origin
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    vis.add_geometry(axes)
    
    # Visualize each camera pose
    for i, c2w in enumerate(Camera2World_Transform_Matrixs):
        # Create a small coordinate frame for each camera
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        camera_frame.transform(c2w)
        vis.add_geometry(camera_frame)
        
        # Create a camera frustum for better visualization (optional)
        # Extract camera position
        camera_pos = c2w[:3, 3]
        
        # Create a small sphere at camera position
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        sphere.translate(camera_pos)
        sphere.paint_uniform_color([1, 0, 0])  # Red for camera
        vis.add_geometry(sphere)
    
    # Set viewing angle
    ctr = vis.get_view_control()
    ctr.set_zoom(0.5)
    
    print("\nVisualization Controls:")
    print("- Mouse: Rotate view")
    print("- Scroll: Zoom")
    print("- Ctrl+Mouse: Pan")
    print("- Q or ESC: Close window")
    
    vis.run()
    vis.destroy_window()

# def visualization(Camera2World_Transform_Matrixs, points3D_df):
#     #TODO: visualize the camera pose
#     pass


if __name__ == "__main__":
    # Load data
    images_df = pd.read_pickle("data/images.pkl")
    train_df = pd.read_pickle("data/train.pkl")
    points3D_df = pd.read_pickle("data/points3D.pkl")
    point_desc_df = pd.read_pickle("data/point_desc.pkl")

    # Process model descriptors
    desc_df = average_desc(train_df, points3D_df)
    kp_model = np.array(desc_df["XYZ"].to_list())
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)


    IMAGE_ID_LIST = images_df[images_df['NAME'].str.contains('valid')]['IMAGE_ID'].tolist()[:10]

    print(f"Processing {len(IMAGE_ID_LIST)} validation images...")

    r_list = []
    t_list = []
    rotation_error_list = []
    translation_error_list = []

    for idx in tqdm(IMAGE_ID_LIST):
        # Load quaery image
        # fname = (images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values[0]
        # rimg = cv2.imread("data/frames/" + fname, cv2.IMREAD_GRAYSCALE)

        # Load query keypoints and descriptors
        points = point_desc_df.loc[point_desc_df["IMAGE_ID"] == idx]
        kp_query = np.array(points["XY"].to_list())
        desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)

        # Find correspondance and solve pnp
        retval, rvec, tvec, inliers = pnpsolver_custom((kp_query, desc_query), (kp_model, desc_model))
        # rotq = R.from_rotvec(rvec.reshape(1,3)).as_quat() # Convert rotation vector to quaternion
        # tvec = tvec.reshape(1,3) # Reshape translation vector

        if not retval:
            print(f"Warning: PnP failed for image {idx}")
            continue

        r_list.append(rvec)
        t_list.append(tvec)

        rotq_pred = R.from_rotvec(rvec.reshape(1,3)).as_quat() # Convert rotation vector to quaternion
        tvec_pred = tvec.reshape(1,3) # Reshape translation vector

        # Get camera pose groudtruth
        ground_truth = images_df.loc[images_df["IMAGE_ID"]==idx]
        rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
        tvec_gt = ground_truth[["TX","TY","TZ"]].values

        # Calculate error
        r_error = rotation_error(rotq_pred, rotq_gt)
        t_error = translation_error(tvec_pred, tvec_gt)
        rotation_error_list.append(r_error)
        translation_error_list.append(t_error)

    # TODO: calculate median of relative rotation angle differences and translation differences and print them

    # Calculate median of errors
    median_rotation_error = np.median(rotation_error_list)
    median_translation_error = np.median(translation_error_list)

    print("\n" + "="*50)
    print("POSE ESTIMATION RESULTS")
    print("="*50)
    print(f"Median Rotation Error: {median_rotation_error:.4f} degrees")
    print(f"Median Translation Error: {median_translation_error:.4f} units")
    print("="*50)

    # TODO: result visualization
    Camera2World_Transform_Matrixs = []
    for r, t in zip(r_list, t_list):
        R_mat, _ = cv2.Rodrigues(r)
        R_cam_to_world = R_mat.T
        t_cam_to_world = -R_cam_to_world @ t.flatten()
        # TODO: calculate camera pose in world coordinate system
        c2w = np.eye(4)
        c2w[:3, :3] = R_cam_to_world
        c2w[:3, 3] = t_cam_to_world
        Camera2World_Transform_Matrixs.append(c2w)
    visualization(Camera2World_Transform_Matrixs, points3D_df)

# if __name__ == "__main__":
#     images_df = pd.read_pickle("data/images.pkl")
#     train_df = pd.read_pickle("data/train.pkl")
#     points3D_df = pd.read_pickle("data/points3D.pkl")
#     point_desc_df = pd.read_pickle("data/point_desc.pkl")

#     print("Images:")
#     print(images_df.columns)
#     print("Train:")
#     print(train_df.columns)
#     print("Points3D:")
#     print(points3D_df.columns)
#     print("Descriptors:")
#     print(point_desc_df.columns)