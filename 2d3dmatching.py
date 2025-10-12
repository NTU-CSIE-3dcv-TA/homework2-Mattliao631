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

def solve_p3p_grunert(points_3d, points_2d, camera_matrix):
    """
    Solve P3P problem using Grunert's method (purely geometric solution).
    This is a from-scratch implementation without using cv2.solveP3P.
    
    Args:
        points_3d: (3, 3) array of 3D points in world coordinates [A, B, C]
        points_2d: (3, 2) array of 2D points in image coordinates [a, b, c]
        camera_matrix: (3, 3) camera intrinsic matrix
        
    Returns:
        list of (R, t) tuples, where R is 3x3 rotation matrix and t is 3x1 translation vector
    """
    # Step 1: Normalize 2D points to get bearing vectors (unit vectors from camera center)
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    # Convert to normalized camera coordinates
    bearing_vectors = np.zeros((3, 3))
    for i in range(3):
        x = (points_2d[i, 0] - cx) / fx
        y = (points_2d[i, 1] - cy) / fy
        bearing_vectors[i] = np.array([x, y, 1.0])
        # Normalize to unit vector
        bearing_vectors[i] = bearing_vectors[i] / np.linalg.norm(bearing_vectors[i])
    
    # Step 2: Compute distances between 3D points
    # Let A, B, C be the 3D points
    A, B, C = points_3d[0], points_3d[1], points_3d[2]
    
    # Distances between 3D world points
    a = np.linalg.norm(B - C)  # distance BC
    b = np.linalg.norm(A - C)  # distance AC
    c = np.linalg.norm(A - B)  # distance AB
    
    # Step 3: Compute angles between bearing vectors (cosines)
    # Let f_a, f_b, f_c be the bearing vectors (normalized image points)
    f_a, f_b, f_c = bearing_vectors[0], bearing_vectors[1], bearing_vectors[2]
    
    # Cosines of angles between bearing vectors
    cos_ab = np.dot(f_a, f_b)  # angle between rays to A and B
    cos_ac = np.dot(f_a, f_c)  # angle between rays to A and C
    cos_bc = np.dot(f_b, f_c)  # angle between rays to B and C
    
    # Step 4: Solve for distances from camera to 3D points
    # Using the law of cosines in the triangles formed by camera center and 3D points
    # We need to solve: x^2 + y^2 - 2*x*y*cos_ab = c^2
    #                  x^2 + z^2 - 2*x*z*cos_ac = b^2
    #                  y^2 + z^2 - 2*y*z*cos_bc = a^2
    # where x, y, z are distances from camera to A, B, C respectively
    
    # This leads to a quartic equation. We'll use a simplified approach:
    # Use Wu's P3P method (more numerically stable)
    
    solutions = solve_p3p_wu(A, B, C, f_a, f_b, f_c, a, b, c, cos_ab, cos_ac, cos_bc)
    
    return solutions


def solve_p3p_kneip(points_3d, points_2d, camera_matrix):
    """
    Solve P3P problem using Kneip's method (more numerically stable).
    Based on: "A Novel Parametrization of the P3P-Problem for a Direct Computation of Absolute Camera Position and Orientation"
    
    Args:
        points_3d: (3, 3) array of 3D points in world coordinates
        points_2d: (3, 2) array of 2D points in image coordinates
        camera_matrix: (3, 3) camera intrinsic matrix
        
    Returns:
        list of (R, t) tuples
    """
    # Normalize 2D points to get bearing vectors
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    # Convert to normalized camera coordinates and create unit vectors
    bearing_vectors = []
    for i in range(3):
        x = (points_2d[i, 0] - cx) / fx
        y = (points_2d[i, 1] - cy) / fy
        vec = np.array([x, y, 1.0])
        vec = vec / np.linalg.norm(vec)
        bearing_vectors.append(vec)
    
    bearing_vectors = np.array(bearing_vectors)
    
    # Get 3D points
    P1, P2, P3 = points_3d[0], points_3d[1], points_3d[2]
    f1, f2, f3 = bearing_vectors[0], bearing_vectors[1], bearing_vectors[2]
    
    # Compute intermediate values
    # Distances between 3D points
    d12 = np.linalg.norm(P2 - P1)
    d13 = np.linalg.norm(P3 - P1)
    d23 = np.linalg.norm(P3 - P2)

    if d12 < 1e-6:
        return []
    
    # Cosines between bearing vectors
    cos_alpha = np.dot(f1, f2)
    cos_beta = np.dot(f1, f3)
    cos_gamma = np.dot(f2, f3)
    
    # Clamp cosines to valid range to avoid numerical issues
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    cos_beta = np.clip(cos_beta, -1.0, 1.0)
    cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
    
    # Build the polynomial coefficients using Kneip's formulation
    # This is more numerically stable than Grunert's method
    a = (d13 / d12) ** 2
    b = (d23 / d12) ** 2
    
    # Construct polynomial coefficients
    a2 = a * a
    a3 = a2 * a
    a4 = a3 * a
    
    b2 = b * b
    b3 = b2 * b
    b4 = b3 * b
    
    cos_alpha2 = cos_alpha * cos_alpha
    cos_beta2 = cos_beta * cos_beta
    cos_gamma2 = cos_gamma * cos_gamma
    
    # Coefficients of 4th degree polynomial
    p4 = -4 * b + 4 * cos_gamma2 * b - a + 2 * a * cos_gamma + 1
    
    p3 = 8 * b - 8 * cos_gamma * cos_alpha * b - 4 * cos_gamma2 * b - 4 * a * b * cos_beta + 4 * a * b * cos_gamma * cos_alpha + 2 * a - 4 * a * cos_gamma - 2
    
    p2 = -12 * b + 4 * cos_gamma * cos_alpha * b + 8 * cos_gamma2 * b + 4 * a * b - 4 * cos_gamma2 * a * b - 2 * a2 + 4 * a * b * cos_beta - 8 * a * b * cos_gamma * cos_alpha + 4 * cos_gamma2 * a * b + 2 * a + 4 * a * cos_gamma + 1
    
    p1 = 4 * b - 8 * cos_gamma * cos_alpha * b - 4 * cos_gamma2 * b - 4 * a * b + 4 * cos_gamma2 * a * b + 4 * a * b * cos_gamma * cos_alpha + 2 * a2 - 4 * a - 2 * a * cos_gamma
    
    p0 = -4 * cos_gamma2 * b + 4 * cos_gamma2 * a * b + a2
    
    # Solve the 4th degree polynomial
    coeffs = [p4, p3, p2, p1, p0]
    
    # Filter out very small coefficients to improve numerical stability
    coeffs = [c if abs(c) > 1e-10 else 0 for c in coeffs]
    
    try:
        roots = np.roots(coeffs)
    except:
        return []
    
    solutions = []
    
    # Process each root
    for root in roots:
        if not np.isreal(root):
            continue
        
        v = np.real(root)
        if v <= 0:
            continue
        
        # Compute u from v
        u_num = (-1 + b - v * b + a * v - a * cos_gamma * v + v * v * cos_gamma * cos_alpha)
        u_den = (v - 1 - cos_beta * cos_alpha + v * cos_beta * cos_alpha + a * cos_beta - a * v * cos_beta)
        
        if abs(u_den) < 1e-10:
            continue
        
        u = u_num / u_den
        
        if u <= 0:
            continue
        
        # Compute distances
        s1 = d12 / np.sqrt(1 + u*u - 2*u*cos_alpha)
        s2 = u * s1
        s3 = v * s1
        
        if s1 <= 0 or s2 <= 0 or s3 <= 0:
            continue
        
        # Compute 3D points in camera frame
        X1 = s1 * f1
        X2 = s2 * f2
        X3 = s3 * f3
        
        # Compute pose using Procrustes
        points_world = np.array([P1, P2, P3])
        points_cam = np.array([X1, X2, X3])
        
        R_mat, t_vec = compute_absolute_pose(points_world, points_cam)
        
        if R_mat is not None:
            solutions.append((R_mat, t_vec))
    
    return solutions


def compute_absolute_pose(points_world, points_cam):
    """
    Compute absolute camera pose using Procrustes/Kabsch algorithm.
    Finds R and t such that: points_cam = R @ points_world + t
    """
    # Center the point sets
    centroid_world = np.mean(points_world, axis=0)
    centroid_cam = np.mean(points_cam, axis=0)
    
    centered_world = points_world - centroid_world
    centered_cam = points_cam - centroid_cam
    
    # Compute the cross-covariance matrix
    H = centered_world.T @ centered_cam
    
    # SVD
    try:
        U, S, Vt = np.linalg.svd(H)
    except:
        return None, None
    
    # Compute rotation
    R = Vt.T @ U.T
    
    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute translation
    t = centroid_cam - R @ centroid_world
    
    return R, t


def project_points(points_3d, R, t, camera_matrix):
    """
    Project 3D points to image plane.
    """
    # Transform to camera coordinates
    points_cam = (R @ points_3d.T).T + t
    
    # Avoid division by zero
    points_cam[:, 2] = np.maximum(points_cam[:, 2], 1e-6)
    
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    # Project to image
    u = fx * points_cam[:, 0] / points_cam[:, 2] + cx
    v = fy * points_cam[:, 1] / points_cam[:, 2] + cy
    
    return np.column_stack([u, v])


def compute_reprojection_error(points_3d, points_2d, R, t, camera_matrix):
    """
    Compute reprojection errors.
    """
    projected = project_points(points_3d, R, t, camera_matrix)
    errors = np.linalg.norm(projected - points_2d, axis=1)
    return errors


def refine_pose_pnp(points_3d, points_2d, R_init, t_init, camera_matrix):
    """
    Refine pose using OpenCV's iterative method (allowed for refinement).
    """
    rvec_init, _ = cv2.Rodrigues(R_init)
    tvec_init = t_init.reshape(3, 1)
    
    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            points_3d.reshape(-1, 1, 3),
            points_2d.reshape(-1, 1, 2),
            camera_matrix,
            None,  # No distortion
            rvec_init,
            tvec_init
        )
        return rvec, tvec
    except:
        return rvec_init, tvec_init


def pnp_ransac_custom(points_3d, points_2d, camera_matrix, 
                      reprojection_threshold=8.0, max_iterations=2000, 
                      confidence=0.99, min_inliers=6):
    """
    Custom RANSAC + P3P implementation.
    
    Args:
        points_3d: (N, 3) array of 3D points
        points_2d: (N, 2) array of 2D points
        camera_matrix: (3, 3) intrinsic matrix
        reprojection_threshold: inlier threshold in pixels
        max_iterations: maximum RANSAC iterations
        confidence: desired confidence level
        min_inliers: minimum number of inliers required
        
    Returns:
        success, rvec, tvec, inlier_indices
    """
    n_points = len(points_3d)
    
    if n_points < 3:
        return False, None, None, None
    
    best_inliers = []
    best_R = None
    best_t = None
    best_num_inliers = 0
    
    iterations = 0
    
    np.random.seed(int(np.random.random() * 10000))  # Random seed for variety
    
    while iterations < max_iterations:
        # Sample 3 points randomly
        if n_points == 3:
            sample_idx = [0, 1, 2]
        else:
            sample_idx = np.random.choice(n_points, 3, replace=False)
        
        sample_3d = points_3d[sample_idx]
        sample_2d = points_2d[sample_idx]
        
        # Solve P3P
        try:
            solutions = solve_p3p_kneip(sample_3d, sample_2d, camera_matrix)
        except:
            iterations += 1
            continue
        
        if not solutions:
            iterations += 1
            continue
        
        # Test each solution
        for R, t in solutions:
            # Compute errors for all points
            try:
                errors = compute_reprojection_error(points_3d, points_2d, R, t, camera_matrix)
            except:
                continue
            
            # Count inliers
            inlier_mask = errors < reprojection_threshold
            inlier_idx = np.where(inlier_mask)[0]
            num_inliers = len(inlier_idx)
            
            # Update best model
            if num_inliers > best_num_inliers:
                best_num_inliers = num_inliers
                best_R = R
                best_t = t
                best_inliers = inlier_idx
                
                # Adaptive RANSAC
                if num_inliers > 3:
                    inlier_ratio = num_inliers / n_points
                    if inlier_ratio > 0.05:
                        try:
                            adaptive_iters = np.log(1 - confidence) / np.log(1 - inlier_ratio**3)
                            max_iterations = min(max_iterations, int(adaptive_iters) + 100)
                        except:
                            pass
        
        iterations += 1
        
        # Early termination if we have enough inliers
        if best_num_inliers > max(min_inliers, int(0.3 * n_points)):
            break
    
    # Check if solution is valid
    if best_num_inliers < min_inliers:
        return False, None, None, None
    
    # Refine with all inliers
    inlier_3d = points_3d[best_inliers]
    inlier_2d = points_2d[best_inliers]
    
    try:
        rvec, tvec = refine_pose_pnp(inlier_3d, inlier_2d, best_R, best_t, camera_matrix)
    except:
        rvec, _ = cv2.Rodrigues(best_R)
        tvec = best_t.reshape(3, 1)
    
    return True, rvec, tvec, best_inliers


def pnpsolver_custom(query, model, cameraMatrix=0, distortion=0):
    """
    Custom PnP solver: Descriptor Matching + P3P + RANSAC.
    P3P and RANSAC are implemented from scratch!
    
    Args:
        query: (kp_query, desc_query) 
        model: (kp_model, desc_model)
        
    Returns:
        success, rvec, tvec, inliers
    """
    kp_query, desc_query = query
    kp_model, desc_model = model
    
    # Camera intrinsics
    cameraMatrix = np.array([[1868.27, 0, 540], [0, 1869.18, 960], [0, 0, 1]], dtype=np.float64)
    
    # Descriptor matching
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(desc_query, desc_model, k=2)
    
    # Ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < 6:
        return False, None, None, None
    
    # Extract correspondences
    query_pts = np.array([kp_query[m.queryIdx] for m in good_matches], dtype=np.float64)
    model_pts = np.array([kp_model[m.trainIdx] for m in good_matches], dtype=np.float64)
    
    # Run custom P3P + RANSAC
    success, rvec, tvec, inlier_idx = pnp_ransac_custom(
        model_pts,
        query_pts,
        cameraMatrix,
        reprojection_threshold=8.0,
        max_iterations=2000,
        confidence=0.99,
        min_inliers=6
    )
    
    if not success:
        return False, None, None, None
    
    inliers = inlier_idx.reshape(-1, 1) if inlier_idx is not None else None
    
    return True, rvec, tvec, inliers
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


    IMAGE_ID_LIST = images_df[images_df['NAME'].str.contains('valid')]['IMAGE_ID'].tolist()

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