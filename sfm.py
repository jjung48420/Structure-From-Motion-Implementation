import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from multiprocessing import Pool


def feature_extraction(rgbs):
    gray = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in rgbs]
    sift = cv2.SIFT_create()
    kps, dess = [], []
    imgs_with_kps = []

    for i in range(len(gray)):
        kp, des = sift.detectAndCompute(gray[i], None) # keypoints and descriptors
        kps.append(kp)
        dess.append(des)
        imgs_with_kps.append(cv2.drawKeypoints(
            rgbs[i], kps[i], None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        ))

    return kps, dess, imgs_with_kps

def feature_matching_FLANN(rgbs, kps, dess):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    length = len(rgbs)-1
    good_matches = np.empty((length), dtype=object) # stores good matches
    matching_points = [] # stores pair of matching points coordinates
    matches_count = []

    for i in range(length):
        pts1 = []
        pts2 = []
        j = i + 1
        matches = flann.knnMatch(np.asarray(dess[i],np.float32), np.asarray(dess[j],np.float32), k=2)
        good = []
        for k, (m,n) in enumerate(matches):
            if m.distance < 0.75*n.distance:
                good.append([m])
                pts1.append(kps[i][m.queryIdx].pt)
                pts2.append(kps[j][m.trainIdx].pt)

        print("\nimage{} and image{} have {} matching points\n".format(i, j, len(good)))
            
       # matching points
        matching_points.append([pts1, pts2])
        good_matches[i] = good
        matches_count.append(len(good))

    return good_matches, matching_points, matches_count

def find_Rotation_and_translation(matching_points, K, dist):  
    pts1, pts2 = matching_points
    p1, p2 = np.ascontiguousarray(pts1, np.float32), np.ascontiguousarray(pts2, np.float32)
    pts_1_norm = cv2.undistortPoints(np.expand_dims(p1, axis=1), cameraMatrix=K, distCoeffs=dist)
    pts_2_norm = cv2.undistortPoints(np.expand_dims(p2, axis=1), cameraMatrix=K, distCoeffs=dist)

    E, mask = cv2.findEssentialMat(pts_1_norm, pts_2_norm, focal=1.0, pp=(0.,0.),
                                method=cv2.RANSAC, prob=0.999, threshold=3.0/500.0)
    
    inliers1 = pts_1_norm[mask.ravel()==1]
    inliers2 = pts_2_norm[mask.ravel()==1]

    # finding Rotation and Translation using Essential matrix
    points, R_est, T_est, mask_pose = cv2.recoverPose(E, inliers1, inliers2)

    return R_est, T_est

def Projection_Matrix(K, R1, T1, R2, T2):
    P1 = K@np.hstack((R1, T1))
    P2 = K@np.hstack((R2, T2))
    return P1, P2

def geometric_verification(rgb1, rgb2, R_, T_, K, dist, w, h):
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1=K, distCoeffs1=dist, cameraMatrix2=K, distCoeffs2=dist,
        imageSize=(w,h), R=R_, T=T_, flags=cv2.CALIB_ZERO_DISPARITY, alpha=1
    )
    map1x, map1y = cv2.initUndistortRectifyMap(
        cameraMatrix=K, distCoeffs=dist, R=R1, newCameraMatrix=P1, size=(w,h), m1type=cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        cameraMatrix=K, distCoeffs=dist, R=R2, newCameraMatrix=P2, size=(w,h), m1type=cv2.CV_32FC1
    )

    img1_rect = cv2.remap(rgb1, map1x, map1y, cv2.INTER_LINEAR)
    img2_rect = cv2.remap(rgb2, map2x, map2y, cv2.INTER_LINEAR)

    return img1_rect, img2_rect

def Points_to_3d(P1, P2, matching_points1, matching_points2, rgb):

    p1, p2 = matching_points1, matching_points2
    p1, p2 = np.ascontiguousarray(p1, np.float32), np.ascontiguousarray(p2, np.float32)

    colors = []
    for pt in p1:
        x, y = int(pt[0]), int(pt[1])
        color = rgb[y,x] / 255.0
        colors.append(color)
    colors = np.array(colors)

    pts1 = p1.T
    pts2 = p2.T

    points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)

    points_3d = (points_4d[:3] / points_4d[3]).T

    return points_3d, colors

def removing_outliers_radius(points_3d, colors, matching_points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    _, ind = pcd.remove_radius_outlier(nb_points=20, radius=0.5)
    inlier_pcd = pcd.select_by_index(ind)

    filtered_points_3d = np.asarray(inlier_pcd.points)
    filtered_colors = np.asarray(inlier_pcd.colors)

    filtered_pts1 = [matching_points[0][i] for i in ind]
    filtered_pts2 = [matching_points[1][i] for i in ind]
    filtered_matching_points = [filtered_pts1, filtered_pts2]

    return filtered_points_3d, filtered_colors, filtered_matching_points

def initialize_reconstruction(matching_points, K, dist, rgbs, matches_count):
    # most_match = np.argmax(matches_count)
    most_match = 0

    R_e, T_e = find_Rotation_and_translation(matching_points[most_match],K,dist)
    P1, P2 = Projection_Matrix(K, np.eye(3), np.zeros((3,1)), R_e, T_e)
    points_3d, colors = Points_to_3d(P1, P2, matching_points[most_match][0], matching_points[most_match][1], rgbs[most_match])
    points_3d, colors, matching_points_ = removing_outliers_radius(points_3d, colors, matching_points[most_match])

    cameras = [(np.eye(3), np.zeros((3, 1))), (R_e, T_e)]

    return cameras, points_3d, colors, most_match, matching_points_


def register_new_image_left(points_3d, matching_points, rgbs, colors, idx, K, cameras):
    """Register new image to the left of base image"""
    curr_pts1 = np.array(matching_points[idx][0])  # Points in current image
    curr_pts2 = np.array(matching_points[idx][1])  # Points in next (base) image
    
    # Project 3D points into base camera view
    base_R, base_t = cameras[0]  # Base camera is the first one
    proj_matrix = K @ np.hstack((base_R, base_t))
    
    pts_homo = np.hstack((points_3d, np.ones((len(points_3d), 1))))
    proj_points = (proj_matrix @ pts_homo.T).T
    projected_points = proj_points[:, :2] / proj_points[:, 2:]
    
    # Find correspondences using KD-tree
    tree = cKDTree(projected_points)
    distances, indices = tree.query(curr_pts2, distance_upper_bound=5.0)
    
    valid_matches = distances != np.inf
    point3d_indices = indices[valid_matches]
    point2d_indices = np.where(valid_matches)[0]
    
    matched_points3d = points_3d[point3d_indices]
    matched_points2d = curr_pts1[point2d_indices]
    
    # Estimate new camera pose using PnP
    _, rvec, tvec, inliers = cv2.solvePnPRansac(
        matched_points3d,
        matched_points2d,
        K,
        None,
        iterationsCount=2000,
        confidence=0.999,
        reprojectionError=8.0,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    R_new, _ = cv2.Rodrigues(rvec)
    
    # Get unmatched points
    all_indices = set(range(len(curr_pts1)))
    matched_indices = set(point2d_indices)
    unmatched_indices = list(all_indices - matched_indices)

    new_points2d_1 = curr_pts1[unmatched_indices]
    new_points2d_2 = curr_pts2[unmatched_indices]
    
    # Triangulate new points
    P1 = K @ np.hstack((R_new, tvec))
    P2 = K @ np.hstack((base_R, base_t))

    print("Number of points being added from image{}: {}".format(idx,len(unmatched_indices)))
    
    pts_3d, new_colors = Points_to_3d(P1, P2, new_points2d_1, new_points2d_2, rgbs[idx])
    pts_3d, new_colors, matching_points_ = removing_outliers_radius(pts_3d, new_colors, [new_points2d_1, new_points2d_2])
    
    # Insert camera at beginning
    cameras.insert(0, (R_new, tvec))
    all_points3d = np.vstack((points_3d, pts_3d))
    all_colors = np.vstack((colors, new_colors))
    
    return cameras, all_points3d, all_colors, matching_points_



def register_new_image_right(points_3d, matching_points, rgbs, colors, idx, K, cameras):

    curr_pts1 = np.array(matching_points[idx-1][0])  # Points in previous image
    curr_pts2 = np.array(matching_points[idx-1][1])  # Points in current image
    
    R_prev, t_prev = cameras[-1]
    proj_matrix = K @ np.hstack((R_prev, t_prev))

    projected_points = []
    # Project 3D points into previous camera view
    for pt3d in points_3d:
        pt_homo = np.append(pt3d, 1)
        pt_proj = proj_matrix @ pt_homo
        pt_proj = pt_proj[:2] / pt_proj[2]
        projected_points.append(pt_proj)
    
    projected_points = np.array(projected_points)
    
    # Find correspondences using KD-tree
    tree = cKDTree(projected_points)
    distances, indices = tree.query(curr_pts1, distance_upper_bound=5.0)
    
    valid_matches = distances != np.inf
    point3d_indices = indices[valid_matches]
    point2d_indices = np.where(valid_matches)[0]

    # Extract matched 3D points and their 2D projections in current image
    matched_points3d = points_3d[point3d_indices]
    matched_points2d = curr_pts2[point2d_indices]

     # Estimate new camera pose using PnP
    _, rvec, tvec, inliers = cv2.solvePnPRansac(
        matched_points3d, 
        matched_points2d, 
        K, 
        None, 
        iterationsCount=1000, 
        confidence=0.999, 
        reprojectionError=5.0,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    R_new, _ = cv2.Rodrigues(rvec)

    # Find points that don't have 3D correspondences yet
    all_indices = set(range(len(curr_pts1)))
    matched_indices = set(point2d_indices)
    unmatched_indices = list(all_indices - matched_indices)

    print("Number of points being added from image{}: {}".format(idx,len(unmatched_indices)))

    # Get unmatched points for triangulation
    new_points2d_1 = curr_pts1[unmatched_indices]
    new_points2d_2 = curr_pts2[unmatched_indices]
    
    P1, P2 = Projection_Matrix(K, R_prev, t_prev, R_new, tvec)

    pts_3d, new_colors = Points_to_3d(P1, P2 ,new_points2d_1, new_points2d_2, rgbs[idx-1])
    pts_3d, new_colors, matching_points_ = removing_outliers_radius(pts_3d, new_colors, [new_points2d_1, new_points2d_2])

    cameras.append((R_new,tvec))
    all_points3d = np.vstack((points_3d, pts_3d))
    all_colors = np.vstack((colors, new_colors))

    return cameras, all_points3d, all_colors, matching_points_

