import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

K_path = 'data/K.txt'
data_path = ["data/0000.jpg", "data/0003.jpg", "data/0018.jpg"]

#########################################################
# T0
def load_data(data_path, K_path):
    K = np.loadtxt(K_path)
    images = []
    for path in data_path:
        img = cv2.imread(path)
        images.append(img)
    return images, K

#########################################################
# T1
def sift_matching(img1, img2, save_name):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    ratio = 0.75
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append([m])

    result_path = "result"
    os.makedirs(result_path, exist_ok=True)
    sift_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(os.path.join(result_path, save_name), sift_matches)

    pts1 = np.float32([kp1[m[0].queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m[0].trainIdx].pt for m in good])
    return pts1, pts2, kp1, kp2, good

#########################################################
# T2

def normalize_points(pts, K):
    K_inv = np.linalg.inv(K)
    pts_hom = np.hstack([pts, np.ones((len(pts), 1))])   # (N, 3)
    pts_norm = (K_inv @ pts_hom.T).T                      # (N, 3)
    return pts_norm[:, :2] / pts_norm[:, 2:3]

def essential_matrix(pts1, pts2, K):
    pts1_norm = normalize_points(pts1, K)
    pts2_norm = normalize_points(pts2, K)

    E, mask = cv2.findEssentialMat(pts1_norm, pts2_norm,
                                   np.eye(3),
                                   method=cv2.RANSAC,
                                   prob=0.999,
                                   threshold=0.001)
    mask = mask.ravel().astype(bool)
    inlier_ratio = mask.sum() / len(mask)
    print(f"RANSAC inlier ratio: {mask.sum()}/{len(mask)} = {inlier_ratio:.2%}")
    return E, mask

#########################################################
# T3
def pose_disambiguation(E, pts1_norm, pts2_norm, K):
    R1, R2, t = cv2.decomposeEssentialMat(E)
    candidates = [
        (R1,  t),
        (R1, -t),
        (R2,  t),
        (R2, -t),
    ]

    P0 = np.hstack((np.eye(3), np.zeros((3, 1))))

    best_R, best_t, best_count = None, None, -1

    for R_cand, t_cand in candidates:
        P_cand = np.hstack((R_cand, t_cand.reshape(3, 1)))

        X_hom = cv2.triangulatePoints(P0, P_cand, pts1_norm.T, pts2_norm.T)
        X = (X_hom[:3] / X_hom[3]).T  # (N, 3)

        z1_pos = X[:, 2] > 0

        X_cam2 = (R_cand @ X.T + t_cand.reshape(3, 1)).T
        z2_pos = X_cam2[:, 2] > 0

        count = (z1_pos & z2_pos).sum()

        if count > best_count:
            best_count = count
            best_R, best_t = R_cand, t_cand

    print(f"Cheirality: best candidate has {best_count}/{len(pts1_norm)} positive-depth points")
    return best_R, best_t.flatten()

#########################################################
# T4
def triangulate_dlt_single(P1, P2, x1, x2):

    A = np.array([
        x1[1] * P1[2] - P1[1],
        P1[0] - x1[0] * P1[2],
        x2[1] * P2[2] - P2[1],
        P2[0] - x2[0] * P2[2],
    ])  # (4, 4)

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X / X[3]
    return X[:3]


def triangulation(R, t, pts1, pts2, K):
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t.reshape(3, 1)))

    X_list = []
    for x1, x2 in zip(pts1, pts2):
        X = triangulate_dlt_single(P1, P2, x1, x2)
        X_list.append(X)

    X = np.array(X_list)  # (N, 3)
    return X, P1, P2

def reprojection_error(X, pts1, pts2, P1, P2, threshold=2.0):
    def project(P, X_3d):
        X_hom = np.hstack((X_3d, np.ones((X_3d.shape[0], 1))))
        proj = (P @ X_hom.T).T
        return proj[:, :2] / proj[:, 2:3]

    proj1 = project(P1, X)
    proj2 = project(P2, X)

    error1_sq = np.sum((proj1 - pts1) ** 2, axis=1)
    error2_sq = np.sum((proj2 - pts2) ** 2, axis=1)
    epsilon = error1_sq + error2_sq

    valid = epsilon < threshold

    mean_err = np.sqrt(epsilon[valid]).mean() if valid.any() else np.nan

    print(f"Filter: {valid.sum()}/{len(X)} points kept. "
          f"Mean reprojection error: {mean_err:.3f} px")

    return X[valid], pts1[valid], pts2[valid], valid

#########################################################
# T5
def pnp_registration(X_3d, pts2d, K):
    X_3d = np.ascontiguousarray(X_3d).astype(np.float64)
    pts2d = np.ascontiguousarray(pts2d).astype(np.float64)
    _, rvec, tvec, inliers = cv2.solvePnPRansac(X_3d, pts2d, K, None,
                                                 flags=cv2.SOLVEPNP_EPNP)
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.flatten()
    return R, t

def save_cameras(cameras, path="result/cameras.txt"):
    os.makedirs("result", exist_ok=True)
    with open(path, "w") as f:
        for i, (R, t) in enumerate(cameras):
            f.write(f"Camera {i}\n")
            f.write(f"R:\n{R}\n")
            f.write(f"t: {t}\n")

def save_ply(X, colors, path="result/points3d.ply"):
    os.makedirs("result", exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(X)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(X, colors):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

def save_reprojection(images, cameras, X, K, path="result/reprojection.png"):
    os.makedirs("result", exist_ok=True)
    fig, axes = plt.subplots(1, len(cameras), figsize=(6 * len(cameras), 5))
    for i, (R, t) in enumerate(cameras):
        P = K @ np.hstack((R, t.reshape(3, 1)))
        X_hom = np.hstack((X, np.ones((len(X), 1))))
        proj = (P @ X_hom.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        img_rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        axes[i].imshow(img_rgb)
        axes[i].set_xlim(0, img_rgb.shape[1])
        axes[i].set_ylim(img_rgb.shape[0], 0)
        axes[i].scatter(proj[:, 0], proj[:, 1], s=1, c='red', alpha=0.5)
        axes[i].set_title(f"Camera {i}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def get_colors(img, pts2d):
    colors = []
    h, w = img.shape[:2]
    for pt in pts2d:
        x, y = int(round(pt[0])), int(round(pt[1]))
        x, y = np.clip(x, 0, w-1), np.clip(y, 0, h-1)
        b, g, r = img[y, x]
        colors.append((r, g, b))
    return np.array(colors)

def main():
    images, K = load_data(data_path, K_path)
    os.makedirs("result", exist_ok=True)

    # T1
    pts1, pts2, kp1, kp2, matches12 = sift_matching(images[0], images[1], "matches_01.png")

    # T2
    E, mask = essential_matrix(pts1, pts2, K)
    pts1_in, pts2_in = pts1[mask], pts2[mask]

    # T3
    pts1_norm = normalize_points(pts1_in, K)
    pts2_norm = normalize_points(pts2_in, K)

    R1, t1 = pose_disambiguation(E, pts1_norm, pts2_norm, K)

    # T4
    X_12, P1, P2 = triangulation(R1, t1, pts1_in, pts2_in, K)
    X_12, pts1_filt, pts2_filt, valid = reprojection_error(X_12, pts1_in, pts2_in, P1, P2)
    colors_12 = get_colors(images[0], pts1_filt)

    cameras = [
        (np.eye(3), np.zeros(3)),
        (R1, t1.flatten()), 
    ]

    # T5
    pts0_new, pts2_new, _, kp2_new, matches02 = sift_matching(images[0], images[2], "matches_02.png")
    E2, mask2 = essential_matrix(pts0_new, pts2_new, K)
    pts0_new, pts2_new = pts0_new[mask2], pts2_new[mask2]

    point_map = {}
    inlier_matches = [m for i, m in enumerate(matches12) if mask[i]]
    filtered_matches = [inlier_matches[i] for i in range(len(valid)) if valid[i]]

    for m, X in zip(filtered_matches, X_12):
        point_map[m[0].queryIdx] = X

    X_pnp, pts_pnp = [], []
    for m in matches02:
        if m[0].queryIdx in point_map:
            X_pnp.append(point_map[m[0].queryIdx])
            pts_pnp.append(kp2_new[m[0].trainIdx].pt)

    X_pnp = np.array(X_pnp)
    pts_pnp = np.array(pts_pnp)

    print("PnP matches:", len(X_pnp))

    R2, t2 = pnp_registration(X_pnp, pts_pnp, K)
    cameras.append((R2, t2))

    P3 = K @ np.hstack((R2, t2.reshape(3, 1)))
    X_13_raw, _, _ = triangulation(R2, t2, pts0_new, pts2_new, K)
    X_13, pts0_filt, _, _ = reprojection_error(X_13_raw, pts0_new, pts2_new, P1, P3)
    colors_13 = get_colors(images[0], pts0_filt)

    X_all = np.vstack([X_12, X_13])
    colors_all = np.vstack([colors_12, colors_13])

    save_cameras(cameras, "result/cameras.txt")
    save_ply(X_all, colors_all, "result/points3d.ply")
    save_reprojection(images, cameras, X_all, K, "result/reprojection.png")

    h0, w0 = images[0].shape[:2]
    h1, w1 = images[1].shape[:2]
    h2, w2 = images[2].shape[:2]

    H = max(h0, h1, h2)

    canvas = np.zeros((H, w1 + w0 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = images[1]
    canvas[:h0, w1:w1+w0] = images[0]
    canvas[:h2, w1+w0:w1+w0+w2] = images[2]

    color01 = (0, 255, 0)   # 0-1 (green)
    color02 = (0, 0, 255)   # 0-2 (red)

    for p0, p1 in zip(pts1, pts2):
        x0, y0 = int(p0[0]) + w1, int(p0[1])   # img0 (가운데)
        x1, y1 = int(p1[0]), int(p1[1])        # img1 (왼쪽)
        cv2.line(canvas, (x0, y0), (x1, y1), color01, 1)

    for p0, p2 in zip(pts0_new, pts2_new):
        x0, y0 = int(p0[0]) + w1, int(p0[1])              # img0 (가운데)
        x2, y2 = int(p2[0]) + w1 + w0, int(p2[1])         # img2 (오른쪽)
        cv2.line(canvas, (x0, y0), (x2, y2), color02, 1)

    cv2.imwrite("result/matches.png", canvas)
    print("Total 3D points:", len(X_all))


if __name__ == "__main__":
    main()