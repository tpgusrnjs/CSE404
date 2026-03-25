"""
pa1.py — Photometric Stereo
CSE404: Introduction to Computer Vision, DGIST Spring 2026

Tasks:
  T1  Sphere fitting (cx, cy, r) from chrome-ball mask
  T2  Analytical sphere normal map
  T3  Light-direction estimation from specular highlights
  T4  Least-squares photometric-stereo solver
  T5  Re-shading under a novel (frontal) light
"""

import sys
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_file_list(txt_path):
    """
    Parse a .txt file whose first line is the image count,
    then one path per line, and whose last line is the mask path.
    Returns (image_paths: list[str], mask_path: str).
    """
    with open(txt_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    n = int(lines[0])
    image_paths = lines[1: 1 + n]
    mask_path   = lines[-1]
    return image_paths, mask_path


def load_images_gray(paths):
    """Load images as float64 grayscale in [0, 1]."""
    imgs = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {p}")
        imgs.append(img.astype(np.float64) / 255.0)
    return imgs  # list of (H, W) arrays


def load_mask(path):
    """Load a mask image and return a boolean (H, W) array."""
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    return m > 128


# ---------------------------------------------------------------------------
# T1 — Sphere fitting
# ---------------------------------------------------------------------------

def fit_sphere(mask):
    """
    Fit a circle to the chrome-ball mask using the bounding box
    of the masked region.

    Returns (cx, cy, r) as floats, where (cx, cy) is the centre
    and r is the radius.
    """
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("Mask is empty — cannot fit sphere.")

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    r  = ((x_max - x_min) + (y_max - y_min)) / 4.0  # average of half-widths

    print(f"[T1] Sphere fit: centre=({cx:.1f}, {cy:.1f}), radius={r:.1f}")
    return cx, cy, r


# ---------------------------------------------------------------------------
# T2 — Sphere normal map
# ---------------------------------------------------------------------------

def compute_sphere_normals(mask, cx, cy, r):
    """
    Compute the analytical surface-normal map of the chrome sphere.

    For each pixel (x, y) inside the mask:
        nx = (x - cx) / r
        ny = (y - cy) / r
        nz = sqrt(1 - nx^2 - ny^2)

    Returns an (H, W, 3) float64 array of unit normals (zeros outside mask).
    """
    H, W = mask.shape
    normals = np.zeros((H, W, 3), dtype=np.float64)

    ys, xs = np.where(mask)
    nx = (xs - cx) / r
    ny = (ys - cy) / r

    # Clamp to avoid sqrt of negatives at mask boundary
    nz_sq = 1.0 - nx**2 - ny**2
    nz_sq = np.clip(nz_sq, 0.0, None)
    nz = np.sqrt(nz_sq)

    normals[ys, xs, 0] = nx
    normals[ys, xs, 1] = ny
    normals[ys, xs, 2] = nz

    print("[T2] Sphere normal map computed.")
    return normals


# ---------------------------------------------------------------------------
# T3 — Light direction estimation
# ---------------------------------------------------------------------------

def detect_highlight(img, mask):
    """
    Find the specular-highlight centre in a grayscale chrome-ball image.
    Only pixels inside the mask are considered.
    Returns (hx, hy) as floats.
    """
    masked = img.copy()
    masked[~mask] = 0.0

    # Use the centroid of the top-1 % brightest pixels for robustness
    threshold = np.percentile(masked[mask], 99)
    bright = (masked >= threshold) & mask

    ys, xs = np.where(bright)
    if len(xs) == 0:
        # Fall back to global maximum
        idx = np.argmax(masked)
        hy, hx = np.unravel_index(idx, masked.shape)
        return float(hx), float(hy)

    hx = xs.mean()
    hy = ys.mean()
    return hx, hy


def estimate_light_directions(chrome_imgs, chrome_mask, cx, cy, r):
    """
    T3: Estimate the 12 light directions from chrome-ball images.

    For each image:
      1. Detect highlight centre h = (hx, hy).
      2. Compute sphere normal N at h.
      3. Apply reflection formula: L = 2(N·R)N - R, with R=(0,0,1).
      4. Normalise L to unit length.

    Returns (12, 3) float64 array of unit light directions.
    """
    R = np.array([0.0, 0.0, 1.0])
    lights = []

    for i, img in enumerate(chrome_imgs):
        hx, hy = detect_highlight(img, chrome_mask)

        # Normal at highlight
        nx = (hx - cx) / r
        ny = (hy - cy) / r
        nz_sq = max(0.0, 1.0 - nx**2 - ny**2)
        nz = np.sqrt(nz_sq)
        N = np.array([nx, ny, nz])

        # Normalise (should already be unit, but be safe)
        n_norm = np.linalg.norm(N)
        if n_norm < 1e-8:
            N = np.array([0.0, 0.0, 1.0])
        else:
            N = N / n_norm

        # L = 2(N·R)N - R
        L = 2.0 * np.dot(N, R) * N - R

        l_norm = np.linalg.norm(L)
        if l_norm < 1e-8:
            L = np.array([0.0, 0.0, 1.0])
        else:
            L = L / l_norm

        lights.append(L)
        print(f"[T3] Light {i:02d}: highlight=({hx:.1f},{hy:.1f})  "
              f"L=({L[0]:+.4f}, {L[1]:+.4f}, {L[2]:+.4f})  ‖L‖={np.linalg.norm(L):.4f}")

    return np.array(lights)  # (12, 3)


# ---------------------------------------------------------------------------
# T4 — Photometric stereo least-squares solver
# ---------------------------------------------------------------------------

def photometric_stereo(subject_imgs, subject_mask, lights):
    """
    T4: Recover surface normals and albedo from n images.

    Per-pixel least-squares:
        i = L * g     →     g = (LᵀL)⁻¹ Lᵀ i
        ρ = ‖g‖,  N = g / ρ

    Parameters
    ----------
    subject_imgs  : list of (H, W) float64 images
    subject_mask  : (H, W) bool array
    lights        : (n, 3) float64 light directions

    Returns
    -------
    normals : (H, W, 3) float64  — unit normals, zero outside mask
    albedo  : (H, W)   float64  — albedo, zero outside mask
    """
    H, W = subject_mask.shape
    n    = len(subject_imgs)

    # Stack intensities: (n, H*W)
    I = np.stack([img.ravel() for img in subject_imgs], axis=0)  # (n, H*W)

    # Only process pixels inside the mask
    mask_flat = subject_mask.ravel()  # (H*W,)
    I_masked  = I[:, mask_flat]       # (n, P) where P = #mask pixels

    L = lights  # (n, 3)

    # Solve least-squares: G (3, P) = (LᵀL)⁻¹ Lᵀ I_masked
    # np.linalg.lstsq solves L @ G = I for each column of I simultaneously
    # We need G^T shape (P, 3): lstsq(L, I_masked) → solution (3, P)
    # lstsq expects (n, 3) and (n, P)
    G, _, _, _ = np.linalg.lstsq(L, I_masked, rcond=None)  # (3, P)

    # Albedo = ‖g‖ per pixel
    rho = np.linalg.norm(G, axis=0)  # (P,)

    # Unit normals
    safe_rho = np.where(rho < 1e-8, 1.0, rho)
    N_pixels = G / safe_rho[np.newaxis, :]  # (3, P)

    # Write back into full-size maps
    normals = np.zeros((H, W, 3), dtype=np.float64)
    albedo  = np.zeros((H, W),    dtype=np.float64)

    normals.reshape(-1, 3)[mask_flat] = N_pixels.T
    albedo.ravel()[mask_flat]         = rho

    print(f"[T4] Photometric stereo done. "
          f"albedo range: [{albedo[subject_mask].min():.4f}, "
          f"{albedo[subject_mask].max():.4f}]")

    return normals, albedo


# ---------------------------------------------------------------------------
# T5 — Re-shading
# ---------------------------------------------------------------------------

def reshade(normals, albedo, mask, light=None):
    """
    T5: Render the object under a given light using Lambertian shading.

        I = ρ · clamp(N · L, 0, 1)

    Default light is frontal: L = (0, 0, 1).
    Returns (H, W) float64 shading image.
    """
    if light is None:
        light = np.array([0.0, 0.0, 1.0])
    light = light / np.linalg.norm(light)

    # N · L per pixel
    NdotL = np.einsum('hwc,c->hw', normals, light)   # (H, W)
    NdotL = np.clip(NdotL, 0.0, 1.0)

    shading = albedo * NdotL
    shading[~mask] = 0.0

    print(f"[T5] Re-shading done. Light direction: {light}")
    return shading


# ---------------------------------------------------------------------------
# Saving outputs
# ---------------------------------------------------------------------------

def save_normal_map(normals, mask, path):
    """Encode normals as RGB: (N + 1) / 2 * 255, uint8."""
    vis = ((normals + 1.0) / 2.0 * 255.0).astype(np.uint8)
    vis[~mask] = 0
    # OpenCV uses BGR; swap R↔B so X→Red, Y→Green, Z→Blue
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, vis_bgr)
    print(f"Saved: {path}")


def save_grayscale(img, mask, path, normalise=True):
    """Save a float64 map as uint8 grayscale."""
    out = img.copy()
    out[~mask] = 0.0
    if normalise and out[mask].max() > 1e-8:
        out = out / out[mask].max()
    out_u8 = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
    cv2.imwrite(path, out_u8)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    chrome_txt  = "chrome.txt"
    subject_txt = "buddha.txt"

    # --- Load data ----------------------------------------------------------
    chrome_paths,  chrome_mask_path  = load_file_list(chrome_txt)
    subject_paths, subject_mask_path = load_file_list(subject_txt)

    print(f"Loading {len(chrome_paths)} chrome images …")
    chrome_imgs  = load_images_gray(chrome_paths)
    chrome_mask  = load_mask(chrome_mask_path)

    print(f"Loading {len(subject_paths)} subject images …")
    subject_imgs = load_images_gray(subject_paths)
    subject_mask = load_mask(subject_mask_path)

    # --- T1: Sphere fitting -------------------------------------------------
    cx, cy, r = fit_sphere(chrome_mask)

    # --- T2: Sphere normal map ----------------------------------------------
    sphere_normals = compute_sphere_normals(chrome_mask, cx, cy, r)
    save_normal_map(sphere_normals, chrome_mask, "sphere_normals.png")

    # --- T3: Light estimation -----------------------------------------------
    lights = estimate_light_directions(chrome_imgs, chrome_mask, cx, cy, r)
    np.savetxt("lights.txt", lights, fmt="%.6f",
               header="Estimated light directions (12 x 3)")
    print("Saved: lights.txt")

    # --- T4: Photometric stereo ---------------------------------------------
    normals, albedo = photometric_stereo(subject_imgs, subject_mask, lights)
    save_normal_map(normals, subject_mask, "normals.png")
    save_grayscale(albedo,  subject_mask, "albedo.png", normalise=True)

    # --- T5: Re-shading -----------------------------------------------------
    shading = reshade(normals, albedo, subject_mask, light=np.array([0.0, 0.0, 1.0]))
    save_grayscale(shading, subject_mask, "shading_frontal.png", normalise=True)

    print("\nAll done! Outputs: normals.png, albedo.png, shading_frontal.png")


if __name__ == "__main__":
    main()