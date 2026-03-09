import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

####################################################################################
# T0

def load_dataset():
    dataset = []

    for i, file in enumerate(glob.glob("psmImages/*.txt")):
        name = file[10:-4]

        with open(file) as f:
            lines = [line.strip() for line in f.readlines()]
            
        num_img = int(lines[0])
        image_paths = lines[1:1+num_img]
        mask = lines[-1]

        dataset.append((name, num_img, image_paths, mask))
    return dataset

def visualize_results(map, title, cmap=None):
    plt.imshow(map, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()
    

####################################################################################
# T1

def sphere(mask):
    edges = cv2.Canny(mask,50,150)
    ys, xs = np.where(edges > 0)

    A = np.column_stack((xs, ys, np.ones(len(xs))))
    b = xs**2 + ys**2

    C, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    cx = C[0] / 2
    cy = C[1] / 2
    r = np.sqrt(C[2] + cx**2 + cy**2)

    return cx, cy, r

####################################################################################
# T2
def surface_normal(x, y, cx, cy, r):
    nx = (x - cx) / r
    ny = (y - cy) / r
    nz = np.sqrt(1 - nx**2 - ny**2)

    return np.array([nx, ny, nz])

####################################################################################
# T3
def light_estimation(image_paths, mask, cx, cy, r):
    L = []
    R = np.array([0, 0, 1]) #(3,)

    for p in image_paths:
        img = cv2.imread(p)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

        _, _, _, maxLoc = cv2.minMaxLoc(masked_gray)
        hx, hy = maxLoc

        # T2 Normal map
        N = surface_normal(hx, hy, cx, cy, r) #(3,)
        assert np.isclose(np.linalg.norm(N), 1)
        
        light_dir = 2 * (N @ R) * N - R #(3,)
        light_dir = light_dir / np.linalg.norm(light_dir)

        L.append(light_dir)

    return np.array(L) # (12, 3)

####################################################################################
# T4
def photometric_stereo(imgs, mask, L):
    n, H, W = imgs.shape
    I = imgs.reshape(n, -1) #(12, H*W)

    mask_flat = mask.reshape(-1) > 0
    I= I[:, mask_flat]

    G_v, _, _, _ = np.linalg.lstsq(L, I, rcond=None)

    G = np.zeros((3, H*W))
    G[:, mask_flat] = G_v

    albedo = np.linalg.norm(G, axis=0)
    normals = G / (albedo + 1e-8)

    albedo_map = albedo.reshape(H, W) # (H, W)
    normal_map = normals.reshape(3, H, W).transpose(1, 2, 0) # (H, W, 3)
    normal_map = (normal_map + 1) / 2 * 255

    return albedo_map, normal_map, normals

####################################################################################
# T5
def re_shading(albedo_map, normal_map):
    L_ = np.array([0, 0, 1]) #(3,)
    shading = albedo_map * np.maximum(0, normal_map @ L_)#(H, W)
    return shading

####################################################################################
if __name__ == "__main__":

    # T0: Load dataset
    dataset = load_dataset()
    sphere_data = dataset[2]
    dataset.pop(2)

    _, _, sphere_img_paths, sphere_mask_path = sphere_data
    sphere_mask = cv2.imread(sphere_mask_path, cv2.IMREAD_GRAYSCALE)
    
    # T1: Sphere fitting
    cx, cy, r = sphere(sphere_mask)

    # T2: Normal map & T3: Light estimation
    L = light_estimation(sphere_img_paths, sphere_mask, cx, cy, r)
    
    for data in dataset:
        save_name, _, img_paths, mask_path = data

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)

        img_list = []
        for p in img_paths:
            img = cv2.imread(p)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_list.append(gray)

        imgs = np.stack(img_list, axis=0) #(12, H, W)

        # T4: Photometric stereo
        albedo_map, normal_map, normals = photometric_stereo(imgs, mask, L)

        # T5: Re-shading (use the 3-channel normal map returned by photometric_stereo)
        shading = re_shading(albedo_map, normal_map)

        #visualize_results(normal_map.astype(np.uint8), "Normal Map")
        #visualize_results(albedo_map, "Albedo Map", cmap='gray')
        #visualize_results(shading, "Shading", cmap='gray')

        save_path = f"results/{save_name}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        plt.imsave(os.path.join(save_path, "normals.png"), normal_map.astype(np.uint8))
        plt.imsave(os.path.join(save_path, "albedo.png"), albedo_map, cmap='gray')
        plt.imsave(os.path.join(save_path, "shading.png"), shading, cmap='gray')
