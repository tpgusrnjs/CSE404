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

    for _, file in enumerate(glob.glob("psmImages/*.txt")):
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
    ys, xs = np.where(mask)

    cx = xs.mean()
    cy = ys.mean()

    r = np.mean(np.sqrt((xs - cx)**2 + (ys - cy)**2))

    return cx, cy, r

####################################################################################
# T2
def surface_normal(hx, hy, cx, cy, r):
    nx = (hx - cx) / r
    ny = (hy - cy) / r
    nz = np.sqrt(max(0, 1 - nx**2 - ny**2))

    return np.array([nx, ny, nz])

####################################################################################
# T3
def light_estimation(image_paths, mask, cx, cy, r):
    L = []
    R = np.array([0, 0, 1]) #(3,)

    for p in image_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        masked_gray = cv2.bitwise_and(img, img, mask=mask)

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
    normals = normals.T.reshape(H, W, 3) # (H, W, 3)
    normal_map = (normals + 1) / 2 * 255

    return albedo_map, normal_map, normals

####################################################################################
# T5
def re_shading(albedo_map, normals):
    L_v = np.array([0, 0, 1]) #(3,)
    #print("L_v:", L_v.shape)
    #print("normals:", normals.shape)
    shading = albedo_map * np.maximum(0, np.sum(normals * L_v, axis=2)) #(H, W, 3) @ (3,) -> (H, W)
    #shading = np.clip(shading, 0, 1)
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
        mask = (mask > 0).astype(np.uint8) * 255

        img_list = []
        for p in img_paths:
            img = cv2.imread(p)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            img_list.append(gray)

        imgs = np.stack(img_list, axis=0) #(12, H, W)

        # T4: Photometric stereo
        albedo_map, normal_map, normals = photometric_stereo(imgs, mask, L)

        # T5: Re-shading
        shading = re_shading(albedo_map, normals) #(H, W)

        #visualize_results(normal_map.astype(np.uint8), "Normal Map")
        #visualize_results(albedo_map, "Albedo Map", cmap='gray')
        #visualize_results(shading, "Shading", cmap='gray')

        save_path = f"results/{save_name}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        albedo_map = albedo_map / (albedo_map.max() + 1e-8)
        shading = shading / (shading.max() + 1e-8)

        plt.imsave(os.path.join(save_path, "normals.png"), normal_map.astype(np.uint8))
        plt.imsave(os.path.join(save_path, "albedo.png"), albedo_map, cmap='gray')
        plt.imsave(os.path.join(save_path, "shading.png"), shading, cmap='gray')
