import os
import glob
import random
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from tqdm import tqdm

from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.spatial.distance import cdist

#=========================================================
def load_dataset(data_dir):
    seed = 42
    rng = random.Random(seed)

    class_names = sorted([
        d for d in os.listdir(data_dir)
    ])

    train_paths, test_paths = [], []
    y_train, y_test = [], []

    num_test = 10

    for idx, cls in enumerate(class_names):
        data_paths = sorted(glob.glob(os.path.join(data_dir, cls, "*")))
        rng.shuffle(data_paths)

        test_class_paths = data_paths[:num_test]
        train_class_paths = data_paths[num_test:]

        test_paths.extend(test_class_paths)
        y_test.extend([idx] * len(test_class_paths))

        train_paths.extend(train_class_paths)
        y_train.extend([idx] * len(train_class_paths))
    
    return train_paths, np.array(y_train), test_paths, np.array(y_test), class_names

#=========================================================
# T1
def dense_sift_extraction(img, step, sizes):
    h, w = img.shape

    descriptors = []
    keypoints = []

    sift = cv2.SIFT_create()

    for size in sizes:
        margin = size // 2

        kps = [
            cv2.KeyPoint(float(x), float(y), float(size))
            for y in range(margin, h-margin, step)
            for x in range(margin, w-margin, step)
        ]

        _, desc = sift.compute(img, kps)

        descriptors.append(desc)
        keypoints.extend(kps)

    return np.vstack(descriptors), keypoints

def visualise_keypoints(img_path, step, sizes, out_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, kps = dense_sift_extraction(gray, step, sizes)

    vis = cv2.drawKeypoints(
        img, kps, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    cv2.imwrite(out_path, vis)
    
#=========================================================
# T2
def get_total_descriptors(train_paths, step, sizes, max_desc):
    all_desc = []

    print(f"Extracting descriptors from {len(train_paths)} training images …\n")
    for path in tqdm(train_paths, leave=False):
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        desc, _ = dense_sift_extraction(gray, step=step, sizes=sizes)

        all_desc.append(desc)

    all_desc = np.vstack(all_desc).astype(np.float32)
    print(f"\nTotal descriptors: {all_desc.shape[0]}")

    if all_desc.shape[0] > max_desc:
        idx = np.random.choice(all_desc.shape[0], max_desc, replace=False)
        all_desc = all_desc[idx]
        print(f"Sub-sampled to {all_desc.shape[0]} descriptors")

    return all_desc

def build_codebook(all_desc, k, codebook_path):
    if os.path.exists(codebook_path):
            return joblib.load(codebook_path)
    
    print(f"\nFitting MiniBatchKMeans with K={k} …")
    km = MiniBatchKMeans(n_clusters=k)
    km.fit(all_desc)

    joblib.dump(km, codebook_path)
    print(f"Codebook saved → {codebook_path}")

    return km


def main():
    data_dir = "caltech20"
    grid_step = 8
    multple_scales = [8, 16, 24]
    max_desc = 100000
    k_list = [64, 128, 256]

    result_path = "result"
    os.makedirs(result_path, exist_ok=True)

    # T0
    print("=" * 55)
    train_paths, y_train, test_paths, y_test, class_names = load_dataset(data_dir)
    n_cls = len(class_names)
    print(f"  Classes : {n_cls}")
    print(f"  Train   : {len(train_paths)}")
    print(f"  Test    : {len(test_paths)}")

    # T1
    visualise_keypoints(train_paths[0], grid_step, multple_scales, 
                        os.path.join(result_path, "keypoints_train0.jpg"))
    visualise_keypoints(test_paths[0], grid_step, multple_scales, 
                        os.path.join(result_path, "keypoints_test0.jpg"))
    
    # T2
    print("=" * 55)
    all_train_desc = get_total_descriptors(train_paths, grid_step, multple_scales, max_desc)
    
    for k in k_list:
        cb_path = os.path.join(result_path, f"codebook_K{k}.pkl")
        codebook = build_codebook(all_train_desc, k, cb_path)

if __name__ == "__main__":
    main()