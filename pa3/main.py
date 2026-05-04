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
from itertools import product

from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

#=========================================================
seed = 42
np.random.seed(seed)
random.seed(seed)

def load_dataset(data_dir):
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
def extract_dense_sift(img, step, sizes):
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

        kps, desc = sift.compute(img, kps)

        descriptors.append(desc)
        keypoints.extend(kps)

    return np.vstack(descriptors), np.array([kp.pt for kp in keypoints], dtype=np.float32)

def visualise_keypoints(img_path, step, sizes, out_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, kps = extract_dense_sift(gray, step, sizes)

    kps = [
            cv2.KeyPoint(float(x), float(y), 1)
            for x, y in kps
        ]
    
    vis = cv2.drawKeypoints(
        img, kps, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    cv2.imwrite(out_path, vis)
    
#=========================================================
# T2
def get_total_descriptors(train_paths, step, sizes, max_desc):
    desc_list = []
    kps_list = []
    shapes = []

    print(f"extracting descriptors from {len(train_paths)} images …\n")
    for path in tqdm(train_paths, leave=False):
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        desc, kps = extract_dense_sift(gray, step, sizes)

        desc_list.append(desc)
        kps_list.append(kps)
        shapes.append(gray.shape)

    all_desc = np.vstack(desc_list).astype(np.float32)
    print(f"\ntotal descriptors: {all_desc.shape[0]}")

    if all_desc.shape[0] > max_desc:
        idx = np.random.choice(all_desc.shape[0], max_desc, replace=False)
        all_desc = all_desc[idx]
        print(f"sub-sampled to {all_desc.shape[0]} descriptors\n")

    return all_desc, desc_list, kps_list, shapes

def build_codebook(all_desc, k, codebook_path):
    if os.path.exists(codebook_path):
            print(f"load codebook ← {codebook_path}")
            return joblib.load(codebook_path)
    
    print(f"\nfitting MiniBatchKMeans with K={k} …")
    km = MiniBatchKMeans(n_clusters=k, random_state=42)
    km.fit(all_desc)

    joblib.dump(km, codebook_path)
    print(f"codebook saved → {codebook_path}")

    return km
#=========================================================
# T3
def encode_bovw(desc, codebook):
    K = codebook.n_clusters
    labels = codebook.predict(desc)
    h, _ = np.histogram(labels, bins=np.arange(K+1))
    h = h.astype(np.float32)
    h /= h.sum() + 1e-7
    return h

def bovw_hist_encoding(desc_list, codebook):
    feats = [encode_bovw(desc, codebook) for desc in tqdm(desc_list)]
    return np.vstack(feats)

#=========================================================
# T4
def train_linear_svm(X_train, y_train, C):
    svm = SVC(kernel="linear", C=C, random_state=42)
    svm.fit(X_train, y_train)
    return svm

def chi2_dist(H1, H2):
    h1 = H1[:, np.newaxis, :] # (n, 1, d)
    h2 = H2[np.newaxis, :, :] # (1, m, d)
    return np.sum(((h1 - h2) ** 2) / (h1 + h2 + 1e-10), axis=2) # (n, m, d) -> (n, m)

def estimate_gamma(X_train):
    idx = np.random.choice(X_train.shape[0], min(500, X_train.shape[0]), replace=False)
    X = X_train[idx]

    dist = chi2_dist(X, X)
    return 1.0 / np.mean(dist)
    
def chi2_kernel(H1, H2, gamma):
    dist = chi2_dist(H1, H2)
    return np.exp(-gamma * dist)

def train_chi2_svm(X_train, y_train, C, gamma=None):
    if gamma is None:
        dist = chi2_dist(X_train, X_train)
        gamma = 1.0 / np.mean(dist)
        print(f"\nestimated gamma = {gamma:.4f}")

    K_train = chi2_kernel(X_train, X_train, gamma)
    svm = SVC(kernel="precomputed", C=C, random_state=42)
    svm.fit(K_train, y_train)
    return svm, gamma

#=========================================================
# T5
def build_spm(words, kps, shape, codebook):
    K = codebook.n_clusters
    h, w = shape[:2]

    levels = [0, 1, 2]
    weights = {0: 1/4, 1: 1/4, 2: 1/2}

    feats = []
    for l in levels:
        n_cell = 2**l

        x_idx = (kps[:, 0] * n_cell / w).astype(np.int32)
        y_idx = (kps[:, 1] * n_cell / h).astype(np.int32)

        x_idx = np.clip(x_idx, 0, n_cell - 1)
        y_idx = np.clip(y_idx, 0, n_cell - 1)

        cell_ids = y_idx * n_cell + x_idx

        hist = np.zeros((n_cell * n_cell, K), dtype=np.float32)
        np.add.at(hist, (cell_ids, words), 1)

        hist /= hist.sum(axis=1, keepdims=True) + 1e-7
        hist *= weights[l]

        feats.append(hist.ravel())

    feats = np.concatenate(feats)
    feats /= np.linalg.norm(feats) + 1e-7
    
    return feats

def spm_encoding(desc_list, kps_list, shapes, codebook):
    words_list = [codebook.predict(desc) for desc in desc_list]

    feats = [build_spm(words, kps, shape, codebook) 
             for words, kps, shape in tqdm(zip(words_list, kps_list, shapes))]
    return np.vstack(feats)

#=========================================================
# T6
def accuracy_per_class(y_true, y_pred, n_classes):
    accs = []
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum() == 0:
            accs.append(0.0)
        else:
            accs.append(float(np.mean(y_pred[mask] == c)))
    return accs

def plot_accuracy_bar(per_class_linear, per_class_chi2, class_names, label1, label2, title, out_path):
    x = np.arange(len(class_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width / 2, per_class_linear, width, label=label1, alpha=0.8)
    ax.bar(x + width / 2, per_class_chi2, width, label=label2, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy per class")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\naccuracy bar chart saved → {out_path}")

def plot_confusion_matrix(y_true, y_pred, class_names, title, out_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-7)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names, yticklabels=class_names,
        title=title, ylabel="True label", xlabel="Predicted label"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"confusion matrix saved → {out_path}")

def plot_spm_vs_flat(K_values, flat_accs, spm_accs, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(K_values, flat_accs, "o-", label="Flat BoVW")
    ax.plot(K_values, spm_accs, "s--", label="SPM BoVW")
    ax.set_xlabel("Vocabulary size K")
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title("Flat BoVW vs SPM BoVW (χ² SVM)")
    ax.set_xticks(K_values)
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"SPM vs Flat plot saved → {out_path}")

#=========================================================
def main(): 
    data_dir = "caltech20"
    grid_step = 8
    multple_scales = [8, 16, 24]
    max_desc = 100000
    k_list = [64, 128, 256]
    c_list = [1, 10, 100]

    result_path = "result"
    os.makedirs(result_path, exist_ok=True)

    subdirs = [
        "keypoints",
        "codebook",
        "bovw_features",
        "spm_features",
        "accuracy_bar_comparison",
        "confusion_matrix"
    ]

    for d in subdirs:
        os.makedirs(os.path.join(result_path, d), exist_ok=True)

    # T0
    print("=" * 55)
    train_paths, y_train, test_paths, y_test, class_names = load_dataset(data_dir)
    n_cls = len(class_names)
    print(f"  Classes : {n_cls}")
    print(f"  Train   : {len(train_paths)}")
    print(f"  Test    : {len(test_paths)}")

    # T1
    visualise_keypoints(train_paths[0], grid_step, multple_scales, 
                        os.path.join(result_path, "keypoints", "keypoints_train0.jpg"))
    visualise_keypoints(test_paths[0], grid_step, multple_scales, 
                        os.path.join(result_path, "keypoints", "keypoints_test0.jpg"))
    
    # T2
    print("=" * 55)
    all_train_desc, train_desc_list, train_kps_list, train_shapes = get_total_descriptors(train_paths, grid_step, multple_scales, max_desc)
    _, test_desc_list, test_kps_list, test_shapes = get_total_descriptors(test_paths, grid_step, multple_scales, max_desc)
    
    results_flat = {}
    results_spm = {}

    for k in k_list:
        cb_path = os.path.join(result_path, "codebook", f"codebook_K{k}.pkl")
        feat_path = os.path.join(result_path, "bovw_features", f"bovw_features_K{k}.npz")
        spm_path = os.path.join(result_path, "spm_features", f"spm_features_K{k}.npz")

        codebook = build_codebook(all_train_desc, k, cb_path)
        
        # T3
        X_train = bovw_hist_encoding(train_desc_list, codebook)
        X_test = bovw_hist_encoding(test_desc_list, codebook)

        np.savez(feat_path, X_train, X_test)
        print(f"Saved → {feat_path}\n")

        results_flat[k] = (X_train, X_test, codebook)

        # T5
        X_train_spm = spm_encoding(train_desc_list, train_kps_list, train_shapes, codebook)
        X_test_spm  = spm_encoding(test_desc_list,  test_kps_list,  test_shapes,  codebook)

        results_spm[k] = (X_train_spm, X_test_spm, codebook)

        np.savez(spm_path, X_train_spm, X_test_spm)
        print(f"Saved → {spm_path}\n")

    #T4
    acc_per_cls_chi2_results = {}
    flat_results = {}
    gamma_cache = {}

    for k, c in list(product(k_list, c_list)):
        X_train, X_test, codebook = results_flat[k]

        print("=" * 55)
        print(f"[BoVW] Training SVM (K={k}, C={c}) …")
        linear_svm = train_linear_svm(X_train, y_train, c)
        y_pred_linear = linear_svm.predict(X_test)
        acc_linear = accuracy_score(y_test, y_pred_linear)
        print(f"\nLinear SVM accuracy: {acc_linear*100:.2f}%")

        gamma = gamma_cache.get(k, None)
        chi2_svm, gamma = train_chi2_svm(X_train, y_train, C=c, gamma=gamma)
        gamma_cache[k] = gamma
        K_test = chi2_kernel(X_test, X_train, gamma)
        y_pred_chi2 = chi2_svm.predict(K_test)
        acc_chi2 = accuracy_score(y_test, y_pred_chi2)
        print(f"χ² SVM accuracy: {acc_chi2*100:.2f}%")

        if k not in flat_results or acc_chi2 > flat_results[k]:
            flat_results[k] = acc_chi2
        
        acc_per_cls_chi2 = accuracy_per_class(y_test, y_pred_chi2, n_cls)
        acc_per_cls_chi2_results[k+c] = acc_per_cls_chi2

        # T6
        plot_accuracy_bar(
            accuracy_per_class(y_test, y_pred_linear, n_cls),
            acc_per_cls_chi2,
            class_names, "Linear SVM", "χ² SVM", f"Linear vs χ² SVM (Flat BoVW, K={k}, C={c})",
            os.path.join(result_path, "accuracy_bar_comparison", f"accuracy_bar_comparison_K{k}_C{c}.png")
        )

        plot_confusion_matrix(y_test, y_pred_linear, class_names,
                          f"Linear SVM (Flat BoVW, K={k}, C={c})",
                          os.path.join(result_path, "confusion_matrix", f"confusion_matrix_linear_K{k}_C{c}.png"))
        plot_confusion_matrix(y_test, y_pred_chi2, class_names,
                          f"χ² SVM (Flat BoVW, K={k}, C={c})",
                          os.path.join(result_path, "confusion_matrix", f"confusion_matrix_chi2_K{k}_C{c}.png"))

    #T5
    spm_results = {}
    gamma_cache = {}

    for k, c in list(product(k_list, c_list)):
        X_train, X_test, codebook = results_spm[k]

        print("=" * 55)
        print(f"[SPM] Training SVM (K={k}, C={c}) …")

        gamma = gamma_cache.get(k, None)
        chi2_svm, gamma = train_chi2_svm(X_train, y_train, C=c, gamma=gamma)
        gamma_cache[k] = gamma
        K_test = chi2_kernel(X_test, X_train, gamma)
        y_pred_chi2 = chi2_svm.predict(K_test)
        acc_chi2 = accuracy_score(y_test, y_pred_chi2)
        print(f"χ² SVM accuracy: {acc_chi2*100:.2f}%")

        if k not in spm_results or acc_chi2 > spm_results[k]:
            spm_results[k] = acc_chi2

        acc_per_cls_chi2 = acc_per_cls_chi2_results[k+c]

        plot_accuracy_bar(
            acc_per_cls_chi2,
            accuracy_per_class(y_test, y_pred_chi2, n_cls),
            class_names, "Flat BoVW", "SPM BoVW", f"Flat BoVW vs SPM BoVW (χ² SVM, K={k}, C={c})",
            os.path.join(result_path, "accuracy_bar_comparison", f"accuracy_bar_comparison_chi2_K{k}_C{c}.png")
        )

        plot_confusion_matrix(y_test, y_pred_chi2, class_names,
                          f"χ² SVM (SPM, K={k}, C={c})",
                          os.path.join(result_path, "confusion_matrix", f"confusion_matrix_chi2_spm_K{k}_C{c}.png"))
        

    flat_accs = [flat_results[k] for k in k_list]
    spm_accs  = [spm_results[k]  for k in k_list]

    # T6
    plot_spm_vs_flat(
        k_list,
        flat_accs,
        spm_accs,
        out_path=os.path.join(result_path, "spm_vs_flat_comparison.png")
    )

if __name__ == "__main__":
    main()