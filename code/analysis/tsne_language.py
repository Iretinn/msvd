"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import torch
import pickle
import os
import matplotlib

# matplotlib.use('TkAgg')
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col


def calculate_centroids(_features, _labels):
    pos = []
    neg = []
    for f, label in zip(_features, _labels):
        if label == 1:
            pos.append(f)
        else:
            neg.append(f)
    posx = [x[0] for x in pos]
    posy = [x[1] for x in pos]
    negx = [x[0] for x in neg]
    negy = [x[1] for x in neg]
    _px = np.median(posx)
    _py = np.median(posy)
    _nx = np.median(negx)
    _ny = np.median(negy)
    return (_px, _py), (_nx, _ny)


def calculate_distance(p1, p2):
    return np.abs(np.sqrt(((p1[0] - p2[0]) * (p1[0] - p2[0])) + ((p1[1] - p2[1]) * (p1[1] - p2[1]))))


def visualize(features, labels):
    """
    Visualize features from different domains using t-SNE.
    """
    # labels
    # Project:
    # 1 represents linux F;      2 represents linux T
    # 3 represents qemu F;       4 represents qemu T
    # 5 represents ffmpeg F;     6 represents ffmpeg T
    # Language:
    # 1 represents c F;         2 represents c T
    # 3 represents cpp F;       4 represents cpp T
    # 5 represents java F;      6 represents java T

    # map features to 2-d using TSNE
    domain1 = []
    label1 = []
    domain2 = []
    label2 = []
    domain3 = []
    label3 = []
    for feature, label in zip(features, labels):
        if label == 1:
            domain1.append(feature)
            label1.append(0)
        if label == 2:
            domain1.append(feature)
            label1.append(1)
        if label == 3:
            domain2.append(feature)
            label2.append(0)
        if label == 4:
            domain2.append(feature)
            label2.append(1)
        if label == 5:
            domain3.append(feature)
            label3.append(0)
        if label == 6:
            domain3.append(feature)
            label3.append(1)
    print(f"domain1: {len(domain1)}, domain2: {len(domain2)}, domain3: {len(domain3)}")
    domain1 = np.array(domain1)
    domain2 = np.array(domain2)
    domain3 = np.array(domain3)
    label1 = np.array(label1)
    label2 = np.array(label2)
    label3 = np.array(label3)
    dlabel1 = np.array([0] * len(domain1))
    dlabel2 = np.array([1] * len(domain2))
    dlabel3 = np.array([2] * len(domain3))
    features = np.concatenate([domain1, domain2, domain3], axis=0)

    X_tsne = TSNE(n_components=2, random_state=33, init='pca', n_iter=8000, perplexity=30,
                  learning_rate=10).fit_transform(features)

    # X_tsne_1 = X_tsne[:len(domain1), ]
    # X_tsne_2 = X_tsne[len(domain1):len(domain1) + len(domain2), ]
    # X_tsne_3 = X_tsne[len(domain1) + len(domain2):len(domain1) + len(domain2) + len(domain3), ]
    # X = np.concatenate([X_tsne_1, X_tsne_2, X_tsne_3])
    Y1 = np.concatenate([label1, label2, label3])  # label
    Y2 = np.concatenate([dlabel1, dlabel2, dlabel3])  # domain
    return X_tsne, Y1, Y2


def main():
    feature_path = "/mnt/data/block/project/cross-linevul/code/analysis/data_large/"
    # file_names = ["LF2Q_Sysevr_feature_large.pkl"]
    file_names = ["CP2J_codebert_feature_large.pkl",
                  "CP2J_codebert_mmd_feature_large.pkl",
                  "CP2J_codebert_dann_feature_large.pkl",
                  "CP2J_madv_feature_large.pkl"]
    titles = ['a', 'b', 'c', 'd']
    methods = ['CodeBERT', 'CodeBERT-mmd', 'CodeBERT-dann', 'MSVD']

    fig, axs = plt.subplots(2, 4, figsize=(30, 10))
    for index, file_name in enumerate(file_names):
        print(file_name)
        f = open(os.path.join(feature_path, file_name), 'rb')
        data = pickle.load(f)
        features = data["features"]
        labels = data["labels"]
        X_tsne, V_label, D_label = visualize(features, labels)

        X_tsne_vul = X_tsne[V_label == 1]
        X_tsne_nonvul = X_tsne[V_label == 0]

        axs[0, index].scatter(X_tsne_vul[:, 0], X_tsne_vul[:, 1], c='red', marker='o', label='Vul', s=1, alpha=0.5)
        axs[0, index].scatter(X_tsne_nonvul[:, 0], X_tsne_nonvul[:, 1], c='limegreen', marker='<', label='Non-vul', s=1, alpha=0.5)
        if index == 0:
            axs[0, index].legend(markerscale=5.0, loc='upper center', fontsize=18, bbox_to_anchor=(2.15, -0.01), ncol=2, frameon=False)
        title0 = '(' + titles[index] + '.1) ' + methods[index]
        axs[0, index].set_xlabel(title0, fontsize=20, labelpad=45)
        axs[0, index].set_xticks([])
        axs[0, index].set_yticks([])

        X_tsne_d0 = X_tsne[D_label == 0]
        X_tsne_d1 = X_tsne[D_label == 1]
        X_tsne_d2 = X_tsne[D_label == 2]
        axs[1, index].scatter(X_tsne_d0[:, 0], X_tsne_d0[:, 1], c='#E38D26', marker='^', label='C(Source)', s=1, alpha=0.5)
        axs[1, index].scatter(X_tsne_d1[:, 0], X_tsne_d1[:, 1], c='#91BFFA', marker='x', label='C++(Source)', s=1, alpha=0.5)
        axs[1, index].scatter(X_tsne_d2[:, 0], X_tsne_d2[:, 1], c='#832F87', marker='D', label='Java(Target)', s=1, alpha=0.5)
        if index == 0:
            axs[1, index].legend(markerscale=5.0, loc='upper center', fontsize=18, bbox_to_anchor=(2.15, -0.01), ncol=3, frameon=False)
        title1 = '(' + titles[index] + '.2) ' + methods[index]
        axs[1, index].set_xlabel(title1, fontsize=20, labelpad=45)
        axs[1, index].set_xticks([])
        axs[1, index].set_yticks([])
    plt.subplots_adjust(wspace=0.1, hspace=0.4)
    # plt.savefig('project.eps', format='eps', dpi=1000)
    plt.savefig('language1.pdf', format='pdf', dpi=1000)
    plt.show()


if __name__ == "__main__":
    main()
