from sklearn import datasets
from sklearn.cluster import KMeans as KM
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.manifold import TSNE

from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, \
    adjusted_rand_score, adjusted_mutual_info_score

from skimage import filters
from skimage.feature import hog
from skimage.exposure import adjust_gamma

from preprocessing import tf_idf_matrix
from preprocessing import SLTransformer

from evaluation import visualization_data

import numpy as np
import random
import matplotlib.pyplot as plt

import cv2
import sys

np.random.seed(42)
random.seed(42)
categories = ['comp.graphics', 'comp.os.ms-windows.misc',
              'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
              'rec.autos', 'rec.motorcycles',
              'rec.sport.baseball', 'rec.sport.hockey']
dataset = datasets.fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=None,
                                      remove=('headers', 'footers'))
fp = open('Result/metrics.txt', 'w+')

# Question 1
data = dataset.data
target = dataset.target
data_matrix = tf_idf_matrix(data)
fp.write('tf_idf_matrix shape: {}\n'.format(data_matrix.shape))

# Question 2
target = [1 if i > 3 else 0 for i in target]

cluster = KM(n_clusters=2, random_state=0, max_iter=1000, n_init=30)
cluster.fit(data_matrix)
predict_target = cluster.labels_
fp.write("Contingency table: \n{}\n".format(contingency_matrix(target, predict_target)))

# Question 3
metrics_dict = {"homogeneity_score": homogeneity_score, 'completeness_score': completeness_score,
                "v_measure_score": v_measure_score, "adjusted_rand_score": adjusted_rand_score,
                "adjusted_mutual_info_score": adjusted_mutual_info_score}

for metrics_name in metrics_dict.keys():
    fp.write(metrics_name + ": {: 0.4f}\n".format(metrics_dict[metrics_name](target, predict_target)))

# Question 4
N_features = 1000
svd = TruncatedSVD(n_components=N_features)
svd_train_1000 = svd.fit_transform(data_matrix)
variance_ratio = svd.explained_variance_ratio_
percent = []
ratio_sum = 0
for r in variance_ratio:
    ratio_sum += r
    percent.append(ratio_sum)

plt.plot(percent)
plt.xlabel(' Principal components ')
plt.ylabel('Percent of variance')
plt.savefig('percentage_of_variance.jpg')
plt.show()

# Question 5
N_components_tuple = [1, 2, 3, 5, 10, 20, 50, 100, 300]
svd_metrics = {}
nmf_metrics = {}

for r in N_components_tuple:

    nmf = NMF(n_components=r, init='random', random_state=0)
    svd_train = svd_train_1000[:, :r]
    nmf_train = nmf.fit_transform(data_matrix)
    svd_cluster = KM(n_clusters=2, random_state=0, max_iter=1000, n_init=30)
    nmf_cluster = KM(n_clusters=2, random_state=0, max_iter=1000, n_init=30)
    svd_cluster.fit(svd_train)
    nmf_cluster.fit(nmf_train)
    svd_predict = svd_cluster.labels_
    nmf_predict = nmf_cluster.labels_
    for metrics_name in metrics_dict.keys():
        metrics_f = metrics_dict[metrics_name]
        if metrics_name not in svd_metrics.keys():
            svd_metrics[metrics_name] = []
        if metrics_name not in nmf_metrics.keys():
            nmf_metrics[metrics_name] = []
        svd_metrics[metrics_name].append(metrics_f(target, svd_predict))
        nmf_metrics[metrics_name].append(metrics_f(target, nmf_predict))

# print result
result_dict = {'SVD': svd_metrics, 'NMF': nmf_metrics}
for method_name in result_dict.keys():
    fp.write('\n' + method_name + ' result: \n')
    d = result_dict[method_name]
    for metrics_name in d.keys():
        fp.write(metrics_name + ": {}\n".format(d[metrics_name]))

# plot histogram
color_tuple = ['r', 'y', 'b', 'g', 'c']
bar_width = 0.15
for method_name in result_dict.keys():
    d = result_dict[method_name]
    x = np.arange(len(N_components_tuple))
    i = 0
    for metrics_name in d.keys():
        plt.bar(x + i * bar_width, d[metrics_name], bar_width, align='center', color=color_tuple[i], label=metrics_name)
        i += 1

    plt.xlabel("r")
    plt.ylabel('Score')
    plt.xticks(x + bar_width * (i - 1) / 2, N_components_tuple)
    plt.legend()
    plt.savefig(method_name + '_metrics' + '.jpg')
    plt.show()

# Question 7
best_nmf_r = 10
best_svd_r = 50

svd = TruncatedSVD(n_components=best_svd_r)
nmf = NMF(n_components=best_nmf_r)

best_svd_model = TruncatedSVD(n_components=best_svd_r)
best_nmf_model = NMF(n_components=best_nmf_r, init='random', random_state=0)

cluster = KM(n_clusters=2, random_state=0, max_iter=1000, n_init=30)

transformer_dict = {
    'None_transformation': SLTransformer(False, False),
    'Scaling_transformation': SLTransformer(True, False),
    'Logarithm_transformation': SLTransformer(False, True),
    'All_transformation': SLTransformer(True, True)
}

visualization_data(cluster, None, best_svd_model, data_matrix, target, metrics_dict, fp, 'best_svd')
visualization_data(cluster, None, best_nmf_model, data_matrix, target, metrics_dict, fp, 'best_nmf')

# Question 8, 10
for name in ['None_transformation', 'Scaling_transformation']:
    visualization_data(cluster, transformer_dict[name], best_svd_model, data_matrix, target, metrics_dict, fp,
                       'SVD_' + name)

for name in transformer_dict.keys():
    visualization_data(cluster, transformer_dict[name], best_nmf_model, data_matrix, target, metrics_dict, fp,
                       'NMF_' + name)

fp.close()

# Part 2
X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
X_size = X.shape[0]
train_samples = 10000
indices = np.random.choice(X_size, train_samples)
X_train = X[indices]
y = y[indices]

X_train = X_train.reshape((train_samples, 28, 28))

cell_size = 7
block_size = 2
hog_feature_d = ((28 // cell_size) + 1 - block_size) ** 2
hog_train = np.zeros((train_samples, np.int(hog_feature_d * 36)))
for i in range(train_samples):
    image = X_train[i]
    filtered_image = filters.gaussian(image, sigma=0.4)
    adjusted_img = adjust_gamma(filtered_image, gamma=0.8)
    hog_feature = hog(adjusted_img, pixels_per_cell=(cell_size, cell_size), cells_per_block=(block_size, block_size))
    hog_train[i, :] = hog_feature

cluster = KM(n_clusters=10, random_state=0, max_iter=1000, n_init=30)
cluster.fit(hog_train)
y_pred = cluster.labels_

for metrics_name in metrics_dict.keys():
    print(metrics_name + ": {: 0.4f}\n".format(metrics_dict[metrics_name](y, y_pred)))

tsne = TSNE(n_components=2)
tsne.fit(hog_train)
result_array = tsne.embedding_
cpool = ['#bd2309', '#bbb12d', '#1480fa', '#14fa2f', '#000000',
         '#faf214', '#2edfea', '#ea2ec4', '#ea2e40', '#cdcdcd']
plt.scatter(result_array[:, 0], result_array[:, 1], c=[cpool[i] for i in y_pred])
plt.title('Cluster result with T-SNE visualization')
plt.savefig('cluster_result.eps')
plt.show()

plt.scatter(result_array[:, 0], result_array[:, 1], c=[cpool[i] for i in y])
plt.title('Ground truth with T-SNE visualization')
plt.savefig('ground_truth.eps')
plt.show()

# part 3
image = cv2.imread("test.jpg")
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
original = img.copy()

image_height = img.shape[0]
image_width = img.shape[1]


def normalized(my_img, image_h, image_w):
    norm = np.zeros((image_h, image_w, 3), np.float32)
    b = my_img[:, :, 0]
    g = my_img[:, :, 1]
    r = my_img[:, :, 2]
    sum = b + g + r
    norm[:, :, 0] = b / (sum+sys.float_info.epsilon) * 255.0
    norm[:, :, 1] = g / (sum+sys.float_info.epsilon) * 255.0
    norm[:, :, 2] = r / (sum+sys.float_info.epsilon) * 255.0
    norm_rgb = cv2.convertScaleAbs(norm)
    return norm_rgb


temp_img = normalized(img, image_height, image_width)
vectorized = temp_img[:, :, :2].reshape((-1, 2))
vectorized = np.float32(vectorized)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
attempts = 10

ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
label = label.reshape(image_height, image_width)
cluster_result = np.zeros_like(img)
for i in range(image_height):
    for j in range(image_width):
        cluster_result[i, j, label[i, j]] = 255

figure_size = 15
plt.figure(figsize=(figure_size, figure_size))
plt.subplot(1, 2, 1)
plt.imshow(original)
plt.xticks([])
plt.yticks([])
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(cluster_result)
plt.title('Clustered Image when K = %i' % K)
plt.xticks([])
plt.yticks([])
plt.savefig('color_cluster.eps')
plt.show()
