import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
import seaborn as sns
from sy2.AGENS import AGENS

# 输入数据
data = np.array([
    [170, 58],
    [172, 60],
    [173, 62],
    [155, 68],
    [158, 70],
    [179, 56],
    [182, 58]
])

# 正规化数据
normalized_data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
print(normalized_data)


# 计算模糊关系矩阵
def calculate_fuzzy_relation_matrix(data, metric='euclidean'):
    n = len(data)
    fuzzy_relation_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if metric == 'euclidean':
                fuzzy_relation_matrix[i, j] = np.linalg.norm(data[i] - data[j])
                max_distance = np.max(fuzzy_relation_matrix)
                fuzzy_relation_matrix = (max_distance - fuzzy_relation_matrix) / max_distance
            elif metric == 'dot_product':
                fuzzy_relation_matrix[i, j] = np.dot(data[i], data[j])
                max_distance = np.max(fuzzy_relation_matrix)
                fuzzy_relation_matrix = fuzzy_relation_matrix / max_distance
                np.fill_diagonal(fuzzy_relation_matrix, 1)
            elif metric == 'min_max':
                min_values = np.minimum(data[i], data[j])
                max_values = np.maximum(data[i], data[j])
                fuzzy_relation_matrix[i, j] = np.sum(min_values) / np.sum(max_values)
            else:
                raise ValueError(f"Unsupported metric: {metric}")

    return fuzzy_relation_matrix


def calculate_equal_matrix(R):
    a = R.shape[0]

    # 创建一个空的 C 矩阵
    C = np.zeros((a, a))

    while True:
        for i in range(a):
            for k in range(a):
                C[i, k] = np.max(np.min([R[i, :], R[:, k].T], axis=0))
        if np.array_equal(R, C):
            print('模糊等价矩阵：C=\n{}'.format(C))
            break
        else:
            R = C

    return C


def lambdaCutMatrix(B, lam):
    return (B >= lam).astype(int)


def getClusters(B):
    def find(x):
        if x == pre[x]:
            return x
        else:
            return find(pre[x])

    a = B.shape[0]
    pre = [i for i in range(a)]
    for i in range(a):
        for j in range(a):
            if B[i, j] == 1:
                fx = find(i)
                fy = find(j)
                if fx != fy:
                    pre[fy] = fx

    clusters = {}
    for i in range(a):
        f = find(i)
        clusters.setdefault(f, [])
        clusters[f].append(i)

    for i, value in enumerate(clusters.values()):
        print(f'Cluster {i + 1}: {value}')


R = calculate_fuzzy_relation_matrix(normalized_data, metric='dot_product')

# 绘制热力图
sns.heatmap(R, annot=True, cmap="YlGnBu", fmt=".2f", xticklabels=False, yticklabels=False)
plt.title('Fuzzy Matrix')
plt.show()

C = calculate_equal_matrix(R)

sns.heatmap(C, annot=True, cmap="YlGnBu", fmt=".2f", xticklabels=False, yticklabels=False)
plt.title('Fuzzy Equal Matrix')
plt.show()

B = lambdaCutMatrix(C, 0.5)

sns.heatmap(B, annot=True, cmap="YlGnBu", xticklabels=False, yticklabels=False)
plt.title('Lambda Cut Matrix')
plt.show()

getClusters(B)

# 生成聚类树谱图
agens = AGENS(method='average', pre_distance=C)
linkage_matrix = agens.start()[1]
plt.figure(figsize=(10, 5))

print(linkage_matrix)

# 绘制树谱图
dendrogram(linkage_matrix, labels=np.array(range(len(B))), leaf_font_size=8)
plt.title('Fuzzy Clustering Dendrogram')
plt.ylim([0, 1])

plt.xlabel('Data Points')
plt.ylabel('$\lambda$')
plt.show()
