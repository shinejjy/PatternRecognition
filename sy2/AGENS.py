import numpy as np
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt


class Cluster:
    def __init__(self, items, id):
        self.id = id
        self.items = items


class AGENS:
    def __init__(self, data=None, pre_distance=None, method='single'):
        self.data = data
        self.method = method
        self.init_distance(pre_distance)

    def init_distance(self, pre_distance=None):
        if pre_distance is not None:
            self.distance = pre_distance
            self.N = self.distance.shape[0]
            return
        self.N = self.data.shape[0]
        self.distance = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(i + 1, self.N):
                self.distance[i, j] = np.linalg.norm(self.data[i] - self.data[j])
                self.distance[j, i] = self.distance[i, j]

    def distance_c2c(self, cluster1, cluster2):
        if self.method == 'single':
            min_dis = float('inf')
            for i in cluster1:
                for j in cluster2:
                    min_dis = min(min_dis, self.distance[i, j])
            return min_dis

        elif self.method == 'complete':
            max_dis = 0
            for i in cluster1:
                for j in cluster2:
                    max_dis = min(max_dis, self.distance[i, j])
            return max_dis

        elif self.method == 'average':
            dist_sum = 0
            for i in cluster1:
                for j in cluster2:
                    dist_sum += self.distance[i, j]
            avg_dist = dist_sum / (len(cluster1) * len(cluster2))
            return avg_dist

    def start(self, threshold=float('inf'), display=False):
        n_clusters = self.N
        clusters = [Cluster([i], i) for i in range(self.N)]
        linkage_matrix = []  # 用于构造链接矩阵的列表

        while len(clusters) > 1:
            min_dist = float('inf')
            merge_i = -1
            merge_j = -1

            # 寻找距离小于阈值的两个簇
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    cluster_i = clusters[i]
                    cluster_j = clusters[j]
                    dist_ij = self.distance_c2c(cluster_i.items, cluster_j.items)
                    if dist_ij < min_dist:
                        min_dist = dist_ij
                        merge_i = i
                        merge_j = j

            # 构造链接矩阵的条目并添加到列表
            new_entry = [clusters[merge_i].id, clusters[merge_j].id, min_dist,
                         len(clusters[merge_i].items + clusters[merge_j].items)]
            linkage_matrix.append(new_entry)

            # 如果距离小于阈值，合并两个簇
            if min_dist <= threshold:
                new_cluster_items = clusters[merge_i].items + clusters[merge_j].items
                if display:
                    print([cluster.items for cluster in clusters], end=' -> ')
                clusters.append(Cluster(new_cluster_items, n_clusters))
                n_clusters = n_clusters + 1
                del clusters[merge_j]
                del clusters[merge_i]

                if display:
                    print([cluster.items for cluster in clusters])
            else:
                # 如果距离大于阈值，停止聚类
                break

        return [cluster.items for cluster in clusters], np.array(linkage_matrix)

if __name__ == '__main__':

    # 给定数据
    data = np.array([
        [0, 3, 1, 2, 0],
        [1, 3, 0, 1, 0],
        [3, 3, 0, 0, 1],
        [1, 1, 0, 2, 0],
        [3, 2, 1, 2, 1],
        [4, 1, 1, 1, 0]
    ])

    # 使用AGENS算法进行层次聚类
    agens = AGENS(data, method='single')

    # 1) 若给定的阈值为threshold，给出每次聚类时类别的合并以及最终的聚类结果
    threshold = np.sqrt(5)  # 根据需要调整阈值
    clusters = agens.start(threshold, display=True)[0]
    print("Cluster assignments with threshold {}:".format(threshold))
    print(clusters)

    # 2) 分别给定不同的阈值，观察和分析聚类结果
    thresholds = [np.sqrt(3), np.sqrt(4), np.sqrt(6)]  # 可根据需要调整不同的阈值
    for threshold in thresholds:
        clusters = agens.start(threshold)[0]
        print("Cluster assignments with threshold {}:".format(threshold))
        print(clusters)

    # 3) 绘制出层次聚类过程的树状图
    linkage_matrix = agens.start()[1]
    plt.figure(figsize=(10, 5))
    dendrogram(linkage_matrix, labels=np.array([str(i) for i in range(1, len(data) + 1)]))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Data Points')
    plt.ylabel('Distance')
    plt.show()

    # 1) 绘制阈值sqrt5的聚类过程
    dendrogram(linkage_matrix, labels=np.array([str(i) for i in range(1, len(data) + 1)]), color_threshold=np.sqrt(5) + 0.1)
    plt.figure(figsize=(10, 5))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Data Points')
    plt.ylabel('Distance')
    plt.show()
