"""
Copyright 2021 Anderson Faustino da Silva.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np

from sklearn import cluster
from sklearn.metrics import pairwise_distances_argmin_min


class classproperty(property):
    """class property decorator."""

    def __get__(self, cls, owner):
        """Decorate."""
        return classmethod(self.fget).__get__(None, owner)()


class Clustering:
    """Static class to cluster data."""

    __version__ = '2.0.0'

    __cluster = None

    @staticmethod
    def mini_batch_kmeans(n_clusters=8,
                          init='k-means++',
                          max_iter=100,
                          batch_size=100,
                          compute_labels=True,
                          tol=0.0,
                          max_no_improvement=10,
                          init_size=None,
                          n_init=3,
                          reassignment_ratio=0.01,
                          random_state=None):
        """Mini Batch KMeans clustering.

        Parameters
        ----------
        n_clusters :

        init :

        max_iter :

        batch_size :

        compute_labels :

        tol :

        max_no_improvement :

        init_size :

        n_init :

        reassignment_ratio :

        random_state=None :
        """
        Clustering.__cluster = cluster.MiniBatchKMeans(
                n_clusters=n_clusters,
                init=init,
                max_iter=max_iter,
                batch_size=batch_size,
                compute_labels=compute_labels,
                tol=tol,
                max_no_improvement=max_no_improvement,
                init_size=init_size * batch_size,
                n_init=n_init,
                reassignment_ratio=reassignment_ratio,
                random_state=random_state if random_state else None
            )

    @staticmethod
    def kmeans(n_clusters=8,
               init='k-means++',
               n_init=10,
               max_iter=300,
               tol=0.0001,
               copy_x=True,
               algorithm='auto',
               random_state=None):
        """K-Means clustering.

        Parameters
        ----------
        n_clusters :

        init :

        n_init :

        max_iter :

        tol :

        precompute_distances :

        copy_x :

        algorithm :

        random_state=None :
        """
        Clustering.__cluster = cluster.KMeans(
                n_clusters=n_clusters,
                init=init,
                n_init=n_init,
                max_iter=max_iter,
                tol=tol,
                copy_x=copy_x,
                algorithm=algorithm,
                random_state=random_state if random_state else None
            )

    @staticmethod
    def affinity_progagation(damping=0.5,
                             max_iter=200,
                             convergence_iter=15,
                             copy=True,
                             preference=None,
                             affinity='euclidean',
                             random_state=None):
        """Affinity Propagation cluster.

        Parameters
        ----------
        damping :

        max_iter :

        convergence_iter :

        copy :

        preference :

        affinity :

        random_state :
        """
        Clustering.__cluster = cluster.AffinityPropagation(
                damping=damping,
                max_iter=max_iter,
                convergence_iter=convergence_iter,
                copy=copy,
                preference=preference,
                affinity=affinity,
                random_state=random_state if random_state else 0
            )

    @staticmethod
    def mean_shift(bandwidth=None,
                   bin_seeding=False,
                   min_bin_freq=1,
                   cluster_all=True):
        """Mean Shift cluster.

        Parameters
        ----------
        bandwidth :

        bin_seeding :

        min_bin_freq :

        cluster_all :
        """
        Clustering.__cluster = cluster.MeanShift(
                bandwidth=bandwidth,
                bin_seeding=bin_seeding,
                min_bin_freq=min_bin_freq,
                cluster_all=cluster_all
            )

    @staticmethod
    def spectral_clustering(n_clusters=8,
                            eigen_solver=None,
                            n_init=10,
                            gamma=1.0,
                            affinity='rbf',
                            n_neighbors=10,
                            eigen_tol=0.0,
                            assign_labels='kmeans',
                            degree=3,
                            coef0=1,
                            random_state=None):
        """Spectral cluster.

        Parameters
        ----------
        n_clusters :

        eigen_solver :

        n_init :

        gamma :

        affinity :

        n_neighbors :

        eigen_tol :

        assign_labels :

        degree :

        coef0 :

        random_state :
        """
        Clustering.__cluster = cluster.SpectralClustering(
                n_clusters=n_clusters,
                eigen_solver=eigen_solver,
                n_init=n_init,
                gamma=gamma,
                affinity=affinity,
                n_neighbors=n_neighbors,
                eigen_tol=eigen_tol,
                assign_labels=assign_labels,
                degree=degree,
                coef0=coef0,
                random_state=random_state if random_state else None
            )

    @staticmethod
    def agglomerative_clustering(n_clusters=2,
                                 affinity='euclidean',
                                 linkage='ward'):
        """Agglomerative cluster.

        Parameters
        ----------
        n_clusters :

        affinity :

        linkage :
        """
        Clustering.__cluster = cluster.AgglomerativeClustering(
                n_clusters=n_clusters,
                affinity=affinity,
                linkage=linkage
            )

    @staticmethod
    def dbscan(eps,
               min_samples,
               algorithm,
               leaf_size):
        """DBSCAN cluster.

        Parameters
        ----------
        eps :

        min_samples :

        algorithm :

        leaf_size :
        """
        Clustering.__cluster = cluster.DBSCAN(
                eps=eps,
                min_samples=min_samples,
                algorithm=algorithm,
                leaf_size=leaf_size
            )

    @staticmethod
    def birch(threshold=0.5,
              branching_factor=50,
              n_clusters=3,
              compute_labels=True,
              copy=True):
        """Birch cluster.

        Parameters
        ----------
        threshold :

        branching_factor :

        n_clusters :

        compute_labels :

        copy :
        """
        Clustering.__cluster = cluster.Birch(
                threshold=threshold,
                branching_factor=branching_factor,
                n_clusters=n_clusters,
                compute_labels=compute_labels,
                copy=copy
            )

    @staticmethod
    def fit(data):
        """Fit a training data.

        Parameter
        ---------
        data : dataFrame
        """
        Clustering.__cluster.fit(data)

    @staticmethod
    def predict(test_data, training_data=None):
        """Predict the clusters.

        Parameters
        ----------
        test_data : dataFrame

        training_data : dataFrame
        """
        if training_data:
            Clustering.__cluster.fit(training_data)

        predict = Clustering.__cluster.predict(test_data)
        predict = np.asscalar(predict[0])

        clusters = [
            test_data.index[i]
            for i, label in enumerate(Clustering.__cluster.labels_)
            if (label == predict)
        ]

        return clusters

    @staticmethod
    def clusters_and_centroids(data, fit_data=None):
        """Get the clusters and centroids.

        Parameter
        ---------
        data : dataFrame

        fit_data : DataFrame

        Return
        ---------
        cluster : dict

        centroids : dict
        """
        if fit_data:
            Clustering.__cluster.fit(fit_data)

        # The clusters
        if hasattr(Clustering.__cluster, 'labels_'):
            y_pred = Clustering.__cluster.labels_.astype(np.int)
        else:
            y_pred = Clustering.__cluster.predict(data)

        clusters = {}
        for i, label in enumerate(y_pred):
            lbl = np.asscalar(label)
            if lbl not in clusters.keys():
                clusters[lbl] = []
            clusters[lbl].append(data.index[i])

        # The centroids
        centroids = {}
        if hasattr(Clustering.__cluster, 'cluster_centers_'):
            closest, _ = pairwise_distances_argmin_min(
                        Clustering.__cluster.cluster_centers_,
                        data
                    )
            for i, point in enumerate(closest):
                centroid = np.asscalar(point)
                centroids[i] = data.index[centroid]

        return clusters, centroids
