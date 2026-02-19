import torch
from sklearn.cluster import KMeans
import random
import copy
import matplotlib.pyplot as plt
import pickle
import os.path
import numpy as np


class ClusterDirSampler:
    def __init__(self, train_cameras, settings=dict()):
        camera_locations = (
            torch.stack([cam.camera_center for cam in train_cameras]).cpu().numpy()
        )
        camera_loc_max, camera_loc_min = np.max(camera_locations), np.min(
            camera_locations
        )
        # Normalize
        camera_locations = (camera_locations - camera_loc_min) / (
            camera_loc_max - camera_loc_min
        )
        camera_dirs = (
            torch.stack([cam.world_view_transform[:3, 2] for cam in train_cameras])
            .cpu()
            .numpy()
        )

        self.camera_vector = np.concatenate((camera_locations, camera_dirs), axis=1)
        self.batch_size = settings["batch_size"]
        self.path = settings["path"]
        print("Clustering Started...")
        kmeans = KMeans(n_clusters=self.batch_size, random_state=0, init="random").fit(
            self.camera_vector
        )
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        self.n_change = 1
        self.plot_cluster(self.camera_vector, labels, centroids)
        self.initial_cluster_dict = self.initialize_clusters(labels)
        print("Clustering Finished...")

        self.current_cluster_dict = copy.deepcopy(self.initial_cluster_dict)
        self.train_cameras = train_cameras
        self.last_cameras = [0] * self.batch_size

    def plot_cluster(self, camera_vector, labels, centroids):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            camera_vector[:, 0],
            camera_vector[:, 1],
            camera_vector[:, 2],
            c=labels,
            cmap="gist_rainbow",
            s=50,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.quiver(
            camera_vector[:, 0],
            camera_vector[:, 1],
            camera_vector[:, 2],  # origins
            camera_vector[:, 3],
            camera_vector[:, 4],
            camera_vector[:, 5],  # directions
            length=0.03,
            color="red",
            arrow_length_ratio=0.2,
        )

        # Visualize the direction

        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            centroids[:, 2],
            c="black",
            marker="X",
            s=250,
            label="Centroids",
        )
        plt.title("Kmeans Clustering")
        pickle.dump(
            fig,
            open(os.path.join(self.path, f"Clusters_{self.n_change}.fig.pickle"), "wb"),
        )
        plt.close(fig)
        self.n_change = self.n_change + 1

    def initialize_clusters(self, labels):
        cluster_dict = {}
        for idx, label in enumerate(labels):
            if label not in cluster_dict:
                cluster_dict[label] = []
            cluster_dict[label].append(idx)
        return cluster_dict

    def sample(self, cluster):
        if len(self.current_cluster_dict[cluster]) == 0:
            self.current_cluster_dict[cluster] = self.initial_cluster_dict[
                cluster
            ].copy()

        cam_indices = self.current_cluster_dict[cluster]
        selected_idx = random.randint(0, len(cam_indices) - 1)
        selected_cam_idx = cam_indices[selected_idx]
        cam_indices.pop(selected_idx)
        return selected_cam_idx

    def get_camera(self, current_batch):
        idx = self.sample(current_batch)
        self.last_cameras[current_batch] = idx
        return self.train_cameras[idx]

    def update(self, key: str, value):
        if key == "batchsize_sched" and value != self.batch_size:
            self.batch_size = value
            kmeans = KMeans(
                n_clusters=self.batch_size, random_state=0, init="random"
            ).fit(self.camera_vector)
            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_
            self.plot_cluster(self.camera_vector, labels, centroids)

            self.initial_cluster_dict = self.initialize_clusters(labels)
            self.current_cluster_dict = copy.deepcopy(self.initial_cluster_dict)
            self.last_cameras = [0] * self.batch_size
