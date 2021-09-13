"""
This is a dummy file for HW5 of CSE353 Machine Learning, Fall 2020
You need to provide implementation for this file

By: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
Created: 26-Oct-2020
Last modified: 26-Oct-2020
"""
#print("WE GOT IN HERE")
import random
#print("AFTER")
import sys
#print(sys.executable)
from sklearn.cluster import KMeans
#print("AFTER AFTER")


class TrackletClustering(object):
    """
    You need to implement the methods of this class.
    Do not change the signatures of the methods
    """

    def __init__(self, num_cluster):
        self.num_cluster = num_cluster
        self.tracklets = []
        self.model = None

    def add_tracklet(self, tracklet):
        "Add a new tracklet into the database"
        center_vec = []
        center_point_x = tracklet['tracks'][0][3] - tracklet['tracks'][0][1]
        center_point_y = tracklet['tracks'][0][2] - tracklet['tracks'][0][4]  
        #center_vec.append([center_point_x,center_point_y])

        center_point_x_end = tracklet['tracks'][-1][3] - tracklet['tracks'][-1][1]
        center_point_y_end = tracklet['tracks'][-1][2] - tracklet['tracks'][-1][4]

        center_vec = [center_point_x, center_point_y, center_point_x_end, center_point_y_end]


        tracklet["feature vect"] = center_vec

        self.tracklets.append(tracklet)

        pass

    def build_clustering_model(self):
        "Perform clustering algorithm"
        #get data for each feature vect
        pure_data = []
        for i in range(0,len(self.tracklets)):
            pure_data.append(self.tracklets[i]["feature vect"])
            
        kmeans = KMeans(self.num_cluster).fit(pure_data)
        self.model = kmeans
        pass

    def get_cluster_id(self, tracklet):
        """
        Assign the cluster ID for a tracklet. This funciton must return a non-negative integer <= num_cluster
        It is possible to return value 0, but it is reserved for special category of abnormal behavior (for Question 2.3)
        """
        center_point_x = tracklet['tracks'][0][3] - tracklet['tracks'][0][1]
        center_point_y = tracklet['tracks'][0][2] - tracklet['tracks'][0][4]  
        #center_vec.append([center_point_x,center_point_y])

        center_point_x_end = tracklet['tracks'][-1][3] - tracklet['tracks'][-1][1]
        center_point_y_end = tracklet['tracks'][-1][2] - tracklet['tracks'][-1][4]

        center_vec = [[center_point_x, center_point_y, center_point_x_end, center_point_y_end]]
        
        label = int(self.model.predict(center_vec)[0])

        return label
