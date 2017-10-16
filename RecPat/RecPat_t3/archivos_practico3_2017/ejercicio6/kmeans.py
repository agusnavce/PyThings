import math

class K_Means(object):

	def __init__(self, centroids, c =3, tolerance = 0.0001, max_iterations = 500):
		self.c = c
		self.tolerance = tolerance
		self.max_iterations = max_iterations
		self.centroids = centroids
	def fit(self, data):
		for i in range(self.max_iterations):
            self.classes = {}
        	for i in range(self.k):
                self.classes[i] = []
	#
    # 		#find the distance between the point and cluster; choose the nearest centroid
    # 		for features in data:
    # 			distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
    # 			classification = distances.index(min(distances))
    # 			self.classes[classification].append(features)
    #         previous = dict(self.centroids)
	#
    #         #average the cluster datapoints to re-calculate the centroids
    #         for classification in self.classes:
    # 	           self.centroids[classification] = np.average(self.classes[classification], axis = 0)
    #         # see if it converged
	#
    #         isOptimal = True
    #         for centroid in self.centroids:
    #             original_centroid = previous[centroid]
    # 	        curr = self.centroids[centroid]
	#
    # 	        if np.sum((curr - original_centroid)/original_centroid * 100.0) > self.tolerance:
    # 		              isOptimal = False
	#
    # 	    #break out of the main loop if the results are optimal, ie. the centroids don't change their positions much(more than our tolerance)
	#
    #         if isOptimal:
    #             break
