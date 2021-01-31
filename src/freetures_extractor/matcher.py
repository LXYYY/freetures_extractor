import open3d as o3d


class Matcher(object):
    def __init__(self, param):
        self.n_neighbors = param['n_neighbors']
        self.n_max_ransac = param['n_max_ransac']
        self.max_ransac_valid = param['max_ransac_valid']
        self.max_corr_distance = param['max_corr_distance']

    def compute_registration(self, kp_q, kp_t, desc_q, desc_t):
        max_validation = int(min(len(kp_q), len(kp_t)) * self.max_ransac_valid)
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(kp_q)
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(kp_t)
        source_feature = o3d.registration.Feature()
        source_feature.data = desc_q
        target_feature = o3d.registration.Feature()
        target_feature.data = desc_t
        result = o3d.registration.registration_ransac_based_on_feature_matching(
            source, target, source_feature, target_feature, self.max_corr_distance,
            o3d.registration.TransformationEstimationPointToPoint(False), 3, [
                o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.registration.CorrespondenceCheckerBasedOnDistance(
                    self.max_corr_distance)
            ],
            o3d.registration.RANSACConvergenceCriteria(
                max_iteration=self.n_max_ransac,
                max_validation=max_validation))
        return result
