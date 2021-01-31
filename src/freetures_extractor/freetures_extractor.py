import os
import threading
import struct
import Queue
import rospy
from math import pi
from datetime import timedelta
import numpy as np
from std_msgs.msg import Header
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import tensorflow as tf
from timeit import default_timer as timer

from . import matcher


class FreeturesExtractor(object):
    def __init__(self, param):
        self.esdf_points = None
        self.esdf_queue = Queue.Queue()
        self.log_dir = param['log_dir']
        self.param = param
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.pc_input_metadata = os.path.join(self.log_dir,
                                              "pc_input_metadata.tsv")
        with open(self.pc_input_metadata, "w") as f:
            for subwords in ['x', 'y', 'z', 'dist']:
                f.write("{}\n".format(subwords))
        self.r_frame = param['r_frame']
        self.vis_scale = param['vis_scale']
        self.ang_res = float(pi / param['n_div'])
        self.n_divs = int(param['n_div'])
        self.matcher = matcher.Matcher(param)
        self.submap_dict = {}
        assert self.n_divs // 2 != 0  # make life easier
        self.n_dims = 2 * self.n_divs**2
        self.solid_angle = self.__compute_solid_angle(
            np.concatenate(
                (np.arange(self.n_divs), np.arange(-self.n_divs,
                                                   0))).repeat(self.n_divs))
        self.dist_pc_pub = rospy.Publisher('distance', PointCloud2)
        self.det_pc_pub = rospy.Publisher('det', PointCloud2)
        self.kp_pc_pub = rospy.Publisher('keypoints', PointCloud2)
        self.lrf_pub = rospy.Publisher('lrc', MarkerArray)
        self.grad_pub = rospy.Publisher('gradient', MarkerArray)
        self.grad_gauss_sum = 0
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        self.process_thread = threading.Thread(target=self._process)
        self.process_thread.start()

    def _get_gaussian(self, sigma):
        x = np.arange(-2, 3, 1)
        g_kernel = np.exp(-(x)**2 / (2 * sigma**2))
        gaussian = tf.convert_to_tensor(g_kernel, dtype=tf.float32)
        x_gaussian = gaussian[:, tf.newaxis, tf.newaxis, tf.newaxis,
                              tf.newaxis]
        y_gaussian = gaussian[tf.newaxis, :, tf.newaxis, tf.newaxis,
                              tf.newaxis]
        z_gaussian = gaussian[tf.newaxis, tf.newaxis, :, tf.newaxis,
                              tf.newaxis]
        return [x_gaussian, y_gaussian, z_gaussian], np.sum(g_kernel)

    def _gaussian(self, in_tensor, gaussian):
        pc_gauss_x = tf.nn.conv3d(in_tensor,
                                  gaussian[0],
                                  strides=[1, 1, 1, 1, 1],
                                  padding="SAME")
        pc_gauss_y = tf.nn.conv3d(in_tensor,
                                  gaussian[1],
                                  strides=[1, 1, 1, 1, 1],
                                  padding="SAME")
        pc_gauss_z = tf.nn.conv3d(in_tensor,
                                  gaussian[2],
                                  strides=[1, 1, 1, 1, 1],
                                  padding="SAME")
        pc_gauss_xyz = tf.math.add(tf.math.add(pc_gauss_x, pc_gauss_y),
                                   pc_gauss_z) / 3
        return pc_gauss_xyz

    def _process(self):
        with self.session.graph.as_default():

            # prepare sobel and gaussian kernel
            sobel = tf.constant([-1, 0, 1], dtype=tf.float32)
            x_sobel = sobel[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
            y_sobel = sobel[tf.newaxis, :, tf.newaxis, tf.newaxis, tf.newaxis]
            z_sobel = sobel[tf.newaxis, tf.newaxis, :, tf.newaxis, tf.newaxis]

            grad_gauss, self.grad_gauss_sum = self._get_gaussian(
                2)  # sigma_grad
            desc_gauss, _ = self._get_gaussian(
                self.param['r_frame'])  # sigma_desc

            # compute det of hessian
            pc_in = tf.placeholder(dtype=tf.float32)
            pc_gauss_xyz = self._gaussian(pc_in, grad_gauss)

            gx = tf.nn.conv3d(pc_gauss_xyz,
                              x_sobel,
                              strides=[1, 1, 1, 1, 1],
                              padding="SAME")
            hxx = tf.nn.conv3d(gx,
                               x_sobel,
                               strides=[1, 1, 1, 1, 1],
                               padding="SAME")
            hxy = tf.nn.conv3d(gx,
                               y_sobel,
                               strides=[1, 1, 1, 1, 1],
                               padding="SAME")
            hxz = tf.nn.conv3d(gx,
                               z_sobel,
                               strides=[1, 1, 1, 1, 1],
                               padding="SAME")
            gy = tf.nn.conv3d(pc_gauss_xyz,
                              y_sobel,
                              strides=[1, 1, 1, 1, 1],
                              padding="SAME")
            hyx = tf.nn.conv3d(gy,
                               x_sobel,
                               strides=[1, 1, 1, 1, 1],
                               padding="SAME")
            hyy = tf.nn.conv3d(gy,
                               y_sobel,
                               strides=[1, 1, 1, 1, 1],
                               padding="SAME")
            hyz = tf.nn.conv3d(gy,
                               z_sobel,
                               strides=[1, 1, 1, 1, 1],
                               padding="SAME")
            gz = tf.nn.conv3d(pc_gauss_xyz,
                              z_sobel,
                              strides=[1, 1, 1, 1, 1],
                              padding="SAME")
            hzx = tf.nn.conv3d(gz,
                               x_sobel,
                               strides=[1, 1, 1, 1, 1],
                               padding="SAME")
            hzy = tf.nn.conv3d(gz,
                               y_sobel,
                               strides=[1, 1, 1, 1, 1],
                               padding="SAME")
            hzz = tf.nn.conv3d(gz,
                               z_sobel,
                               strides=[1, 1, 1, 1, 1],
                               padding="SAME")

            # TODO(mikexyl): check if order is correct
            h_row1 = tf.squeeze(tf.stack([hxx, hxy, hxz], axis=4), axis=5)
            h_row2 = tf.squeeze(tf.stack([hyx, hyy, hyz], axis=4), axis=5)
            h_row3 = tf.squeeze(tf.stack([hzx, hzy, hzz], axis=4), axis=5)
            hessian = tf.stack([h_row1, h_row2, h_row3], axis=4)
            det = tf.matrix_determinant(hessian)[:, :, :, :, tf.newaxis]
            local_max = tf.nn.max_pool3d(det, [
                self.param['r_local_max'], self.param['r_local_max'],
                self.param['r_local_max']
            ],
                strides=1,
                padding='SAME')
            non_zeros = tf.cast(tf.math.not_equal(pc_in, tf.constant(0.0)),
                                tf.float32)
            non_zeros = tf.cast(
                -tf.nn.max_pool3d(-non_zeros, [
                    self.param['r_frame'], self.param['r_frame'],
                    self.param['r_frame']
                ],
                    strides=1,
                    padding='SAME'), tf.bool)
            kp = tf.math.logical_and(tf.math.equal(det, local_max), non_zeros)

            gx_gauss = self._gaussian(gx, desc_gauss)
            gy_gauss = self._gaussian(gy, desc_gauss)
            gz_gauss = self._gaussian(gz, desc_gauss)

            g_gauss = tf.stack([gx_gauss, gy_gauss, gz_gauss], axis=4)

            gxx = tf.math.square(gx_gauss)
            gxy = tf.math.multiply(gx_gauss, gy_gauss)
            gxz = tf.math.multiply(gx_gauss, gz_gauss)
            gyx = tf.math.multiply(gy_gauss, gx_gauss)
            gyy = tf.math.square(gy_gauss)
            gyz = tf.math.multiply(gy_gauss, gz_gauss)
            gzx = tf.math.multiply(gz_gauss, gx_gauss)
            gzy = tf.math.multiply(gz_gauss, gy_gauss)
            gzz = tf.math.square(gz_gauss)

            s_row1 = tf.squeeze(tf.stack([gxx, gxy, gxz], axis=4), axis=5)
            s_row2 = tf.squeeze(tf.stack([gyx, gyy, gyz], axis=4), axis=5)
            s_row3 = tf.squeeze(tf.stack([gzx, gzy, gzz], axis=4), axis=5)
            s_omega = tf.stack([s_row1, s_row2, s_row3], axis=4)

            init = tf.global_variables_initializer()
            self.session.run(init)

            with self.session.as_default():
                while True:
                    pc_in_numpy, frame = self.esdf_queue.get(True)
                    start_timer = timer()
                    print('processed esdf point cloud size')
                    print(np.multiply.accumulate(pc_in_numpy.shape))
                    det_np, kp_np, g_gauss_np, s_omega_np = self.session.run(
                        [det, kp, g_gauss, s_omega],
                        feed_dict={pc_in: pc_in_numpy})
                    end_timer = timer()
                    print(timedelta(seconds=(end_timer - start_timer)))

                    kp_ids = np.where(kp_np[0, :, :, :, 0] == 1)
                    s_omega_rdc = s_omega_np[0, :, :, :, :, :]
                    g_gauss_rdc = g_gauss_np[0, :, :, :, :, 0]
                    sdf_rdc = pc_in_numpy[0, :, :, :, :]
                    lrf, eigen_val = self._compute_lrf(s_omega_rdc,
                                                       g_gauss_rdc, kp_ids)
                    if lrf is not None:
                        kp_dup, descriptors = self._compute_descriptor(
                            lrf, eigen_val, kp_ids, g_gauss_rdc, sdf_rdc)
                        if len(self.submap_dict) > 0:
                            for frame in self.submap_dict:
                                result=self.matcher.compute_registration(
                                    kp_dup, self.submap_dict[frame][0],
                                    descriptors, self.submap_dict[frame][1])
                                print(result.fitness)
                        self.submap_dict[frame] = (kp_dup, descriptors)

                    end_timer = timer()
                    print(timedelta(seconds=(end_timer - start_timer)))

                    # rviz visualization
                    self._publish_gradient_arrow(g_gauss_rdc, kp_ids)
                    # self._publish_lrf_arrow(lrf, kp_ids)
                    self._publish_pointcloud(pc_in_numpy[0, :, :, :, :],
                                             self.dist_pc_pub, 'dist')
                    self._publish_pointcloud(det_np[0, :, :, :, :],
                                             self.det_pc_pub, 'det')
                    self._publish_pointcloud(kp_np[0, :, :, :, :],
                                             self.kp_pc_pub, 'kp')

                    # det_np = np.array(det)
                    # kp_np = np.array(kp)

    def __compute_gk(self, g, kp_id, shape=None):
        i0 = kp_id[0] - self.r_frame
        i1 = kp_id[0] + self.r_frame
        j0 = kp_id[1] - self.r_frame
        j1 = kp_id[1] + self.r_frame
        k0 = kp_id[2] - self.r_frame
        k1 = kp_id[2] + self.r_frame
        if shape is not None:
            if i0 < 0 or i1 > shape[0] or j0 < 0 or j1 > shape[
                    1] or k0 < 0 or k1 > shape[2]:
                return None
        return g[i0:i1, j0:j1, k0:k1]

    def _compute_lrf(self, s_omega, g, kp_ids):  # np.array [i,j,k,c]
        if len(kp_ids[0]) == 0:
            return None, None
        shape = g.shape
        kp_a = np.zeros(shape=(kp_ids[0].shape[0], 6, 3))
        eigen_value = []
        for i in range(len(kp_ids[0])):
            kp_id = [kp_ids[0][i], kp_ids[1][i], kp_ids[2][i]]
            e_val, v = np.linalg.eig(s_omega[kp_id[0], kp_id[1], kp_id[2]])
            a = []
            for j in range(3):
                # Equation 8
                gk = self.__compute_gk(g, kp_id, shape)
                if gk is None:
                    continue
                # TODO(mikexyl): check dimension
                gkvi = np.tensordot(gk, v[j], axes=(3, 0))
                s = np.sum(gkvi)
                s = s / np.sum(np.abs(gkvi))

                # Equation 7
                k_axis = self.param['k_axis']
                if s >= k_axis:
                    kp_a[i, j, :] = v[j]
                    kp_a[i, j + 3, :] = v[j]
                elif -k_axis <= s <= k_axis:
                    kp_a[i, j, :] = v[j]
                    kp_a[i, j + 3, :] = -v[j]
                else:
                    kp_a[i, j, :] = -v[j]
                    kp_a[i, j + 3, :] = -v[j]
            eigen_value.append(e_val)
        return kp_a, eigen_value

    def __compute_azi_pol_mag(self, p):
        mag = np.linalg.norm(p, axis=3)
        azi = np.arctan2(p[:, :, :, 1], p[:, :, :, 0])
        pol = np.arctan2(p[:, :, :, 2],
                         np.linalg.norm([p[:, :, :, 0], p[:, :, :, 1]]))
        return azi, pol, mag

    def __compute_bin_id(self, azi, pol):
        azi_id, azi_mod = divmod(azi, self.ang_res)
        pol_id, pol_mod = divmod(pol, self.ang_res)
        return azi_id, azi_mod, pol_id, pol_mod

    def __desc_soft_binning(self, azi_id, azi_mod, pol_id, pol_mod, mag):
        # TODO(mikexyl): this can be more vectorized
        n_dim = [
            azi_id.shape[0] * azi_id.shape[1] * azi_id.shape[2], self.n_dims
        ]

        new_descriptor = np.zeros(n_dim)
        # bilinear interpolation
        desc_id = np.zeros([n_dim[0], 2])
        desc_id[:, 1] = np.reshape(azi_id * self.n_divs + pol_id, (n_dim[0]))
        desc_id[:, 0] = range(n_dim[0])
        desc_id = desc_id.astype(np.int).T.tolist()
        new_descriptor[desc_id] = new_descriptor[desc_id] + (
            mag * (self.ang_res - azi_mod) / self.ang_res *
            (self.ang_res - pol_mod) / self.ang_res).reshape(n_dim[0])

        desc_id = np.zeros([n_dim[0], 2])
        desc_id[:, 1] = ((azi_id + 1) * self.n_divs + pol_id).reshape(n_dim[0])
        desc_id[:, 0] = range(n_dim[0])
        desc_id = desc_id.astype(np.int).T.tolist()
        new_descriptor[desc_id] = new_descriptor[desc_id] + (
            mag * azi_mod / self.ang_res *
            (self.ang_res - pol_mod) / self.ang_res).reshape(n_dim[0])

        desc_id = np.zeros([n_dim[0], 2])
        desc_id[:,
                1] = ((azi_id * self.n_divs + (pol_id + 1))).reshape(n_dim[0])
        desc_id[:, 0] = range(n_dim[0])
        desc_id = desc_id.astype(np.int).T.tolist()
        new_descriptor[desc_id] = new_descriptor[desc_id] + (
            mag * (self.ang_res - azi_mod) / self.ang_res * pol_mod /
            self.ang_res).reshape(n_dim[0])

        desc_id = np.zeros([n_dim[0], 2])
        desc_id[:, 1] = ((azi_id + 1) * self.n_divs + (pol_id + 1)).reshape(
            (n_dim[0]))
        desc_id[:, 0] = range(n_dim[0])
        desc_id = desc_id.astype(np.int).T.tolist()
        new_descriptor[desc_id] = new_descriptor[desc_id] + (
            mag * azi_mod / self.ang_res * pol_mod / self.ang_res).reshape(
                n_dim[0])
        return np.sum(new_descriptor, axis=0)

    def __compute_solid_angle(self, azi_id):
        azi = azi_id * self.ang_res
        return self.ang_res * (np.cos(azi) - np.cos(azi + self.ang_res))

    def __normalize_desc(self, descriptor, n_voxels):
        descriptor = descriptor / (n_voxels * self.solid_angle)
        return descriptor

    def _compute_descriptor(self, lrf, e_val, kp_ids, g, sdf):
        descriptors = None
        kp_dup = None
        for i in range(len(kp_ids[0])):
            kp_id = [kp_ids[0][i], kp_ids[1][i], kp_ids[2][i]]
            # Equation 9
            gk_s = self.__compute_gk(g, kp_id)
            a_ls = self.__compute_axes(lrf[i])
            for a in a_ls:
                # TODO(mikexyl): verify dimension
                R_f_s = np.linalg.inv(np.transpose(a))
                # TODO(mikexyl): there should a np function for this
                gk_f = np.matmul(R_f_s, gk_s[:, :, :, :, np.newaxis])
                azi, pol, mag = self.__compute_azi_pol_mag(gk_f)
                azi_id, azi_mod, pol_id, pol_mod = self.__compute_bin_id(
                    azi, pol)

                descriptor = self.__desc_soft_binning(azi_id, azi_mod, pol_id,
                                                      pol_mod, mag)
                # Normalize descriptors
                # TODO(mikexyl): count n of valid voxels
                descriptor = self.__normalize_desc(descriptor, self.r_frame**2)

                # Equation 10
                # TODO(mikexyl): for now, just give up points having
                #  invalid sdf in its desc support
                b_dist = sdf[kp_id[0], kp_id[1],
                             kp_id[2]] / self.grad_gauss_sum
                # Equation 12
                b_class = np.sum(e_val[i] > 0)
                alp_dist = 1e-7
                alp_class = 1e-5
                d_dist = alp_dist * b_dist
                d_class = alp_class * b_class
                descriptor = np.concatenate((descriptor, [d_dist, d_class]))

                if kp_dup is None and descriptors is None:
                    kp_dup = np.array(kp_id)
                    descriptors = descriptor
                else:
                    kp_dup = np.vstack([kp_dup, np.array(kp_id)])
                    descriptors = np.vstack([descriptors, descriptor])
        return kp_dup, descriptors.astype(np.float)

    def _publish_pointcloud(self, pc_np, publisher, data_type):
        assert len(pc_np.shape) == 4
        points = []
        pt_ids = np.where(pc_np != 0)
        max_value = pc_np.max() - pc_np.min()
        min_value = pc_np.min()
        if not pt_ids:
            return
        for i in range(len(pt_ids[0])):
            x = pt_ids[0][i]
            y = pt_ids[1][i]
            z = pt_ids[2][i]
            if data_type == 'dist':
                if pc_np[x, y, z][0] > 0:
                    r = 0
                    g = int(
                        np.log((
                            (pc_np[x, y, z][0] - min_value) / max_value) + 1) *
                        255)
                    b = 0
                else:
                    r = 0
                    g = 0
                    b = int(
                        np.log((
                            (pc_np[x, y, z][0] - min_value) / max_value) + 1) *
                        255)
            elif data_type == 'det':
                r = 0
                g = 0
                b = int(
                    np.log(((pc_np[x, y, z][0] - min_value) / max_value) + 1) *
                    255)
            elif data_type == 'kp':
                r = 255
                g = 0
                b = 0
            a = 255
            rgba = struct.unpack("I", struct.pack('BBBB', b, g, r, a))[0]
            pt = [
                float(x * self.vis_scale),
                float(y * self.vis_scale),
                float(z * self.vis_scale), rgba
            ]
            points.append(pt)
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgba', 12, PointField.UINT32, 1)
        ]
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'world'
        pc2 = point_cloud2.create_cloud(header, fields, points)
        publisher.publish(pc2)
        # pylint: disable=line-too-long
        print('published point cloud %s size: ' % data_type,
              pc2.height * pc2.width)

    def __new_marker(self, ns, id, frame_id, type, action):
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = frame_id
        marker.ns = ns
        marker.id = id
        marker.type = type
        # TODO(mikexyl): not sure
        marker.action = action
        marker.scale.x = 0.01
        marker.scale.y = 0.02
        marker.color.r = 0
        marker.color.g = 0
        marker.color.b = 255
        marker.color.a = 255
        return marker

    def _publish_gradient_arrow(self, g, kp_ids):
        if len(kp_ids[0]) == 0:
            return
        reset_marker = self.__new_marker('gradient', 0, 'world', Marker.ARROW,
                                         Marker.DELETEALL)
        reset_marker.action = Marker.DELETEALL
        self.grad_pub.publish([reset_marker])
        grad_markers = MarkerArray()
        for i in range(len(kp_ids[0])):
            kp_id = [kp_ids[0][i], kp_ids[1][i], kp_ids[2][i]]
            grad_marker = self.__new_marker('gradient', i, 'world',
                                            Marker.ARROW, Marker.ADD)
            x = kp_id[0] * self.vis_scale
            y = kp_id[1] * self.vis_scale
            z = kp_id[2] * self.vis_scale
            grad_marker.points.append(Point(x=x, y=y, z=z))
            gp = g[kp_id[0], kp_id[1], kp_id[2]]
            grad_marker.points.append(
                Point(x=x + gp[0], y=y + gp[1], z=z + gp[2]))
            grad_markers.markers.append(grad_marker)
        self.grad_pub.publish(grad_markers)

    def __compute_axes(self, lrf):  # return [v1,v1,v1;v2,v2,v2;v3,v3,v3]
        a = np.zeros((2, 3, 3))
        a_ls = []
        a = [[], [], []]
        for i in range(3):
            a[i].append(lrf[i, :])
            if (a[i][0] != lrf[i + 3, :]).any():
                a[i].append(lrf[i + 3, :])
        for i in range(len(a[0])):
            for j in range(len(a[1])):
                for k in range(len(a[2])):
                    a_ls.append(np.stack((a[0][i], a[1][j], a[2][k])))
        return a_ls

    def extractFreetures(self, pointcloud, frame):
        i_max = np.max(pointcloud['gi'])
        i_min = np.min(pointcloud['gi'])
        j_max = np.max(pointcloud['gj'])
        j_min = np.min(pointcloud['gj'])
        k_max = np.max(pointcloud['gk'])
        k_min = np.min(pointcloud['gk'])
        print('received point cloud size:', len(pointcloud['gi']))
        pc = np.zeros(
            (1, i_max - i_min + 1, j_max - j_min + 1, k_max - k_min + 1, 1),
            dtype=np.float32)
        for i, j, k, dist in zip(pointcloud['gi'], pointcloud['gj'],
                                 pointcloud['gk'], pointcloud['distance']):
            pc[0, i - i_min, j - j_min, k - k_min] = dist
        self.esdf_queue.put((pc, frame))
