import os
import threading
import struct
import Queue
import rospy
import numpy as np
from std_msgs.msg import Header
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
import tensorflow as tf


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
        self.dist_pc_pub = rospy.Publisher("distance", PointCloud2)
        self.det_pc_pub = rospy.Publisher("det", PointCloud2)
        self.kp_pc_pub = rospy.Publisher("keypoints", PointCloud2)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        self.process_thread = threading.Thread(target=self._process)
        self.process_thread.start()

    def _get_gaussian(self, sigma):
        x = np.arange(-2, 3, 1)
        gaussian = tf.convert_to_tensor(np.exp(-(x)**2 / (2 * sigma**2)),
                                        dtype=tf.float32)
        x_gaussian = gaussian[:, tf.newaxis, tf.newaxis, tf.newaxis,
                              tf.newaxis]
        y_gaussian = gaussian[tf.newaxis, :, tf.newaxis, tf.newaxis,
                              tf.newaxis]
        z_gaussian = gaussian[tf.newaxis, tf.newaxis, :, tf.newaxis,
                              tf.newaxis]
        return x_gaussian, y_gaussian, z_gaussian

    def _gaussian(self, in_tensor, x_gauss, y_gauss, z_gauss):
        pc_gauss_x = tf.nn.conv3d(in_tensor,
                                  x_gauss,
                                  strides=[1, 1, 1, 1, 1],
                                  padding="SAME")
        pc_gauss_y = tf.nn.conv3d(in_tensor,
                                  y_gauss,
                                  strides=[1, 1, 1, 1, 1],
                                  padding="SAME")
        pc_gauss_z = tf.nn.conv3d(in_tensor,
                                  z_gauss,
                                  strides=[1, 1, 1, 1, 1],
                                  padding="SAME")
        pc_gauss_xyz = tf.math.add(tf.math.add(pc_gauss_x, pc_gauss_y),
                                   pc_gauss_z)
        return pc_gauss_xyz

    def _process(self):
        with self.session.graph.as_default():

            # prepare sobel and gaussian kernel
            sobel = tf.constant([-1, 0, 1], dtype=tf.float32)
            x_sobel = sobel[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
            y_sobel = sobel[tf.newaxis, :, tf.newaxis, tf.newaxis, tf.newaxis]
            z_sobel = sobel[tf.newaxis, tf.newaxis, :, tf.newaxis, tf.newaxis]

            x_grad_gauss, y_grad_gauss, z_grad_gauss = self._get_gaussian(
                2)  # sigma_grad
            x_desc_gauss, y_desc_gauss, z_desc_gauss = self._get_gaussian(
                self.param['r_frame'])  # sigma_grad

            # compute det of hessian
            pc_in = tf.placeholder(dtype=tf.float32)
            pc_gauss_xyz = self._gaussian(pc_in, x_grad_gauss, y_grad_gauss,
                                          z_grad_gauss)

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

            gx_gauss = self._gaussian(gx, x_desc_gauss, y_desc_gauss,
                                      z_desc_gauss)
            gy_gauss = self._gaussian(gy, y_desc_gauss, y_desc_gauss,
                                      z_desc_gauss)
            gz_gauss = self._gaussian(gz, z_desc_gauss, y_desc_gauss,
                                      z_desc_gauss)

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
                    pc_in_numpy = self.esdf_queue.get(True)
                    print 'processed esdf point cloud size'
                    print np.multiply.accumulate(pc_in_numpy.shape)
                    det_np, kp_np, g_gauss_np, s_omega_np = self.session.run(
                        [det, kp, g_gauss, s_omega],
                        feed_dict={pc_in: pc_in_numpy})
                    self._compute_lrf(s_omega_np[0, :, :, :, :, :],
                                      g_gauss_np[0, :, :, :, :,
                                                 0], kp_np[0, :, :, :, 0])
                    self._publish_pointcloud(pc_in_numpy[0, :, :, :, :],
                                             self.dist_pc_pub, 'dist')
                    self._publish_pointcloud(det_np[0, :, :, :, :],
                                             self.det_pc_pub, 'det')
                    self._publish_pointcloud(kp_np[0, :, :, :, :],
                                             self.kp_pc_pub, 'kp')

                    # det_np = np.array(det)
                    # kp_np = np.array(kp)

    def _compute_lrf(self, s_omega, g, kp):  # np.array [i,j,k,c]
        kp_ids = np.where(kp == 1)
        shape = kp.shape
        kp_a = np.empty(shape=(shape[0], shape[1], shape[2]) + (0, )).tolist()
        if not kp_ids[0]:
            return
        for i in range(len(kp_ids[0])):
            kp_id = [kp_ids[0][i], kp_ids[1][i], kp_ids[2][i]]
            _, v = np.linalg.eig(s_omega[kp_id[0], kp_id[1], kp_id[2]])
            a = []
            for j in range(3):
                # Equation 8
                i0 = kp_id[0] - self.r_frame
                i1 = kp_id[0] + self.r_frame
                j0 = kp_id[1] - self.r_frame
                j1 = kp_id[1] + self.r_frame
                k0 = kp_id[2] - self.r_frame
                k1 = kp_id[2] + self.r_frame
                if i0 < 0 or i1 > shape[0] or j0 < 0 or j1 > shape[
                        1] or k0 < 0 or k1 > shape[2]:
                    continue
                gk = g[i0:i1, j0:j1, k0:k1]
                # TODO(mikexyl): check dimension
                gkvi = np.tensordot(gk, v[j], axes=(3, 0))
                s = np.sum(gkvi)
                s = s / np.sum(np.abs(gkvi))

                # Equation 7
                k_axis = self.param['k_axis']
                if s >= k_axis:
                    a.append(v[j])
                    a.append(0)
                elif -k_axis <= s <= k_axis:
                    a.append(v[j])
                    a.append(-v[j])
                else:
                    a.append(-v[j])
                    a.append(0)
            kp_a[kp_id[0]][kp_id[1]][kp_id[2]] = a
        return kp_a

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
            pt = [float(x * 0.1), float(y * 0.1), float(z * 0.1), rgba]
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
        print 'published point cloud %s size: ' % data_type, pc2.height * pc2.width

    def extractFreetures(self, pointcloud):
        i_max = np.max(pointcloud['gi'])
        i_min = np.min(pointcloud['gi'])
        j_max = np.max(pointcloud['gj'])
        j_min = np.min(pointcloud['gj'])
        k_max = np.max(pointcloud['gk'])
        k_min = np.min(pointcloud['gk'])
        pc = np.zeros(
            (1, i_max - i_min + 1, j_max - j_min + 1, k_max - k_min + 1, 1),
            dtype=np.float32)
        for i, j, k, dist in zip(pointcloud['gi'], pointcloud['gj'],
                                 pointcloud['gk'], pointcloud['distance']):
            pc[0, i - i_min, j - j_min, k - k_min] = dist
        self.esdf_queue.put(pc)
