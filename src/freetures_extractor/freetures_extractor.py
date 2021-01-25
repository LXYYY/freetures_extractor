import os
import threading
import Queue
import numpy as np
import tensorflow as tf
import pcl


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
            non_zeros = tf.cast(tf.math.not_equal(local_max, tf.constant(0.0)),
                                tf.float32)
            non_zeros = tf.cast(
                -tf.nn.max_pool3d(-non_zeros, [
                    self.param['r_frame'], self.param['r_frame'],
                    self.param['r_frame']
                ],
                                  strides=1,
                                  padding='SAME'), tf.bool)
            kp = tf.math.logical_and(tf.math.equal(det, local_max), non_zeros)
            n_kp = tf.reduce_sum(tf.cast(kp, dtype=tf.float32))

            gxx = self._gaussian(tf.math.square(gx), x_desc_gauss,
                                 y_desc_gauss, z_desc_gauss)
            gxy = self._gaussian(tf.math.multiply(gx, gy), x_desc_gauss,
                                 y_desc_gauss, z_desc_gauss)
            gxz = self._gaussian(tf.math.multiply(gx, gz), x_desc_gauss,
                                 y_desc_gauss, z_desc_gauss)
            gyx = self._gaussian(tf.math.multiply(gy, gx), x_desc_gauss,
                                 y_desc_gauss, z_desc_gauss)
            gyy = self._gaussian(tf.math.square(gy), x_desc_gauss,
                                 y_desc_gauss, z_desc_gauss)
            gyz = self._gaussian(tf.math.multiply(gy, gz), x_desc_gauss,
                                 y_desc_gauss, z_desc_gauss)
            gzx = self._gaussian(tf.math.multiply(gz, gx), x_desc_gauss,
                                 y_desc_gauss, z_desc_gauss)
            gzy = self._gaussian(tf.math.multiply(gz, gy), x_desc_gauss,
                                 y_desc_gauss, z_desc_gauss)
            gzz = self._gaussian(tf.math.square(gz), x_desc_gauss,
                                 y_desc_gauss, z_desc_gauss)
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
                    self.session.run([n_kp, s_omega],
                                     feed_dict={pc_in: pc_in_numpy})
                    print np.array(n_kp)

                    # det_np = np.array(det)
                    # kp_np = np.array(kp)

    def _compute_lrf(self, s_omega, g, kp):  # np.array [i,j,k,c]
        kp_id = np.where(kp == 1)
        for kp_id in kp_id:
            _, kp_v = np.linalg.eig(s_omega[kp_id])
            for i in range(3):
                # Equation 8
                s = np.sum(
                    np.dot(
                        g[kp_id[0] - self.r_frame:kp_id[0] +
                          self.r_frame, :kp_id[1] - self.r_frame:kp_id[1] +
                          self.r_frame,
                          kp_id[2] - self.r_frame:kp_id[2] + self.r_frame],
                        kp_v[i]))
                print s

    def _visualize_pcl(self, in_np, det_np, kp_np):
        in_pcl = pcl.PointCloud(in_np, dtype=np.float32)
        det_pcl = pcl.PointCloud(det_np, dtype=np.float32)
        kp_pcl = pcl.PointCloud(kp_np, dtype=np.float32)
        print in_pcl
        print det_pcl
        print kp_pcl

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
