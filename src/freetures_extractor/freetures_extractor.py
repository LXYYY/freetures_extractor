import os
import threading
import Queue
import numpy as np
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

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        self.process_thread = threading.Thread(target=self._process)
        self.process_thread.start()

    def _process(self):
        with self.session.graph.as_default():

            sobel = tf.constant([-1, 0, 1], dtype=tf.float32)
            x_sobel = sobel[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
            y_sobel = sobel[tf.newaxis, :, tf.newaxis, tf.newaxis, tf.newaxis]
            z_sobel = sobel[tf.newaxis, tf.newaxis, :, tf.newaxis, tf.newaxis]
            x = np.arange(-2, 3, 1)
            sigma = 2
            gaussian = tf.convert_to_tensor(np.exp(-(x)**2 / (2 * sigma**2)),
                                            dtype=tf.float32)
            x_gaussian = gaussian[:, tf.newaxis, tf.newaxis, tf.newaxis,
                                  tf.newaxis]
            y_gaussian = gaussian[tf.newaxis, :, tf.newaxis, tf.newaxis,
                                  tf.newaxis]
            z_gaussian = gaussian[tf.newaxis, tf.newaxis, :, tf.newaxis,
                                  tf.newaxis]

            pc_in = tf.placeholder(dtype=tf.float32)
            # pc_in = tf.Variable([1, dim_x, dim_y, dim_z, 1], trainable=False,
            #                     dtype=tf.float32, name='input_esdf')

            pc_gauss_x = tf.nn.conv3d(pc_in,
                                      x_gaussian,
                                      strides=[1, 1, 1, 1, 1],
                                      padding="SAME")
            pc_gauss_y = tf.nn.conv3d(pc_in,
                                      y_gaussian,
                                      strides=[1, 1, 1, 1, 1],
                                      padding="SAME")
            pc_gauss_z = tf.nn.conv3d(pc_in,
                                      z_gaussian,
                                      strides=[1, 1, 1, 1, 1],
                                      padding="SAME")
            pc_gauss_xyz = tf.math.add(tf.math.add(pc_gauss_x, pc_gauss_y),
                                       pc_gauss_z)

            hx = tf.nn.conv3d(pc_gauss_xyz,
                              x_sobel,
                              strides=[1, 1, 1, 1, 1],
                              padding="SAME")
            hxx = tf.nn.conv3d(hx,
                               x_sobel,
                               strides=[1, 1, 1, 1, 1],
                               padding="SAME")
            hxy = tf.nn.conv3d(hx,
                               y_sobel,
                               strides=[1, 1, 1, 1, 1],
                               padding="SAME")
            hxz = tf.nn.conv3d(hx,
                               z_sobel,
                               strides=[1, 1, 1, 1, 1],
                               padding="SAME")
            hy = tf.nn.conv3d(pc_gauss_xyz,
                              y_sobel,
                              strides=[1, 1, 1, 1, 1],
                              padding="SAME")
            hyx = tf.nn.conv3d(hy,
                               x_sobel,
                               strides=[1, 1, 1, 1, 1],
                               padding="SAME")
            hyy = tf.nn.conv3d(hy,
                               y_sobel,
                               strides=[1, 1, 1, 1, 1],
                               padding="SAME")
            hyz = tf.nn.conv3d(hy,
                               z_sobel,
                               strides=[1, 1, 1, 1, 1],
                               padding="SAME")
            hz = tf.nn.conv3d(pc_gauss_xyz,
                              z_sobel,
                              strides=[1, 1, 1, 1, 1],
                              padding="SAME")
            hzx = tf.nn.conv3d(hz,
                               x_sobel,
                               strides=[1, 1, 1, 1, 1],
                               padding="SAME")
            hzy = tf.nn.conv3d(hz,
                               y_sobel,
                               strides=[1, 1, 1, 1, 1],
                               padding="SAME")
            hzz = tf.nn.conv3d(hz,
                               z_sobel,
                               strides=[1, 1, 1, 1, 1],
                               padding="SAME")

            # TODO(mikexyl): check if order is correct
            row1 = tf.squeeze(tf.stack([hxx, hxy, hxz], axis=4), axis=5)
            row2 = tf.squeeze(tf.stack([hyx, hyy, hyz], axis=4), axis=5)
            row3 = tf.squeeze(tf.stack([hzx, hzy, hzz], axis=4), axis=5)
            hessian = tf.stack([row1, row2, row3], axis=4)
            det = tf.matrix_determinant(hessian)[:, :, :, :, tf.newaxis]
            local_max = tf.nn.max_pool3d(det, [5, 5, 5],
                                         strides=[1, 1, 1, 1, 1],
                                         padding='SAME')
            non_zeros = tf.math.not_equal(local_max, tf.constant(0.0))
            kp_id = tf.math.logical_and(tf.math.equal(det, local_max),
                                        non_zeros)
            n_kp = tf.reduce_sum(tf.cast(kp_id, dtype=tf.float32))

            init = tf.global_variables_initializer()
            self.session.run(init)

            with self.session.as_default():
                while True:
                    print 'processed esdf point cloud size'
                    pc_in_numpy = self.esdf_queue.get(True)
                    print np.multiply.accumulate(pc_in_numpy.shape)
                    print self.session.run([n_kp],
                                           feed_dict={pc_in: pc_in_numpy})

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
