from tensorflow.python.framework import dtypes
from tensorflow.python.ops.gen_array_ops import shape
import numpy as np
from scipy.signal import convolve
import os
import threading
import Queue
import tensorflow as tf
from tensorboard.plugins import projector
from tensorflow.python.ops.summary_ops_v2 import graph


class FreeturesExtractor(object):
    def __init__(self, param):
        self.esdf_points = None
        self.esdf_queue = Queue.Queue()
        self.log_dir = param['log_dir']
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.pc_input_metadata = os.path.join(
            self.log_dir, "pc_input_metadata.tsv")
        with open(self.pc_input_metadata, "w") as f:
            for subwords in ['x', 'y', 'z', 'dist']:
                f.write("{}\n".format(subwords))

        config = tf.ConfigProto()
        self.session = tf.Session(config=config)
        self.process_thread = threading.Thread(target=self._process)
        self.process_thread.start()

    def _process(self):
        with self.session.graph.as_default():

            sobel = tf.constant([-1, 0, 1], dtype=tf.float32)
            self.x_sobel = sobel[:, tf.newaxis,
                                 tf.newaxis, tf.newaxis, tf.newaxis]
            self.y_sobel = sobel[tf.newaxis, :,
                                 tf.newaxis, tf.newaxis, tf.newaxis]
            self.z_sobel = sobel[tf.newaxis,
                                 tf.newaxis, :, tf.newaxis, tf.newaxis]
            x = np.arange(-2, 3, 1)
            sigma = 2
            gaussian = tf.convert_to_tensor(
                np.exp(-(x)**2/(2*sigma**2)), dtype=tf.float32)
            self.x_gaussian = gaussian[:, tf.newaxis,
                                       tf.newaxis, tf.newaxis, tf.newaxis]
            self.y_gaussian = gaussian[tf.newaxis, :,
                                       tf.newaxis, tf.newaxis, tf.newaxis]
            self.z_gaussian = gaussian[tf.newaxis,
                                       tf.newaxis, :, tf.newaxis, tf.newaxis]

            # print(self.x_sobel.eval())
            # print(self.y_sobel.eval())
            # print(self.z_sobel.eval())
            self.xx_sobel = tf.nn.conv3d(self.x_sobel, self.x_sobel, strides=[
                1, 1, 1, 1, 1], padding='SAME')
            # print('------xx------')
            # print(self.xx_sobel.eval())
            self.xy_sobel = tf.nn.conv3d(self.x_sobel, self.y_sobel, strides=[
                1, 1, 1, 1, 1], padding='SAME')
            # print('------xy------')
            # print(self.xy_sobel.eval())
            self.xz_sobel = tf.nn.conv3d(self.x_sobel, self.z_sobel, strides=[
                1, 1, 1, 1, 1], padding='SAME')
            # print(self.xz_sobel.eval())
            self.yx_sobel = tf.nn.conv3d(self.y_sobel, self.x_sobel, strides=[
                1, 1, 1, 1, 1], padding='SAME')
            # print(self.yx_sobel.eval())
            self.yy_sobel = tf.nn.conv3d(self.y_sobel, self.y_sobel, strides=[
                1, 1, 1, 1, 1], padding='SAME')
            # print(self.yy_sobel.eval())
            self.yz_sobel = tf.nn.conv3d(self.y_sobel, self.z_sobel, strides=[
                1, 1, 1, 1, 1], padding='SAME')
            # print(self.yz_sobel.eval())
            self.zx_sobel = tf.nn.conv3d(self.z_sobel, self.x_sobel, strides=[
                1, 1, 1, 1, 1], padding='SAME')
            self.zy_sobel = tf.nn.conv3d(self.z_sobel, self.y_sobel, strides=[
                1, 1, 1, 1, 1], padding='SAME')
            self.zz_sobel = tf.nn.conv3d(self.z_sobel, self.z_sobel, strides=[
                1, 1, 1, 1, 1], padding='SAME')

            file_writer = tf.summary.FileWriter(
                self.log_dir, self.session.graph)

            with self.session.as_default():
                while(1):
                    pc_var = tf.Variable(
                        self.esdf_queue.get(True), trainable=False, dtype=tf.float32, name='input_esdf')

                    pc_gauss_x = tf.nn.conv3d(pc_var, self.x_gaussian, strides=[
                        1, 1, 1, 1, 1], padding="SAME")
                    pc_gauss_y = tf.nn.conv3d(pc_var, self.y_gaussian, strides=[
                        1, 1, 1, 1, 1], padding="SAME")
                    pc_gauss_z = tf.nn.conv3d(pc_var, self.z_gaussian, strides=[
                        1, 1, 1, 1, 1], padding="SAME")
                    pc_gauss_xyz = tf.math.add(tf.math.add(
                        pc_gauss_x, pc_gauss_y), pc_gauss_z)

                    hxx = tf.nn.conv3d(pc_gauss_xyz, self.xx_sobel, strides=[
                        1, 1, 1, 1, 1], padding="SAME")
                    hxy = tf.nn.conv3d(pc_gauss_xyz, self.xy_sobel, strides=[
                        1, 1, 1, 1, 1], padding="SAME")
                    hxz = tf.nn.conv3d(pc_gauss_xyz, self.xz_sobel, strides=[
                        1, 1, 1, 1, 1], padding="SAME")
                    hyx = tf.nn.conv3d(pc_gauss_xyz, self.yx_sobel, strides=[
                        1, 1, 1, 1, 1], padding="SAME")
                    hyy = tf.nn.conv3d(pc_gauss_xyz, self.yy_sobel, strides=[
                        1, 1, 1, 1, 1], padding="SAME")
                    hyz = tf.nn.conv3d(pc_gauss_xyz, self.yz_sobel, strides=[
                        1, 1, 1, 1, 1], padding="SAME")
                    hzx = tf.nn.conv3d(pc_gauss_xyz, self.zx_sobel, strides=[
                        1, 1, 1, 1, 1], padding="SAME")
                    hzy = tf.nn.conv3d(pc_gauss_xyz, self.zy_sobel, strides=[
                        1, 1, 1, 1, 1], padding="SAME")
                    hzz = tf.nn.conv3d(pc_gauss_xyz, self.zz_sobel, strides=[
                        1, 1, 1, 1, 1], padding="SAME")

                    # TODO(mikexyl): check if order is correct
                    row1 = tf.squees(
                        tf.stack([hxx, hxy, hxz], axis=3), axis=[4, 5])
                    row2 = tf.squeeze(
                        tf.stack([hyx, hyy, hyz], axis=3), axis=[4, 5])
                    row3 = tf.squeeze(
                        tf.stack([hzx, hzy, hzz], axis=3), axis=[4, 5])
                    hessian = tf.stack([row1, row2, row3], axis=3)
                    det = tf.linalg.det(hessian)
                    # max, voxel_id=tf.nn.max_pool_with_argmax(det, [10,10,10])

                    print('processed esdf point cloud size')
                    print(pc_var.get_shape())
                    print(det)

                    init = tf.global_variables_initializer()
                    self.session.run(init)

                    # visulization
                    checkpoint = tf.train.Checkpoint(embedding=pc_var)
                    checkpoint.save(os.path.join(
                        self.log_dir, "embedding.ckpt"))

                    proj_config = projector.ProjectorConfig()
                    embedding = proj_config.embeddings.add()
                    embedding.tensor_name = pc_var.name
                    embedding.metadata_path = self.pc_input_metadata
                    projector.visualize_embeddings(file_writer, proj_config)

    def extractFreetures(self, pointcloud):
        i_max = np.max(pointcloud['gi'])
        i_min = np.min(pointcloud['gi'])
        j_max = np.max(pointcloud['gj'])
        j_min = np.min(pointcloud['gj'])
        k_max = np.max(pointcloud['gk'])
        k_min = np.min(pointcloud['gk'])
        pc = np.zeros((1,i_max-i_min+1, j_max-j_min+1, k_max-k_min+1,  1),
                      dtype=np.float32)
        for i, j, k, dist in zip(pointcloud['gi'], pointcloud['gj'], pointcloud['gk'],  pointcloud['distance']):
            pc[0, i-i_min, j-j_min, k-k_min] = dist
        self.esdf_queue.put(pc)
