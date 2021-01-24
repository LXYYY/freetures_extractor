#!/usr/bin/env python

from freetures_extractor import FreeturesExtractor

import ros_numpy
import rospy
from sensor_msgs.msg import PointCloud2


class EsdfPointcloudSub(object):
    def __init__(self):
        self.parameters = rospy.get_param("~")
        self.extractor = FreeturesExtractor(self.parameters)
        self.pc_sub = rospy.Subscriber(
            "esdf_pointcloud2", PointCloud2, callback=self.callback)

    def callback(self, pointcloud_msg):
        if len(pointcloud_msg.data)==0:
            return
        pc_numpy = ros_numpy.numpify(pointcloud_msg)
        self.extractor.extractFreetures(pc_numpy)


if __name__ == '__main__':
    rospy.init_node('freetures_extractor', anonymous=True)
    esdf_sub = EsdfPointcloudSub()
    rospy.spin()
