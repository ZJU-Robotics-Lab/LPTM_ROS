import sys
import rospy
from lptm_ros.srv import ComputePtWeights, ComputePtWeightsResponse
from cv_bridge import CvBridge
import cv2

cvBridge1 = CvBridge()

def handle_compute_weight(req):
    global done_all, template_msg, source_msg, x_coords, y_coords, particle_number, header
    print("handle_compute_weight", req.particle_number)
    
    template_msg = req.TemplateImage
    source_msg = req.SourceImage
    x_coords = req.x_position_of_particle
    y_coords = req.y_position_of_particle
    particle_number = req.particle_number
    source_msg = cvBridge1.imgmsg_to_cv2(source_msg)
    template_msg = cvBridge1.imgmsg_to_cv2(template_msg)
    cv2.imshow("tempppppp", template_msg)
    cv2.waitKey(1000)
    cv2.imshow("srvccccccccc", source_msg)
    cv2.waitKey(1000)
    # header = req.header
    done_all = 1

def add_server():
    s = rospy.Service('compute_weight_server', ComputePtWeights, handle_compute_weight)
    print("Read to provide service")
    rospy.spin()


if __name__ == "__main__":
    rospy.init_node('Detecter', anonymous=True)
    add_server()