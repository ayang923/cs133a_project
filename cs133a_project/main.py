'''hw6p5.py

   This is the skeleton code for HW6 Problem 5.  Please EDIT.

   This uses the inverse kinematics from Problem 4, but adds a more
   complex trajectory.

   Node:        /generator
   Publish:     /joint_states           sensor_msgs/JointState

'''

import rclpy
import numpy as np

from math import pi, sin, cos, acos, atan2, sqrt, fmod, exp

from scipy.linalg import diagsvd

# Grab the utilities
from cs133a_project.nodes      import RobotNode, BallNode
from cs133a_project.TransformHelpers   import *
from cs133a_project.TrajectoryUtils    import *

# Grab the general fkin from HW5 P5.
from cs133a_project.KinematicChain     import KinematicChain
from cs133a_project.joint_info import ATLAS_JOINT_NAMES, ATLAS_L_LEG_JOINT_NAMES, ATLAS_R_LEG_JOINT_NAMES

from visualization_msgs.msg     import Marker
from visualization_msgs.msg     import MarkerArray

#
#   Trajectory Class
#
class Trajectory():
    # Initialization.
    def __init__(self, node):
        self.node = node
        # Set up the kinematic chain object.
        self.node.create_subscription(
            MarkerArray,
            '/visualization_marker_array',
            self.process_ball,
            10  # Queue size for topic 1
        )

        self.l_leg_chain = KinematicChain(self.node, 'pelvis', 'l_foot_paddle', ATLAS_L_LEG_JOINT_NAMES)
        self.r_leg_chain = KinematicChain(self.node, 'pelvis', 'r_foot_paddle', ATLAS_R_LEG_JOINT_NAMES)

        self.q0 = np.zeros((len(ATLAS_L_LEG_JOINT_NAMES), 1))

        self.x0 = self.l_leg_chain.fkin(self.q0)[0]
        self.xf = self.x0 + np.array([0, 0, 0.5]).reshape(3, 1)

        self.q = np.vstack((self.q0, self.q0))
        self.pd = self.x0
        self.Rd = Reye()

        self.p_world = np.array([0, 0, 0.75]).reshape([3, 1])
        self.v_world = np.zeros((3, 1))
        self.R_world = Reye()

        # wrt to world
        self.ball_p = None
        self.ball_v = None
        self.ball_t = None

    def process_ball(self, msg):
        if msg.markers:
            ball = msg.markers[0]
            new_ball_p = p_from_Point(ball.pose.position)
            new_ball_t = ball.header.stamp.sec + ball.header.stamp.nanosec*10**-9
            if self.ball_p is not None:
                self.ball_v = (new_ball_p - self.ball_p) / (new_ball_t - self.ball_t)
            self.ball_p = new_ball_p
            self.ball_t = new_ball_t

    # Declare the joint names.
    def jointnames(self):
        # Return a list of joint names FOR THE EXPECTED URDF!
        return ATLAS_JOINT_NAMES
    
    def ikin_leg(self, qlast, pdlast, Rdlast, kinematic_chain, pd, vd, Rd, wd, dt):
        p, R, Jv, Jw = kinematic_chain.fkin(qlast)
        
        J = np.vstack((Jv, Jw))
        xdot = np.vstack((pd, wd))

        e = np.vstack((ep(pdlast, p), eR(Rdlast, R)))
        J_inv = self.weighted_svd_inverse(J)

        qdot = J_inv @ (xdot + 20*e)
        q = qlast + dt*qdot

        return q, qdot
            
    def weighted_svd_inverse(self, J, gamma=0.1):
        U, S, V = np.linalg.svd(J)

        msk = np.abs(S) >= gamma
        S_inv = np.zeros(len(S))
        S_inv[msk] = 1/S[msk]
        S_inv[~msk] = S[~msk]/gamma**2

        return V.T @ diagsvd(S_inv, *J.T.shape) @ U.T

    # Evaluate at the given time.  This was last called (dt) ago.
    def evaluate(self, t, dt):
        if self.ball_v is not None:
            pd = self.R_world.T @ (self.ball_p - self.p_world)
            vd = self.R_world.T @ (self.ball_v - self.v_world)
        else:
            pd, vd = self.x0, np.zeros((3, 1))
        
        Rd, wd = Reye(), np.zeros((3, 1))

        qlast_l_leg = self.q[:len(ATLAS_L_LEG_JOINT_NAMES), :]

        pdlast = self.pd
        Rdlast = self.Rd

        q_l_leg, qdot_l_leg = self.ikin_leg(qlast_l_leg, pdlast, Rdlast, self.l_leg_chain, pd, vd, Rd, wd, dt)
        q_r_leg, qdot_r_leg = self.ikin_leg(qlast_l_leg, pdlast, Rdlast, self.r_leg_chain, pd, vd, Rd, wd, dt)
        
        self.q = np.vstack((q_l_leg, q_r_leg))
        self.pd = pd
        self.Rd = Rd

        q_dict_l_leg = dict(zip(ATLAS_L_LEG_JOINT_NAMES, q_l_leg.flatten()))
        q_dict_r_leg = dict(zip(ATLAS_R_LEG_JOINT_NAMES, q_r_leg.flatten()))
        q_dict = {**q_dict_l_leg, **q_dict_r_leg}

        qdot_dict_l_leg = dict(zip(ATLAS_L_LEG_JOINT_NAMES, qdot_l_leg.flatten()))
        qdot_dict_r_leg = dict(zip(ATLAS_R_LEG_JOINT_NAMES, qdot_r_leg.flatten()))
        qdot_dict = {**qdot_dict_l_leg, **qdot_dict_r_leg}

        q = np.array([q_dict[joint_name] if joint_name in q_dict else 0 for joint_name in self.jointnames()])
        qdot = np.array([qdot_dict[joint_name] if joint_name in q_dict else 0 for joint_name in self.jointnames()])
        return (self.p_world, self.R_world, q.flatten().tolist(), qdot.flatten().tolist())

#
#  Main Code
#
def main(args=None):
    rclpy.init(args=args)
    ball_node = BallNode('balldemo', 100)
    robot_node = RobotNode('generator', 100, Trajectory)

    while rclpy.ok():
        rclpy.spin_once(robot_node)  # Spinning node_a
        rclpy.spin_once(ball_node)  # Spinning node_b

    # Shutdown the node and ROS.
    ball_node.shutdown()
    robot_node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main(
)
