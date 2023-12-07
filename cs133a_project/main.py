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
from cs133a_project.nodes      import ProjectNode, RobotNode, BallNode, CollisionInfo
from cs133a_project.TransformHelpers   import *
from cs133a_project.TrajectoryUtils    import *

# Grab the general fkin from HW5 P5.
from cs133a_project.KinematicChain     import KinematicChain
from cs133a_project.joint_info import *

from visualization_msgs.msg     import Marker
from visualization_msgs.msg     import MarkerArray


#
#   Trajectory Class
#
class Trajectory():
    # Initialization.
    def __init__(self, node: RobotNode):
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

        self.x0_l_leg = self.l_leg_chain.fkin(self.q0)[0]
        self.x0_r_leg = self.r_leg_chain.fkin(self.q0)[0]

        # put left leg at origin
        self.v_world = np.zeros((3, 1))
        self.R_world = Reye()
        self.p_world = np.zeros((3, 1)) - self.R_world @ self.x0_l_leg

        # trajectory info
        self.T = None
        self.xf_l_leg = None
        self.xf_l_leg_height = 0.3

        self.q = np.vstack((self.q0, self.q0))
        self.pd = self.x0_l_leg
        self.Rd = Reye()
        #self.Rd = Roty(0.5)

        self.collision = False


    def process_ball(self, msg):
        # if msg.markers:
        #     ball = msg.markers[0]
        #     radius = p_from_Vector3(ball.scale)[0, 0] / 2
        #     new_ball_p = p_from_Point(ball.pose.position)
        #     new_ball_t = ball.header.stamp.sec + ball.header.stamp.nanosec*10**-9
        #     if self.ball_p is not None:
        #         self.ball_v = (new_ball_p - self.ball_p) / (new_ball_t - self.ball_t)
        #     self.ball_p = new_ball_p
        #     self.ball_t = new_ball_t

        #     if not self.T and self.ball_v is not None:
        #         roots = np.roots([1/2*self.ball_a[2, 0], self.ball_v[2, 0], self.ball_p[2, 0] - self.xf_l_leg_height])
        #         print([1/2*self.ball_a[2, 0], self.ball_v[2, 0], self.ball_p[2, 0] - self.xf_l_leg_height])
        #         self.T = roots[0] if roots[0] > 0 else roots[1]
        #         print(self.T)
        #         ball_final = self.ball_p + self.ball_v * self.T + 1/2*self.ball_a * self.T**2

        #         self.xf_l_leg = self.R_world.T @ (ball_final - self.p_world)

        #     # ball wrt pelvis
        #     pelvis_ball_p = self.R_world.T @ (new_ball_p - self.p_world)

        #     # if ((pelvis_ball_p <= self.pd + radius) & (self.pd - radius <= pelvis_ball_p)).all():
        #     #     if not self.collision:
        #     #         self.node.collision_pub.publish(Pose_from_T(T_from_Rp(self.R_world @ self.Rd, self.p_world  + self.R_world @ self.pd)))
        #     #         self.node.collision = True
        #     #         print("collision")
        #     # else:
        #     #     self.collision = False
        pass
            

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
    
    # converts frame relative to pelvis to world frame
    def convert_to_world(self, pd, Rd):
        return self.p_world + self.R_world @ pd, self.R_world @ Rd
    
    def check_touching(self, pd, Rd):
        # transform ball into foot coordinates
        ball_p_foot = Rd @ (self.node.ball_p - pd)

        return ((ball_p_foot <= ATLAS_PADDLE_DIMENSION/2 + self.node.ball_r) & (ball_p_foot >= -ATLAS_PADDLE_DIMENSION/2 - self.node.ball_r)).all()

    def check_collision(self):
        p, R = self.convert_to_world(self.pd, self.Rd)
        if self.check_touching(p, R):
            # insures collision is only processed once
            if not self.collision:
                self.node.collision.collision_bool = True
                self.node.collision.T = T_from_Rp(R, p)
                self.collision = True

                self.recalculate() # recalculate trajectory
            else:
                self.node.collision.collision_bool = False
        else:
            self.collision = False
            self.node.collision.collision_bool = False


    # Evaluate at the given time.  This was last called (dt) ago.
    def evaluate(self, t, dt):
        self.check_collision()

        pd, vd = self.x0_l_leg, np.zeros((3, 1))

        Rd, wd = self.Rd, np.zeros((3, 1))

        qlast_l_leg = self.q[:len(ATLAS_L_LEG_JOINT_NAMES), :]

        pdlast = self.pd
        Rdlast = self.Rd

        q_l_leg, qdot_l_leg = self.ikin_leg(qlast_l_leg, pdlast, Rdlast, self.l_leg_chain, pd, vd, Rd, wd, dt)
        #q_r_leg, qdot_r_leg = self.ikin_leg(qlast_l_leg, pdlast, Rdlast, self.r_leg_chain, pd, vd, Rd, wd, dt)
        q_r_leg, qdot_r_leg = np.zeros((len(ATLAS_R_LEG_JOINT_NAMES), 1)), np.zeros((len(ATLAS_R_LEG_JOINT_NAMES), 1))

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

    project_node = ProjectNode(100, Trajectory)
    project_node.start()

    # Shutdown the node and ROS.
    project_node.shutdown()

if __name__ == "__main__":
    main(
)
