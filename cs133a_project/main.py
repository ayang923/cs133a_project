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
        self.l_leg_chain = KinematicChain(self.node, 'pelvis', 'l_foot_paddle', ATLAS_L_LEG_JOINT_NAMES)
        self.r_leg_chain = KinematicChain(self.node, 'pelvis', 'r_foot_paddle', ATLAS_R_LEG_JOINT_NAMES)

        self.q0 = np.zeros((len(ATLAS_L_LEG_JOINT_NAMES), 1))

        # initial positions, in world frame
        self.x0_l_leg = np.zeros((3, 1)) # world frame
        self.theta0_l_leg = 0.0

        # transform of pelvis wrt world
        self.v_world = np.zeros((3, 1))
        self.R_world = Reye()
        self.p_world = np.zeros((3, 1)) - self.R_world @ self.l_leg_chain.fkin(self.q0)[0]

        # trajectory info
        self.T = None
        self.xf_l_leg = None # world frame
        self.thetaf_l_leg = self.theta0_l_leg

        self.q = np.vstack((self.q0, self.q0))
        self.pd = self.x0_l_leg
        self.Rd = Roty(self.theta0_l_leg)

        self.collision = False

        self.start_time = 0.0

        x0_r_leg = self.r_leg_chain.fkin(self.q0)[0].reshape((3, 1))
        self.x_r_leg = self.convert_to_world(x0_r_leg, Reye())[0]

    # Declare the joint names.
    def jointnames(self):
        # Return a list of joint names FOR THE EXPECTED URDF!
        return ATLAS_JOINT_NAMES
    
    def ikin_leg(self, qlast, pdlast, Rdlast, kinematic_chain, pd, vd, Rd, wd, dt):
        p, R, Jv, Jw = kinematic_chain.fkin(qlast)
        
        J = np.vstack((Jv, Jw))
        xdot = np.vstack((vd, wd))

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
    
    # converts frame relative to world to pelvis frame
    def convert_to_pelvis(self, pd, Rd):
        return self.R_world.T @ (pd - self.p_world), self.R_world.T @ Rd
    
    # converts frame relative to right leg frame, arguments are positions that are wrt same frame
    def convert_to_r_leg(self, pd, Rd, pd_r_leg, Rd_r_leg):
        return Rd_r_leg.T @ (pd - pd_r_leg), Rd.T @ Rd_r_leg
        
    def recalculate(self):
        # upward time
        T_up = -self.node.ball_v[2, 0] / self.node.ball_a[2, 0]
        b_up = self.node.ball_p + self.node.ball_v * T_up + 1/2*self.node.ball_a * T_up**2

        # chooses random height from 0.1 to 0.5
        roots = np.roots([1/2*self.node.ball_a[2, 0], 0, b_up[2, 0] - np.random.uniform(0.1, 0.5)])
        T_down = roots[0] if roots[0] > 0 else roots[1]

        self.T = T_up + T_down

        self.xf_l_leg = self.node.ball_p + self.node.ball_v * self.T + 1/2*self.node.ball_a * self.T**2
        self.thetaf_l_leg = np.random.uniform(-0.1, 0.1)

    def check_touching(self, pd, Rd):
        # transform ball into foot coordinates
        ball_p_foot = Rd @ (self.node.ball_p - pd)

        return ((ball_p_foot <= ATLAS_PADDLE_DIMENSION/2 + self.node.ball_r) & (ball_p_foot >= -ATLAS_PADDLE_DIMENSION/2 - self.node.ball_r)).all()

    def check_collision(self):
        if self.check_touching(self.pd, self.Rd):
            # insures collision is only processed once
            if not self.collision:
                self.node.collision.collision_bool = True
                self.node.collision.T = T_from_Rp(self.Rd, self.pd)
                self.collision = True
                return True
            else:
                self.node.collision.collision_bool = False
        else:
            self.collision = False
            self.node.collision.collision_bool = False

        return False


    # Evaluate at the given time.  This was last called (dt) ago.
    def evaluate(self, t, dt):
        if self.check_collision():
            self.start_time = t
            self.x0_l_leg = self.pd
            self.theta0_l_leg = self.thetaf_l_leg
        
        if self.xf_l_leg is None:
            self.start_time = t
            return (self.p_world, self.R_world, np.zeros((len(ATLAS_JOINT_NAMES), 1)).flatten().tolist(), np.zeros((len(ATLAS_JOINT_NAMES), 1)).flatten().tolist())
            
        # position of right leg in pelvis frame
        p_r_pelvis, Rd_r_pelvis, _, _ = self.r_leg_chain.fkin(self.q[len(ATLAS_L_LEG_JOINT_NAMES):, :])
        p_r_world, Rd_r_world = self.convert_to_world(p_r_pelvis, Rd_r_pelvis)

        # calculate theta wrt world
        theta, thetadot = goto(t - self.start_time, self.T, self.theta0_l_leg, self.thetaf_l_leg)

        # computes trajectories wrt world to be saved
        pd_world, wd_world = goto(t - self.start_time, self.T, self.x0_l_leg, self.xf_l_leg)
        Rd_world = Roty(theta)

        # # desired position of left leg in pelvis frame
        # x0_leg_pelvis, R_pelvis = self.convert_to_pelvis(self.x0_l_leg, Roty(theta))
        # xf_leg_pelvis, _ = self.convert_to_pelvis(self.xf_l_leg, Reye())

        # # desired position of left leg in right leg frame
        # x0_leg_r = Rd_r.T @ (x0_leg_pelvis - p_r)
        # xf_leg_r = Rd_r.T @ (xf_leg_r - p_r)

        x0_l_leg_r, Rd_r = self.convert_to_r_leg(self.x0_l_leg, Rd_world, p_r_world, Rd_r_world)
        xf_l_leg_r, _ = self.convert_to_r_leg(self.xf_l_leg, Rd_world, p_r_world, Rd_r_world)

        vd = goto(t - self.start_time, self.T, x0_l_leg_r, xf_l_leg_r)[1]
        wd = Rd_r_world.T @ (thetadot * ey())

        qlast_l_leg = self.q[:len(ATLAS_L_LEG_JOINT_NAMES), :]
        qlast_r_leg = self.q[len(ATLAS_L_LEG_JOINT_NAMES):, :]

        pdlast, Rdlast = self.convert_to_pelvis(self.pd, self.Rd)

        p_l, R_l, Jv_l, Jw_l = self.l_leg_chain.fkin(qlast_l_leg)
        p_r, R_r, Jv_r, Jw_r = self.r_leg_chain.fkin(qlast_r_leg)

        Jv = R_r.T @ (np.hstack((Jv_l, -Jv_r)) + crossmat(p_l - p_r) @ np.hstack((np.zeros((3, len(ATLAS_L_LEG_JOINT_NAMES))), Jw_r)))
        Jw = R_r.T @ (np.hstack((Jw_l, -Jw_r)))
        J = np.vstack((Jv, Jw))
        xdot = np.vstack((vd, wd))

        e = np.vstack((ep(pdlast, p_l), eR(Rdlast, R_l)))
        J_inv = self.weighted_svd_inverse(J)

        qdot = J_inv @ (xdot)
        q = self.q + dt*qdot

        #q_l_leg, qdot_l_leg = self.ikin_leg(qlast_l_leg, pdlast, Rdlast, self.l_leg_chain, pd_world, vd, Rd_world, wd, dt)
        #q_r_leg, qdot_r_leg = self.ikin_leg(qlast_l_leg, pdlast, Rdlast, self.r_leg_chain, pd, vd, Rd, wd, dt)
        #q_r_leg, qdot_r_leg = np.zeros((len(ATLAS_R_LEG_JOINT_NAMES), 1)), np.zeros((len(ATLAS_R_LEG_JOINT_NAMES), 1))

        self.q = q
        self.pd, self.Rd = pd_world, Rd_world

        q_dict = dict(zip(ATLAS_L_LEG_JOINT_NAMES + ATLAS_R_LEG_JOINT_NAMES, q.flatten()))
        qdot_dict = dict(zip(ATLAS_L_LEG_JOINT_NAMES + ATLAS_R_LEG_JOINT_NAMES, qdot.flatten()))

        q = np.array([q_dict[joint_name] if joint_name in q_dict else 0 for joint_name in self.jointnames()])
        qdot = np.array([qdot_dict[joint_name] if joint_name in q_dict else 0 for joint_name in self.jointnames()])
        
        self.p_world = self.x_r_leg - self.R_world @ p_r
        self.R_world = R_r.T
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
