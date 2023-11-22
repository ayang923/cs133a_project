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
from cs133a_project.GeneratorNode      import GeneratorNode
from cs133a_project.TransformHelpers   import *
from cs133a_project.TrajectoryUtils    import *

# Grab the general fkin from HW5 P5.
from cs133a_project.KinematicChain     import KinematicChain
from cs133a_project.joint_info import ATLAS_JOINT_NAMES, ATLAS_L_LEG_JOINT_NAMES, ATLAS_R_LEG_JOINT_NAMES

#
#   Trajectory Class
#
class Trajectory():
    # Initialization.
    def __init__(self, node):
        # Set up the kinematic chain object.
        #self.chain = KinematicChain(node, 'world', 'tip', self.jointnames())

        self.l_leg_chain = KinematicChain(node, 'l_uglut', 'l_foot', ATLAS_L_LEG_JOINT_NAMES)
        self.r_leg_chain = KinematicChain(node, 'r_uglut', 'r_foot', ATLAS_R_LEG_JOINT_NAMES)

        self.q0 = np.zeros((len(ATLAS_L_LEG_JOINT_NAMES), 1))

        self.x0 = self.l_leg_chain.fkin(self.q0)[0]
        self.xf = self.x0 + np.array([0, 0, 0.5]).reshape(3, 1)

        self.q = np.vstack((self.q0, self.q0))
        self.pd = self.x0
        self.Rd = Reye()

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
        t = t % 10

        if t < 5:
            pd, vd = goto(t, 5, self.x0, self.xf)
            Rd, wd = Reye(), np.zeros((3, 1))
        else:
            pd, vd = goto(t-5, 5, self.xf, self.x0)
            Rd, wd = Reye(), np.zeros((3, 1))

        qlast_l_leg = self.q[:len(ATLAS_L_LEG_JOINT_NAMES), :]
        qlast_r_leg = self.q[len(ATLAS_L_LEG_JOINT_NAMES):, :]

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
        return (q.flatten().tolist(), qdot.flatten().tolist())


#
#  Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Initialize the generator node for 100Hz udpates, using the above
    # Trajectory class.
    generator = GeneratorNode('generator', 100, Trajectory)

    # Spin, meaning keep running (taking care of the timer callbacks
    # and message passing), until interrupted or the trajectory ends.
    generator.spin()

    # Shutdown the node and ROS.
    generator.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main(
)


