import numpy as np

ATLAS_JOINT_NAMES = [
    "back_bkx",
    "back_bky",
    "back_bkz",
    "l_arm_elx",
    "l_arm_ely",
    "l_arm_shx",
    "l_arm_shz",
    "l_arm_wrx",
    "l_arm_wry",
    "l_arm_wry2",
    "l_leg_akx",
    "l_leg_aky",
    "l_leg_hpx",
    "l_leg_hpy",
    "l_leg_hpz",
    "l_leg_kny",
    "neck_ry",
    "r_arm_elx",
    "r_arm_ely",
    "r_arm_shx",
    "r_arm_shz",
    "r_arm_wrx",
    "r_arm_wry",
    "r_arm_wry2",
    "r_leg_akx",
    "r_leg_aky",
    "r_leg_hpx",
    "r_leg_hpy",
    "r_leg_hpz",
    "r_leg_kny",
]

ATLAS_L_LEG_JOINT_NAMES = [
    "l_leg_hpz",
    "l_leg_hpx",
    "l_leg_hpy",
    "l_leg_kny",
    "l_leg_aky",
    "l_leg_akx",
]

ATLAS_R_LEG_JOINT_NAMES = [
    "r_leg_hpz",
    "r_leg_hpx",
    "r_leg_hpy",
    "r_leg_kny",
    "r_leg_aky",
    "r_leg_akx",
]

ATLAS_R_ARM_JOINT_NAMES = [
    "back_bkz",
    "back_bky",
    "back_bkx",
    "r_arm_shz",
    "r_arm_shx",
    "r_arm_ely",
    "r_arm_elx",
    "r_arm_wry",
    "r_arm_wrx",
    "r_arm_wry2"
]

ATLAS_HEAD_JOINT_NAMES = [
    "back_bkz",
    "back_bky",
    "back_bkx",
    "neck_ry"
]

ATLAS_ARM_JOINT_CONSTRAINTS = np.array([
    [-np.pi/2, np.pi/2],
    [-np.pi/2, np.pi/2],
    [-np.pi/2, np.pi/2],
    [-np.pi/2, np.pi/2],
    [-np.pi/2, np.pi/2],
    [0, np.pi],
    [-np.pi/2, 0],
    [-np.pi/2, np.pi/2],
    [-np.pi/2, np.pi/2],
    [-np.pi/2, np.pi/2]
])

ATLAS_R_LEG_JOINT_CONSTRAINTS = np.array([
    [-np.pi/2, np.pi/2],
    [-np.pi/2, np.pi/2],
    [-np.pi/2, np.pi/2],
    [0, 2*np.pi/3],
    [-np.pi/6, np.pi/6], 
    [-np.pi/6, np.pi/6]
])

# left knee constrained more
ATLAS_L_LEG_JOINT_CONSTRAINTS = np.array([
    [-np.pi/2, np.pi/2],
    [-np.pi/2, np.pi/2],
    [-np.pi/2, np.pi/2],
    [0, np.pi/3],
    [-np.pi/6, np.pi/6], 
    [-np.pi/6, np.pi/6]
])

ATLAS_PADDLE_DIMENSION = np.array([.4, .15, .075]).reshape((3, 1))