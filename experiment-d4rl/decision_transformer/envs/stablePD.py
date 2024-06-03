import numpy as np
class StablePD:
    def __init__(self, sim):
        self.sim = sim
        self.qpos = self.sim.data.qpos
        self.qvel = self.sim.data.qvel
        self.qacc = self.sim.data.qacc
        self.PID_Gains = _RRIS_Constants.RAJAGOPAL_JOINT_PID
        self.joint_namelist = _RRIS_Constants.jointName_with_actuators
    def CalcControlForces(self,target_pose):
       
        '''
        https://mujoco.readthedocs.io/en/latest/computation.html#simulation-pipeline
        - Step 20 indicates that qacc already takes into account the constraint forces

        - Using equation 5 of stable PD paper
        '''
        t = self.sim.model.opt.timestep

        Kp = self.GetKp() # 1D np-array
        Kd = self.GetKd() # 1D np-array

        qpos = self.qpos # 1D np-array
        qvel = self.qvel # 1D np-array
        qacc = self.qacc # 1D np-array

        self.target_vel=self.BuildTargetVel()

        tar_pose = target_pose
        tar_vel = self.target_vel

        pose_err = tar_pose - qpos
        vel_err = tar_vel - qvel

        out_tau = np.zeros(qpos.shape)
        out_tau = Kp * pose_err + Kd * (vel_err - t * qacc)
        return out_tau
    def BuildTargetVel(self):
        return np.zeros(self.qvel.shape)
    def GetKp(self):
        # Proportional Gain
        Kp_list = []
        for jointName in self.joint_namelist:
            if jointName in self.PID_Gains:
                Kp_list.append(self.PID_Gains[jointName][0])
            else:
                Kp_list.append(0)
        Kp = np.array(Kp_list)
        return Kp
    
    def GetKi(self):
        # Integral Gain (should be 0 for all if its a PD controller)
        Ki_list = []
        for jointName in self.joint_namelist:
            if jointName in self.PID_Gains:
                Ki_list.append(self.PID_Gains[jointName][1])
            else:
                Ki_list.append(0)
        Ki = np.array(Ki_list)
        return Ki

    def GetKd(self):
        # Derivative Gain
        Kd_list = []
        for jointName in self.joint_namelist:
            if jointName in self.PID_Gains:
                Kd_list.append(self.PID_Gains[jointName][2])
            else:
                Kd_list.append(0)
        Kd = np.array(Kd_list)
        return Kd
    


class _RRIS_Constants():
        POS_DIM = 3
        ROT_DIM = 6
        LIN_VEL_DIM = 3
        ANG_VEL_DIM = 3

        # Simplified Model
        NUM_BALL_JOINTS_SIMPLIFIED = 6 # Pelvis, Hip R, Hip L, Lumbar, Arm R, Arm L
        NUM_REVOLUTE_JOINTS_SIMPLIFIED = 6 # Knee R, Knee L, Elbow R, Elbow L, Ankle R, Ankle L
        NUM_EEF_BODIES = 6 # Calcn R, Calcn L, Toes R, Toes L, Hand R, Hand L
        
        RAJAGOPAL_ROOT_JOINT_NAMES = ["pelvis_tx", "pelvis_ty", "pelvis_tz", "pelvis_list", "pelvis_tilt", "pelvis_rotation"]
        RAJAGOPAL_EEF_BODY_NAMES = ["calcn_r", "toes_r", "calcn_l", "toes_l", "hand_r", "hand_l"]

        # For Rajagopal Model
        # Geoms to exclude if contact detected for early termination with floor
        RAJAGOPAL_EARLY_TERMINATION = ["r_foot", "r_bofoot", "l_foot", "l_bofoot"]

        # Key -> Joint Name in MJCF
        # Value -> Weights of the joint (used for pose error calculation, not the actual "weight/mass")
        RAJAGOPAL_JOINT_WEIGHTS = {
            "pelvis_tz"                         : 1, 
            "pelvis_ty"                         : 1,
            "pelvis_tx"                         : 1,
            "pelvis_tilt"                       : 1,
            "pelvis_list"                       : 1,
            "pelvis_rotation"                   : 1,

            "hip_flexion_r"                     : 1,
            "hip_adduction_r"                   : 1,
            "hip_rotation_r"                    : 1,
            "knee_angle_r"                      : 1,
            "ankle_angle_r"                     : 1,

            "hip_flexion_l"                     : 1,
            "hip_adduction_l"                   : 1,
            "hip_rotation_l"                    : 1,
            "knee_angle_l"                      : 1,
            "ankle_angle_l"                     : 1,

            "lumbar_extension"                  : 1,
            "lumbar_bending"                    : 1,
            "lumbar_rotation"                   : 1,

            "arm_flex_r"                        : 1,
            "arm_add_r"                         : 1,
            "arm_rot_r"                         : 1,
            "elbow_flex_r"                      : 1,

            "arm_flex_l"                        : 1,
            "arm_add_l"                         : 1,
            "arm_rot_l"                         : 1,
            "elbow_flex_l"                      : 1,

            "pro_sup_r"                         : 1,
            "wrist_flex_r"                      : 1,
            "wrist_dev_r"                       : 1,

            "pro_sup_l"                         : 1,
            "wrist_flex_l"                      : 1,
            "wrist_dev_l"                       : 1,

            "mtp_angle_l"                       : 1,
            "mtp_angle_r"                       : 1,
            "subtalar_angle_l"                  : 1,
            "subtalar_angle_r"                  : 1,

            "knee_angle_r_translation1"         : 1,
            "knee_angle_r_translation2"         : 1,
            "knee_angle_r_rotation2"            : 1,
            "knee_angle_r_rotation3"            : 1,

            "knee_angle_r_beta_translation1"    : 1,
            "knee_angle_r_beta_translation2"    : 1,
            "knee_angle_r_beta_rotation1"       : 1,

            "knee_angle_l_translation1"         : 1,
            "knee_angle_l_translation2"         : 1,
            "knee_angle_l_rotation2"            : 1,
            "knee_angle_l_rotation3"            : 1,

            "knee_angle_l_beta_translation1"    : 1,
            "knee_angle_l_beta_translation2"    : 1,
            "knee_angle_l_beta_rotation1"       : 1,   
        }

        # Kp, Ki, Kd values for each joint
        RAJAGOPAL_JOINT_PID = {
            "pelvis_tz"                         : [0, 0, 0], 
            "pelvis_ty"                         : [0, 0, 0],
            "pelvis_tx"                         : [0, 0, 0],
            "pelvis_tilt"                       : [0, 0, 0],
            "pelvis_list"                       : [0, 0, 0],
            "pelvis_rotation"                   : [0, 0, 0],

            "hip_flexion_r"                     : [500, 0, 50],
            "hip_adduction_r"                   : [500, 0, 50],
            "hip_rotation_r"                    : [500, 0, 50],
            "knee_angle_r"                      : [500, 0, 50],
            "ankle_angle_r"                     : [400, 0, 40],

            "hip_flexion_l"                     : [500, 0, 50],
            "hip_adduction_l"                   : [500, 0, 50],
            "hip_rotation_l"                    : [500, 0, 50],
            "knee_angle_l"                      : [500, 0, 50],
            "ankle_angle_l"                     : [400, 0, 40],

            "lumbar_extension"                  : [1000, 0, 100],
            "lumbar_bending"                    : [1000, 0, 100],
            "lumbar_rotation"                   : [1000, 0, 100],

            "arm_flex_r"                        : [400, 0, 40],
            "arm_add_r"                         : [400, 0, 40],
            "arm_rot_r"                         : [400, 0, 40],
            "elbow_flex_r"                      : [300, 0, 30],

            "arm_flex_l"                        : [400, 0, 40],
            "arm_add_l"                         : [400, 0, 40],
            "arm_rot_l"                         : [400, 0, 40],
            "elbow_flex_l"                      : [300, 0, 30],        

            "pro_sup_r"                         : [1, 0, 0.1],
            "wrist_flex_r"                      : [1, 0, 0.1],
            "wrist_dev_r"                       : [1, 0, 0.1],

            "pro_sup_l"                         : [1, 0, 0.1],
            "wrist_flex_l"                      : [1, 0, 0.1],
            "wrist_dev_l"                       : [1, 0, 0.1],

            "mtp_angle_l"                       : [1, 0, 0.1],
            "mtp_angle_r"                       : [1, 0, 0.1],
            "subtalar_angle_l"                  : [1, 0, 0.1],
            "subtalar_angle_r"                  : [1, 0, 0.1],

            "knee_angle_r_translation1"         : [1, 0, 0.1],
            "knee_angle_r_translation2"         : [1, 0, 0.1],
            "knee_angle_r_rotation2"            : [1, 0, 0.1],
            "knee_angle_r_rotation3"            : [1, 0, 0.1],

            "knee_angle_r_beta_translation1"    : [1, 0, 0.1],
            "knee_angle_r_beta_translation2"    : [1, 0, 0.1],
            "knee_angle_r_beta_rotation1"       : [1, 0, 0.1],

            "knee_angle_l_translation1"         : [1, 0, 0.1],
            "knee_angle_l_translation2"         : [1, 0, 0.1],
            "knee_angle_l_rotation2"            : [1, 0, 0.1],
            "knee_angle_l_rotation3"            : [1, 0, 0.1],

            "knee_angle_l_beta_translation1"    : [1, 0, 0.1],
            "knee_angle_l_beta_translation2"    : [1, 0, 0.1],
            "knee_angle_l_beta_rotation1"       : [1, 0, 0.1],
        }
        jointName_with_actuators=['pelvis_tx',
                                 'pelvis_ty', 
                                 'pelvis_tz', 
                                 'pelvis_tilt', 
                                 'pelvis_list', 
                                 'pelvis_rotation', 
                                 'hip_flexion_r', 
                                 'hip_adduction_r', 
                                 'hip_rotation_r', 
                                 'knee_angle_r', 
                                 'ankle_angle_r',
                                 'hip_flexion_l', 
                                 'hip_adduction_l', 
                                 'hip_rotation_l', 
                                 'knee_angle_l', 
                                 'ankle_angle_l',
                                 "lumbar_extension",
                                 "lumbar_bending",
                                "lumbar_rotation",
                                "arm_flex_r",
                                "arm_add_r",
                                "arm_rot_r",
                                "elbow_flex_r",
                                "arm_flex_l",
                                "arm_add_l",
                                "arm_rot_l",
                                "elbow_flex_l"]