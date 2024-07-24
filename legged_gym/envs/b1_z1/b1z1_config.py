# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np

class B1Z1Cfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_actions = 19
        n_proprio = 69
        # n_proprio = 69 + 3
        history_len = 0
        num_observations = n_proprio * history_len + n_proprio
        num_heights = 187
        # num_observations += num_heights # Add height field map

    class terrain( LeggedRobotCfg.terrain ):
        measure_heights = False

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.5] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.2,   # [rad]
            'FL_thigh_joint': 0.8,     # [rad]
            'FL_calf_joint': -1.5,   # [rad]

            'RL_hip_joint': 0.2,   # [rad]
            'RL_thigh_joint': 0.8,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]

            'FR_hip_joint': -0.2 ,  # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'FR_calf_joint': -1.5,  # [rad]

            'RR_hip_joint': -0.2,   # [rad]
            'RR_thigh_joint': 0.8,   # [rad]
            'RR_calf_joint': -1.5,    # [rad]

            'z1_waist': 0.0,
            'z1_shoulder': 1.48,
            'z1_elbow': -1.5,
            'z1_wrist_angle': 0,
            'z1_forearm_roll': 0.0,
            'z1_wrist_rotate': 1.57,
            'z1_jointGripper': -0.785,
        }
        # dof_names ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FR_hip_joint', 'FR_thigh_joint', 
        # 'FR_calf_joint', 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 
        # 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', 
        # 'z1_waist', 'z1_shoulder', 'z1_elbow', 'z1_wrist_angle', 'z1_forearm_roll', 'z1_wrist_rotate', 'z1_jointGripper']

    class goal_ee:
        num_commands = 3
        traj_time = [1, 3]
        hold_time = [0.5, 2]
        num_collision_check_samples = 10
        command_mode = 'sphere'
        collision_upper_limits = [0.1, 0.2, 0.1]
        collision_lower_limits = [-0.8, -0.2, -0.4]
        underground_limit = -0.7
        arm_induced_pitch = 0.38 # Added to -pos_p (negative goal pitch) to get default eef orn_p
        sphere_error_scale = [1, 1, 1]#[1 / (ranges.final_pos_l[1] - ranges.final_pos_l[0]), 1 / (ranges.final_pos_p[1] - ranges.final_pos_p[0]), 1 / (ranges.final_pos_y[1] - ranges.final_pos_y[0])]
        orn_error_scale = [1, 1, 1]
    
        class sphere_center:
            x_offset = 0.3 # Relative to base
            y_offset = 0 # Relative to base
            z_invariant_offset = 0.7 # Relative to terrain
        
        class sphere_center_stand:
            x_offset = 0.3
            y_offset = 0
            z_offset = 0.09

        class ranges:
            init_pos_start = [0.5, np.pi/8, 0]
            init_pos_end = [0.7, 0, 0]
            pos_l = [0.4, 1.5]
            pos_p = [-1 * np.pi / 2.5, 1 * np.pi / 3]
            pos_y = [-1.2, 1.2]

            pos_l_stand = [0.7, 0.7]
            # pos_p_stand = [-1 * np.pi / 4, 1 * np.pi / 6]
            # pos_p_stand = [-1 * np.pi / 6, -1 * np.pi / 12]
            pos_p_stand = [-1 * np.pi / 6, -1 * np.pi / 12]
            # pos_y_stand = [-1 * np.pi / 4, 1 * np.pi / 4]
            pos_y_stand = [0, 0]

            delta_orn_r = [-0.5, 0.5]
            delta_orn_p = [-0.5, 0.5]
            delta_orn_y = [-0.5, 0.5]

            final_tracking_ee_reward = 0.55
    
    class arm:
        init_target_ee_base = [0.2, 0.0, 0.2]
        grasp_offset = 0.08
        osc_kp = np.array([100, 100, 100, 30, 30, 30])
        osc_kd = 2 * (osc_kp ** 0.5)


    class commands( LeggedRobotCfg.commands ):
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [0., 0.] # min max [m/s]
            lin_vel_y = [0., 0.]   # min max [m/s]
    
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters: only for legs
        control_type = 'P'
        stiffness = {'joint': 80.}  # [N*m/rad]
        damping = {'joint': 2.0}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/b1z1/urdf/b1z1.urdf'
        name = "b1z1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "trunk"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        gripper_name = "ee_gripper_link"

  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.8
        only_positive_rewards = False
        class scales( LeggedRobotCfg.rewards.scales ):
            # torques = -0.0002
            torques = 0
            dof_pos_limits = -10.0
            stand_up_x = 3
            flfr_footforce = -5 # -0.2
            orientation = -0.2
            base_height = -1 # -0.5
            feet_air_time = 0
            tracking_lin_vel = 0.2
            tracking_ang_vel = 0.2
            arm_action = -0.1
            stand_front_feet_shrink = -0.1
            # tracking_ee_world = 0.1


class B1Z1CfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'b1z1'
        max_iterations = 5000

  