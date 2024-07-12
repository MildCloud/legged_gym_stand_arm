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

class B1Z1Cfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_actions = 19
        n_proprio = 69
        history_len = 20
        num_observations = n_proprio * history_len + n_proprio
        num_heights = 187
        num_observations += num_heights

    class terrain( LeggedRobotCfg.terrain ):
        measure_heights = True

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

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
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
        penalize_contacts_on = ["thigh", "calf", "trunk"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.8
        only_positive_rewards = False
        class scales( LeggedRobotCfg.rewards.scales ):
            # torques = -0.0002
            torques = 0
            dof_pos_limits = -10.0
            stand_up_x = 2
            # flfr_footforce = -0.2
            flfr_footforce = 0.
            orientation = -0.2
            base_height = -0.5
            feet_air_time = 0

class B1Z1CfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'b1z1'

  