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

from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR

from .visual_whole_body_math import *
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi
from legged_gym.utils.helpers import class_to_dict

class B1Z1(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.goal_ee_ranges = class_to_dict(self.cfg.goal_ee.ranges)
    
    def _init_buffers(self):
        super()._init_buffers()
        # actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        # self._root_states = gymtorch.wrap_tensor(actor_root_state).view(self.num_envs, 2, 13) # 2 actors
        # self.root_states = gymtorch.wrap_tensor(actor_root_state)
        # self.box_root_state = self._root_states[:, 1, :]

        base_roll = euler_from_quat(self.base_quat)[0]
        base_pitch = euler_from_quat(self.base_quat)[1]
        base_yaw = euler_from_quat(self.base_quat)[2]
        self.arm_base_offset = torch.tensor([0.3, 0., 0.09], device=self.device, dtype=torch.float).repeat(self.num_envs, 1)
        self.base_yaw_euler = torch.cat([torch.zeros(self.num_envs, 2, device=self.device), base_yaw.view(-1, 1)], dim=1)
        self.base_yaw_quat = quat_from_euler_xyz(torch.tensor(0), torch.tensor(0), base_yaw)
        self.base_pitch_quat = quat_from_euler_xyz(torch.tensor(0), base_pitch, torch.tensor(0))
        self.base_roll_pitch_quat = quat_from_euler_xyz(base_roll, base_pitch, torch.tensor(0))
        self.base_pitch_yaw_quat = quat_from_euler_xyz(torch.tensor(0), base_pitch, base_yaw)

        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self._contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.contact_forces = self._contact_forces[:, :-1, :]
        self.box_contact_force = self._contact_forces[:, -1, :]

        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, self.num_bodies, 13)
        self.rigid_body_state = self._rigid_body_state[:, :-1, :]
        self.box_rigid_body_state = self._rigid_body_state[:, -1, :]

        jacobian_tensor = self.gym.acquire_jacobian_tensor(self.sim, 'b1z1')
        self.jacobian_whole = gymtorch.wrap_tensor(jacobian_tensor)

        self.gripper_idx = self.body_names_to_idx[self.cfg.asset.gripper_name]
        self.ee_pos = self.rigid_body_state[:, self.gripper_idx, :3]
        self.ee_orn = self.rigid_body_state[:, self.gripper_idx, 3:7]
        self.ee_vel = self.rigid_body_state[:, self.gripper_idx, 7:]
        self.ee_j_eef = self.jacobian_whole[:, self.gripper_idx, :6, -7:-1]

        # box info & target_ee info
        # self.box_pos = self.box_root_state[:, 0:3]
        self.grasp_offset = self.cfg.arm.grasp_offset
        self.init_target_ee_base = torch.tensor(self.cfg.arm.init_target_ee_base, device=self.device).unsqueeze(0)

        self.traj_timesteps = torch_rand_float(self.cfg.goal_ee.traj_time[0], self.cfg.goal_ee.traj_time[1], (self.num_envs, 1), device=self.device).squeeze(1) / self.dt
        self.traj_total_timesteps = self.traj_timesteps + torch_rand_float(self.cfg.goal_ee.hold_time[0], self.cfg.goal_ee.hold_time[1], (self.num_envs, 1), device=self.device).squeeze(1) / self.dt
        self.goal_timer = torch.zeros(self.num_envs, device=self.device)
        self.ee_start_sphere = torch.zeros(self.num_envs, 3, device=self.device)

        self.ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)

        self.ee_goal_orn_euler = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_orn_euler[:, 0] = np.pi / 2
        self.ee_goal_orn_quat = quat_from_euler_xyz(self.ee_goal_orn_euler[:, 0], self.ee_goal_orn_euler[:, 1], self.ee_goal_orn_euler[:, 2])
        self.ee_goal_orn_delta_rpy = torch.zeros(self.num_envs, 3, device=self.device)

        self.curr_ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)

        self.init_start_ee_sphere = torch.tensor(self.cfg.goal_ee.ranges.init_pos_start, device=self.device).unsqueeze(0)
        self.init_end_ee_sphere = torch.tensor(self.cfg.goal_ee.ranges.init_pos_end, device=self.device).unsqueeze(0)

        self.collision_lower_limits = torch.tensor(self.cfg.goal_ee.collision_lower_limits, device=self.device, dtype=torch.float)
        self.collision_upper_limits = torch.tensor(self.cfg.goal_ee.collision_upper_limits, device=self.device, dtype=torch.float)
        self.underground_limit = self.cfg.goal_ee.underground_limit
        self.num_collision_check_samples = self.cfg.goal_ee.num_collision_check_samples
        self.collision_check_t = torch.linspace(0, 1, self.num_collision_check_samples, device=self.device)[None, None, :]
        assert(self.cfg.goal_ee.command_mode in ['cart', 'sphere'])
        self.sphere_error_scale = torch.tensor(self.cfg.goal_ee.sphere_error_scale, device=self.device)
        self.orn_error_scale = torch.tensor(self.cfg.goal_ee.orn_error_scale, device=self.device)
        self.ee_goal_center_offset = torch.tensor([self.cfg.goal_ee.sphere_center.x_offset, 
                                                   self.cfg.goal_ee.sphere_center.y_offset, 
                                                   self.cfg.goal_ee.sphere_center.z_invariant_offset], 
                                                   device=self.device).repeat(self.num_envs, 1)
        self.ee_goal_center_offset_stand = torch.tensor([self.cfg.goal_ee.sphere_center_stand.x_offset, 
                                                   self.cfg.goal_ee.sphere_center_stand.y_offset, 
                                                   self.cfg.goal_ee.sphere_center_stand.z_offset], 
                                                   device=self.device).repeat(self.num_envs, 1)
        
        self.curr_ee_goal_cart_world = self.get_ee_goal_spherical_center() + quat_apply(self.base_yaw_quat, self.curr_ee_goal_cart)

        if self.cfg.env.history_len > 0:
            self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.n_proprio, device=self.device, dtype=torch.float)
            print('self.obs_history_buf', self.obs_history_buf.shape)
        
        self.global_steps = 0

    def compute_observations(self):
        """ Computes observations
        """
        dof_pos = self.dof_pos.clone()
        # dof_pos[:, -7:] = self.default_dof_pos[:, -7:]
        dof_vel = self.dof_vel.clone()
        # dof_vel[:, -7:] = 0
        arm_base_pos = self.root_states[:, :3] + quat_apply(self.base_quat, self.arm_base_offset)
        ee_goal_local_cart = quat_rotate_inverse(self.base_quat, self.curr_ee_goal_cart_world - arm_base_pos)
        prop_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    ee_goal_local_cart,
                                    (dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            prop_height_obs_buf = torch.cat((prop_obs_buf, heights), dim=-1)
        else:
            prop_height_obs_buf = prop_obs_buf
        # add noise if needed
        if self.add_noise:
            prop_height_obs_buf += (2 * torch.rand_like(prop_height_obs_buf) - 1) * self.noise_scale_vec
        
        if self.cfg.env.history_len > 0:
            self.obs_buf = torch.cat([prop_height_obs_buf, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
            self.obs_history_buf = torch.where(
                (self.episode_length_buf <= 1)[:, None, None], 
                torch.stack([prop_obs_buf] * self.cfg.env.history_len, dim=1),
                torch.cat([self.obs_history_buf[:, 1:], prop_obs_buf.unsqueeze(1)], dim=1)
            )
        else:
            self.obs_buf = prop_height_obs_buf

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        if self.cfg.terrain.measure_heights:
            noise_vec = torch.zeros(self.num_envs, cfg.env.n_proprio + cfg.env.num_heights, device=self.device, dtype=torch.float)
        else:
            noise_vec = torch.zeros(self.num_envs, cfg.env.n_proprio, device=self.device, dtype=torch.float)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:31] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:31] = 0.
        noise_vec[31:50] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[43:50] = 0.
        noise_vec[50:69] = 0. # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[69:256] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec
    
    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.body_names_to_idx = self.gym.get_asset_rigid_body_dict(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props_asset['driveMode'][12:].fill(gymapi.DOF_MODE_POS)  # set arm to pos control
            dof_props_asset['stiffness'][12:].fill(400)
            dof_props_asset['damping'][12:].fill(40.0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])


    def _draw_collision_bbox(self):
        center = self.ee_goal_center_offset_stand
        bbox0 = center + self.collision_upper_limits
        bbox1 = center + self.collision_lower_limits
        bboxes = torch.stack([bbox0, bbox1], dim=1)
        sphere_geom = gymutil.WireframeSphereGeometry(0.2, 4, 4, None, color=(1, 1, 0))

        for i in range(self.num_envs):
            bbox_geom = gymutil.WireframeBBoxGeometry(bboxes[i], None, color=(1, 0, 0))
            quat = self.base_pitch_quat[i]
            quat = self.base_quat[i]
            r = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])
            pose0 = gymapi.Transform(gymapi.Vec3(self.root_states[i, 0], self.root_states[i, 1], self.root_states[i, 2]), r=r)
            gymutil.draw_lines(bbox_geom, self.gym, self.viewer, self.envs[i], pose=pose0) 

            # pose0 = gymapi.Transform(gymapi.Vec3(bboxes[i, ww 0, 0], bboxes[i, 0, 1], bboxes[i, 0, 2]))    
            # pose1 = gymapi.Transform(gymapi.Vec3(bboxes[i, 1, 0], bboxes[i, 1, 1], bboxes[i, 1, 2]))
            # gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], pose=pose0)
            # gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], pose=pose1)

    def _draw_ee_goal_curr(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(1, 1, 0)) # yellow

        sphere_geom_3 = gymutil.WireframeSphereGeometry(0.05, 16, 16, None, color=(0, 1, 1)) # cyan
        upper_arm_pose = self.get_ee_goal_spherical_center()

        sphere_geom_2 = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(0, 0, 1))
        ee_pose = self.rigid_body_state[:, self.gripper_idx, :3]

        sphere_geom_origin = gymutil.WireframeSphereGeometry(0.1, 8, 8, None, color=(0, 1, 0))
        sphere_pose = gymapi.Transform(gymapi.Vec3(0, 0, 0), r=None)
        gymutil.draw_lines(sphere_geom_origin, self.gym, self.viewer, self.envs[0], sphere_pose)

        axes_geom = gymutil.AxesGeometry(scale=0.2)

        sphere_geom_4 = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(0.7, 0, 1)) # purple
        sphere_geom_5 = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(0.7, 0.4, 0.4)) # orange

        for i in range(self.num_envs):
            sphere_pose = gymapi.Transform(gymapi.Vec3(self.curr_ee_goal_cart_world[i, 0], self.curr_ee_goal_cart_world[i, 1], self.curr_ee_goal_cart_world[i, 2]), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 
            
            sphere_pose_2 = gymapi.Transform(gymapi.Vec3(ee_pose[i, 0], ee_pose[i, 1], ee_pose[i, 2]), r=None)
            gymutil.draw_lines(sphere_geom_2, self.gym, self.viewer, self.envs[i], sphere_pose_2) 

            sphere_pose_3 = gymapi.Transform(gymapi.Vec3(upper_arm_pose[i, 0], upper_arm_pose[i, 1], upper_arm_pose[i, 2]), r=None)
            gymutil.draw_lines(sphere_geom_3, self.gym, self.viewer, self.envs[i], sphere_pose_3) 

            pose = gymapi.Transform(gymapi.Vec3(self.curr_ee_goal_cart_world[i, 0], self.curr_ee_goal_cart_world[i, 1], self.curr_ee_goal_cart_world[i, 2]), 
                                    r=gymapi.Quat(self.ee_goal_orn_quat[i, 0], self.ee_goal_orn_quat[i, 1], self.ee_goal_orn_quat[i, 2], self.ee_goal_orn_quat[i, 3]))
            gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], pose)
            
            sphere_pose_4 = gymapi.Transform(gymapi.Vec3(self.final_ee_goal_cart_world[i, 0], self.final_ee_goal_cart_world[i, 1], self.final_ee_goal_cart_world[i, 2]), r=None)
            gymutil.draw_lines(sphere_geom_4, self.gym, self.viewer, self.envs[i], sphere_pose_4) 

            sphere_pose_5 = gymapi.Transform(gymapi.Vec3(self.final_ee_start_cart_world[i, 0], self.final_ee_start_cart_world[i, 1], self.final_ee_start_cart_world[i, 2]), r=None)
            gymutil.draw_lines(sphere_geom_5, self.gym, self.viewer, self.envs[i], sphere_pose_5) 

    def _draw_ee_goal_traj(self):
        sphere_geom = gymutil.WireframeSphereGeometry(0.005, 8, 8, None, color=(1, 0, 0))
        sphere_geom_yellow = gymutil.WireframeSphereGeometry(0.01, 16, 16, None, color=(1, 1, 0))

        t = torch.linspace(0, 1, 10, device=self.device)[None, None, None, :]
        ee_target_all_sphere = torch.lerp(self.ee_start_sphere[..., None], self.ee_goal_sphere[..., None], t).squeeze(0)
        ee_target_all_cart_world = torch.zeros_like(ee_target_all_sphere)
        for i in range(10):
            ee_target_cart = sphere2cart(ee_target_all_sphere[..., i])
            ee_target_all_cart_world[..., i] = quat_apply(self.base_quat, ee_target_cart)
        ee_target_all_cart_world += self.get_ee_goal_spherical_center()[:, :, None]
        # curr_ee_goal_cart_world = quat_apply(self.base_yaw_quat, self.curr_ee_goal_cart) + self.root_states[:, :3]
        for i in range(self.num_envs):
            for j in range(10):
                pose = gymapi.Transform(gymapi.Vec3(ee_target_all_cart_world[i, 0, j], ee_target_all_cart_world[i, 1, j], ee_target_all_cart_world[i, 2, j]), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], pose)
            # pose_curr = gymapi.Transform(gymapi.Vec3(curr_ee_goal_cart_world[i, 0], curr_ee_goal_cart_world[i, 1], curr_ee_goal_cart_world[i, 2]), r=None)
            # gymutil.draw_lines(sphere_geom_yellow, self.gym, self.viewer, self.envs[i], pose_curr)
        
    
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.total_actions = self.actions.clone()
        self.actions[:, -7:] = 0
        # step physics and render each frame
        self.render()

        dpos = self.curr_ee_goal_cart_world - self.ee_pos
        drot = orientation_error(self.ee_goal_orn_quat, self.ee_orn / torch.norm(self.ee_orn, dim=-1).unsqueeze(-1))
        dpose = torch.cat([dpos, drot], -1).unsqueeze(-1)
        arm_pos_targets = self.control_ik(dpose) + self.dof_pos[:, -7:-1]
        all_pos_targets = torch.zeros_like(self.dof_pos)
        all_pos_targets[:, -7:-1] = torch.where(self.is_init[:, None].repeat([1, 6]), self.default_dof_pos[:, -7:-1], arm_pos_targets)

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.torques[:, -7:] = 0
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            # all_pos_targets = torch.zeros_like(self.dof_pos)
            # all_pos_targets[:, -7:] = self.default_dof_pos[:, -7:]
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(all_pos_targets))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        base_roll = euler_from_quat(self.base_quat)[0]
        base_pitch = euler_from_quat(self.base_quat)[1]
        base_yaw = euler_from_quat(self.base_quat)[2]
        # print('base_yaw in degree', base_yaw / 3.14 * 180)
        self.base_yaw_euler[:] = torch.cat([torch.zeros(self.num_envs, 2, device=self.device), base_yaw.view(-1, 1)], dim=1)
        self.base_yaw_quat[:] = quat_from_euler_xyz(torch.tensor(0), torch.tensor(0), base_yaw)
        self.base_pitch_quat[:] = quat_from_euler_xyz(torch.tensor(0), base_pitch, torch.tensor(0))
        self.base_roll_pitch_quat = quat_from_euler_xyz(base_roll, base_pitch, torch.tensor(0))
        self.base_pitch_yaw_quat = quat_from_euler_xyz(torch.tensor(0), base_pitch, base_yaw)

        self._post_physics_step_callback()
        self.update_curr_ee_goal()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        # if self.viewer and self.enable_viewer_sync and self.debug_viz:
        #     self.gym.clear_lines(self.viewer)
        #     self._draw_ee_goal_curr()
        #     self._draw_ee_goal_traj()
        #     self._draw_collision_bbox()
        if not self.headless:
            self.gym.clear_lines(self.viewer)
            self._draw_ee_goal_curr()
            # self._draw_ee_goal_traj()
            self._draw_collision_bbox()
    
    # def _post_physics_step_callback(self):
    #     """ Callback called before computing terminations, rewards, and observations
    #         Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
    #     """
    #     # 
    #     env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
    #     self._resample_commands(env_ids)
    #     if self.cfg.commands.heading_command:
    #         forward = quat_apply(self.base_quat, self.forward_vec)
    #         heading = torch.atan2(forward[:, 1], forward[:, 0])
    #         self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

    #     if self.cfg.terrain.measure_heights:
    #         self.measured_heights = self._get_heights()
    #     if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
    #         self._push_robots()
    
    def check_termination(self):
        """ Check if environments need to be reset
        """
        r, p, _ = euler_from_quat(self.base_quat) 
        z = self.root_states[:, 2]
        r_term = torch.abs(r) > 0.8
        p_term = torch.abs(p) > 0.8
        z_term = z < 0.1
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        # self.reset_buf = self.reset_buf | self.time_out_buf | r_term | p_term | z_term
        self.reset_buf = self.reset_buf | self.time_out_buf

    def control_ik(self, dpose):
        # solve damped least squares
        j_eef_T = torch.transpose(self.ee_j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (0.05 ** 2)
        # u = (j_eef_T @ torch.inverse(self.ee_j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 6)
        A = torch.bmm(self.ee_j_eef, j_eef_T) + lmbda[None, ...]
        u = torch.bmm(j_eef_T, torch.linalg.solve(A, dpose))#.view(self.num_envs, 6)
        return u.squeeze(-1)
    
    # def control_ik(self, dpose):
    #     j_eef_T = torch.transpose(self.ee_j_eef, 1, 2)
    #     lmbda = torch.eye(6, device=self.device) * (0.05 ** 2)
    #     A = torch.bmm(j_eef_T, self.ee_j_eef) + lmbda[None, ...]
    #     print('dpose.shape', dpose.shape)
    #     u = torch.bmm(torch.bmm(torch.inverse(A), j_eef_T), dpose)
    #     return u.squeeze(-1)

    def get_ee_goal_spherical_center(self): # cyan point in render
        # center = torch.cat([self.root_states[:, :2], torch.zeros(self.num_envs, 1, device=self.device)], dim=1)
        center = self.root_states[:, :3]
        center = center + quat_apply(self.base_quat, self.ee_goal_center_offset_stand)
        return center

    def _resample_ee_goal_sphere_once(self, env_ids):
        self.ee_goal_sphere[env_ids, 0] = torch_rand_float(self.goal_ee_ranges["pos_l"][0], self.goal_ee_ranges["pos_l"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.ee_goal_sphere[env_ids, 1] = torch_rand_float(self.goal_ee_ranges["pos_p"][0], self.goal_ee_ranges["pos_p"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.ee_goal_sphere[env_ids, 2] = torch_rand_float(self.goal_ee_ranges["pos_y"][0], self.goal_ee_ranges["pos_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)

    def _resample_ee_goal_sphere_once_stand(self, env_ids):
        self.ee_goal_sphere[env_ids, 0] = torch_rand_float(self.goal_ee_ranges["pos_l_stand"][0], self.goal_ee_ranges["pos_l_stand"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.ee_goal_sphere[env_ids, 1] = torch_rand_float(self.goal_ee_ranges["pos_p_stand"][0], self.goal_ee_ranges["pos_p_stand"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.ee_goal_sphere[env_ids, 2] = torch_rand_float(self.goal_ee_ranges["pos_y_stand"][0], self.goal_ee_ranges["pos_y_stand"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        base_pitch = euler_from_quat(self.base_quat)[1]
        heigh_ee_goal_sphere = torch.zeros_like(self.ee_goal_sphere)
        heigh_ee_goal_sphere[env_ids, 0] = torch_rand_float(0.9, 0.9, (len(env_ids), 1), device=self.device).squeeze(1)
        heigh_ee_goal_sphere[env_ids, 1] = torch_rand_float(np.pi / 2, np.pi / 2, (len(env_ids), 1), device=self.device).squeeze(1)
        stand_mask = base_pitch[env_ids] > -np.pi / 6
        self.ee_goal_sphere[env_ids] = torch.where(stand_mask[:, None].repeat(1, 3), self.ee_goal_sphere[env_ids], heigh_ee_goal_sphere[env_ids])
        
    def _resample_ee_goal_orn_once(self, env_ids):
        ee_goal_delta_orn_r = torch_rand_float(self.goal_ee_ranges["delta_orn_r"][0], self.goal_ee_ranges["delta_orn_r"][1], (len(env_ids), 1), device=self.device)
        ee_goal_delta_orn_p = torch_rand_float(self.goal_ee_ranges["delta_orn_p"][0], self.goal_ee_ranges["delta_orn_p"][1], (len(env_ids), 1), device=self.device)
        ee_goal_delta_orn_y = torch_rand_float(self.goal_ee_ranges["delta_orn_y"][0], self.goal_ee_ranges["delta_orn_y"][1], (len(env_ids), 1), device=self.device)
        self.ee_goal_orn_delta_rpy[env_ids, :] = torch.cat([ee_goal_delta_orn_r, ee_goal_delta_orn_p, ee_goal_delta_orn_y], dim=-1)

    def _recenter_ee_goal_sphere(self, env_ids):
        """
        Change the center of the ee_goal_sphere from the arm_base to base, and record the quat
        """
        self.base_quat_fix = self.base_quat.clone()
        ee_goal_cart = sphere2cart(self.ee_goal_sphere[env_ids])
        ee_goal_cart = ee_goal_cart + self.ee_goal_center_offset_stand[env_ids]
        self.ee_goal_sphere[env_ids] = cart2sphere(ee_goal_cart)

    def _compute_ee_start_sphere_from_cur_ee(self, env_ids):
        """Get the spherical coordinate centered at the base of the curr_ee and give that value to ee_start"""
        ee_pose = self.rigid_body_state[env_ids, self.gripper_idx, :3]
        ee_pose_base = quat_rotate_inverse(self.base_quat_fix[env_ids], (ee_pose - self.root_states[env_ids, :3]))
        self.ee_start_sphere[env_ids] = cart2sphere(ee_pose_base)
    
    def _resample_ee_goal(self, env_ids, is_init=False):

        if len(env_ids) > 0:
            init_env_ids = env_ids.clone()
            
            if is_init:
                self.ee_goal_orn_delta_rpy[env_ids, :] = 0
                self.ee_start_sphere[env_ids] = self.init_start_ee_sphere[:]
                self.ee_goal_sphere[env_ids] = self.init_end_ee_sphere[:]
                self._recenter_ee_goal_sphere(env_ids)
                self.is_init = torch.ones(self.num_envs, 1, device=self.device, dtype=torch.bool).squeeze(-1)
            else:
                self.is_init[env_ids] = 0
                self._resample_ee_goal_orn_once(env_ids)
                self.ee_start_sphere[env_ids] = self.ee_goal_sphere[env_ids].clone()
                self._resample_ee_goal_sphere_once_stand(env_ids)
                self._recenter_ee_goal_sphere(env_ids)
                self._compute_ee_start_sphere_from_cur_ee(env_ids)
                # for i in range(10):
                #     self._resample_ee_goal_sphere_once_stand(env_ids)
                #     collision_mask = self.collision_check(env_ids)
                #     env_ids = env_ids[collision_mask]
                #     if len(env_ids) == 0:
                #         break
            self.ee_goal_cart[init_env_ids, :] = sphere2cart(self.ee_goal_sphere[init_env_ids, :])
            self.goal_timer[init_env_ids] = 0.0

        # self.ee_goal_cart[env_ids, :] = sphere2cart(self.ee_goal_sphere[env_ids, :])
        # self.goal_timer[env_ids] = 0.0

    def collision_check(self, env_ids):
        ee_target_all_sphere = torch.lerp(self.ee_start_sphere[env_ids, ..., None], self.ee_goal_sphere[env_ids, ...,  None], self.collision_check_t).squeeze(-1)
        ee_target_cart = sphere2cart(torch.permute(ee_target_all_sphere, (2, 0, 1)).reshape(-1, 3)).reshape(self.num_collision_check_samples, -1, 3)
        collision_mask = torch.any(torch.logical_and(torch.all(ee_target_cart < self.collision_upper_limits, dim=-1), torch.all(ee_target_cart > self.collision_lower_limits, dim=-1)), dim=0)
        underground_mask = torch.any(ee_target_cart[..., 2] < self.underground_limit, dim=0)
        return collision_mask | underground_mask

    def update_curr_ee_goal(self):
        
        t = torch.clip(self.goal_timer / self.traj_timesteps, 0, 1)
        self.curr_ee_goal_sphere[:] = torch.lerp(self.ee_start_sphere, self.ee_goal_sphere, t[:, None])

        self.curr_ee_goal_cart[:] = sphere2cart(self.curr_ee_goal_sphere)
        ee_goal_cart_global = quat_apply(self.base_quat_fix, self.curr_ee_goal_cart)
        self.curr_ee_goal_cart_world = self.root_states[:, :3] + ee_goal_cart_global
        self.final_ee_goal_cart_world = self.root_states[:, :3] + quat_apply(self.base_quat_fix, sphere2cart(self.ee_goal_sphere))
        self.final_ee_start_cart_world = self.root_states[:, :3] + quat_apply(self.base_quat_fix, sphere2cart(self.ee_start_sphere))
        
        # default_yaw = torch.atan2(ee_goal_cart_global[:, 1], ee_goal_cart_global[:, 0])
        # default_pitch = -self.curr_ee_goal_sphere[:, 1] + self.cfg.goal_ee.arm_induced_pitch
        # self.ee_goal_orn_quat = quat_from_euler_xyz(self.ee_goal_orn_delta_rpy[:, 0] + np.pi / 2, default_pitch + self.ee_goal_orn_delta_rpy[:, 1], self.ee_goal_orn_delta_rpy[:, 2] + default_yaw)
        self.ee_goal_orn_euler = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_orn_euler[:, 0] = np.pi / 2
        self.ee_goal_orn_euler[:, 1] = -np.pi / 12
        self.ee_goal_orn_quat = quat_from_euler_xyz(self.ee_goal_orn_euler[:, 0], self.ee_goal_orn_euler[:, 1], self.ee_goal_orn_euler[:, 2])

        self.goal_timer += 1
        resample_id = (self.goal_timer > self.traj_total_timesteps).nonzero(as_tuple=False).flatten()
        self._resample_ee_goal(resample_id)
    
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self._resample_ee_goal(env_ids, is_init=True)
        self.goal_timer[env_ids] = 0.
        if self.cfg.env.history_len > 0:
            self.obs_history_buf[env_ids, :, :] = 0.
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits[:, :12], dim=1)  
    
    def _reward_stand_up_x(self):
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        body_x_in_world = quat_rotate(self.base_quat, self.forward_vec)
        body_x_in_world_z= body_x_in_world[:, 2]
        rew= to_torch(body_x_in_world_z)
        torch.clip(rew, min = -0.7)
        return rew
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, 1:]), dim=1)

    def _reward_flfr_footforce(self):
        feet_force = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        body_x_in_world = quat_rotate(self.base_quat, self.forward_vec)
        body_x_in_world_z= body_x_in_world[:, 2]
        mask = torch.where(body_x_in_world_z > 0.6, 1., 0.3)
        contact = (feet_force[:, 0] + feet_force[:, 1]) > 0.1
        rew = torch.where(contact, 
                          torch.ones(self.num_envs, device=self.device, dtype=torch.float), 
                          torch.zeros(self.num_envs, device=self.device, dtype=torch.float))
        return mask*rew
    
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :1] + self.base_lin_vel[:, 2:3]), dim=1)
        lin_vel_error += torch.sum(torch.square(self.commands[:, 1:2] - self.base_lin_vel[:, 1:2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 0])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 0])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, 1:]), dim=1)
    
    def _reward_arm_action(self):
        # Penalize useless arm action
        return torch.sum(torch.square(self.total_actions[:, -7:]), dim=1)
    
    def _reward_stand_front_feet_shrink(self):
        # Penalize stretching the front feet when standing
        base_pitch = euler_from_quat(self.base_quat)[1]
        front_feet_dofs = self.dof_pos[:, :6]
        front_shrink_dofs = torch.tensor([[0.0, 1.6, -2.6, 0.0, 1.6, -2.6]], device=self.device, dtype=torch.float)
        front_shrink_dofs = front_shrink_dofs.repeat(self.num_envs, 1)
        shrink_error = torch.sum(torch.square(front_feet_dofs - front_shrink_dofs), dim=1)
        torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        # print('base_pitch', base_pitch)
        rew = torch.where(base_pitch < -np.pi/6, 
                          shrink_error, 
                          torch.zeros(self.num_envs, device=self.device, dtype=torch.float))
        return rew
    
    def _reward_tracking_ee_world(self):
        ee_pos_error = torch.sum(torch.abs(self.ee_pos - self.curr_ee_goal_cart_world), dim=1)
        rew = torch.exp(-ee_pos_error/self.cfg.rewards.tracking_ee_sigma * 2)
        return rew