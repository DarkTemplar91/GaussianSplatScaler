#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
import torch
from plyfile import PlyElement, PlyData

from utils.general_utils import build_scaling_rotation, strip_symmetric, \
    inverse_sigmoid, matrices_to_quaternions


class GaussianModel:

    def __init__(self, sh_degree: int = 3):
        self.sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symmetric = strip_symmetric(actual_covariance)
            return symmetric

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    def from_ply(self, plydata):
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coefficients) to (P, F, SH_coefficients except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = torch.tensor(xyz, dtype=torch.double, device='cuda')
        self._features_dc = (torch.tensor(features_dc, dtype=torch.double, device='cuda')
                             .transpose(1, 2).contiguous())
        self._features_rest = (torch.tensor(features_extra, dtype=torch.double, device='cuda')
                               .transpose(1, 2).contiguous())
        self._opacity = torch.tensor(opacities, dtype=torch.double, device='cuda')
        self._scaling = torch.tensor(scales, dtype=torch.double, device='cuda')
        self._rotation = torch.tensor(rots, dtype=torch.double, device='cuda')

    def construct_list_of_attributes(self):
        attribute_list = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            attribute_list.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            attribute_list.append('f_rest_{}'.format(i))
        attribute_list.append('opacity')
        for i in range(self._scaling.shape[1]):
            attribute_list.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            attribute_list.append('rot_{}'.format(i))
        return attribute_list

    def save_ply(self, path):
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        plydata = PlyData(text=False,
                          byte_order='<',
                          elements=[el])
        PlyData.write(plydata, path)

    def transform_gaussian_model(self, transformation_matrix):
        rotation_matrix = transformation_matrix[:3, :3]
        self._xyz = torch.matmul(self._xyz, rotation_matrix.T)
        self._xyz += transformation_matrix[:3, 3]

        # From gaussian-splatting-lightning
        def quat_multiply(quaternion0, quaternion1):
            w0, x0, y0, z0 = torch.chunk(quaternion0, 4, dim=-1)
            w1, x1, y1, z1 = torch.chunk(quaternion1, 4, dim=-1)

            return torch.cat((
                -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0
            ), dim=-1)

        quaternions = matrices_to_quaternions(rotation_matrix).unsqueeze(0).to('cuda')
        rotations_from_quats = quat_multiply(self._rotation, quaternions)
        self._rotation = rotations_from_quats / torch.norm(rotations_from_quats, p=2, dim=-1, keepdim=True)

    def scale_gaussian_model(self, scale):
        centroid = torch.mean(self._xyz, dim=0)
        translated_points = self._xyz - centroid
        scale_factor = 1 / scale
        scaled_points = translated_points * scale_factor
        self._xyz = scaled_points - centroid
        self._scaling += self.scaling_inverse_activation(scale_factor)
