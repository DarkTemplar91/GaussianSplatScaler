from enum import IntEnum, auto

import plyfile
import os.path

import torch

from models.gaussian_model import GaussianModel


class PointCloudType(IntEnum):
    GAUSSIAN = auto()
    INPUT = auto()
    UNKNOWN = auto()


def load_plyfile_pc(pc_path):
    if not os.path.isfile(pc_path):
        return None

    return plyfile.PlyData.read(pc_path)


def load_gaussian_pc(pc_path):
    torch.cuda.empty_cache()
    plyfile_point_cloud = load_plyfile_pc(pc_path)

    if not is_point_cloud_gaussian(plyfile_point_cloud):
        return None

    gaussian_point_cloud = GaussianModel()
    gaussian_point_cloud.from_ply(plyfile_point_cloud)

    torch.cuda.empty_cache()
    return gaussian_point_cloud


def is_point_cloud_gaussian(point_cloud):
    if not point_cloud:
        return False

    props = [p.name for p in point_cloud['vertex'].properties]
    if "f_dc_0" in props:
        return True

    return False
