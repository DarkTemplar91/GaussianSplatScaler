import argparse
import os

import numpy as np
import torch

from utils.colmap_loader import readColmapSceneInfo
from utils.file_loader import load_gaussian_pc
from utils.scaler_utils import load_transformation, check_file


def transform_point_cloud():
    transformation_matrix, scale = load_transformation(transformation_path)
    transformation_full = np.vstack([transformation_matrix, [0, 0, 0, 1]])
    transformation_full = np.linalg.inv(transformation_full)

    print("Loading Gaussian Splat...")
    gaussian_model = load_gaussian_pc(model_path)

    scale_tensor = torch.tensor(scale, device='cuda')
    transformation_tensor = torch.tensor(transformation_full, device='cuda')

    print("Scaling Gaussian Splat...")
    gaussian_model.scale_gaussian_model(scale_tensor)
    print("Transforming Gaussian Splat...")
    gaussian_model.transform_gaussian_model(transformation_tensor)
    print("Saving Gaussian Splat...")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    gaussian_model.save_ply(os.path.join(output_path, "point_cloud_scaled.ply"))
    if cameras_path is None:
        print("Done!")
        return

    print("Creating cameras.json file")
    readColmapSceneInfo(cameras_path, output_path)
    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=lambda f: check_file(f, '.ply'),
                        help='PLY file of the Gaussian Splat', required=True)
    parser.add_argument('--transformation', '-t', type=lambda f: check_file(f, '.json'),
                        help='Path to the dataparser_transforms.json file', required=True)
    parser.add_argument('--output', '-o', help='Output dir', required=True)
    parser.add_argument('--cameras', '-c', help='Path to the cameras and images COLMAP files', required=False)
    args = parser.parse_args()

    model_path = args.model
    transformation_path = args.transformation
    output_path = args.output
    cameras_path = args.cameras

    if os.path.isfile(output_path):
        raise Exception("Output path is a file. Aborting execution!")

    transform_point_cloud()
