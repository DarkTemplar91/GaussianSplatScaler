import argparse
import os

import numpy as np
import torch

from models.cameras_model import CamerasModel
from utils.scaler_utils import load_transformation


def transform_cameras():
    cameras_model = CamerasModel()
    cameras_model.load_from_json(input_path)

    transformation_matrix, scale = load_transformation(transformation_path)
    transformation_full = np.vstack([transformation_matrix, [0, 0, 0, 1]])

    scale_tensor = torch.tensor(scale, device='cuda', dtype=torch.double)
    transformation_tensor = torch.tensor(transformation_full, device='cuda', dtype=torch.double)

    cameras_model.transform_cameras_model(transformation_tensor, scale_tensor)
    cameras_model.save_to_json(output_path)


def check_file(filepath, extension):
    if not os.path.isfile(filepath):
        raise argparse.ArgumentTypeError(f"File '{filepath}' does not exist.")
    if not filepath.lower().endswith(extension):
        raise argparse.ArgumentTypeError(f"File '{filepath}' must have a '{extension}' extension.")
    return filepath


# Main code
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=lambda f: check_file(f, '.json'),
                        help='Input cameras file', required=True)
    parser.add_argument('--transformation', '-t', type=lambda f: check_file(f, '.json'),
                        help='Path to the dataparser_transforms.json file', required=True)
    parser.add_argument('--output', '-o', help='output cameras file', required=True)
    args = parser.parse_args()

    input_path = args.input
    transformation_path = args.transformation
    output_path = args.output

    transform_cameras()
