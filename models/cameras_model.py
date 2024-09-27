import json

import numpy as np
import torch


class CamerasModel:
    def __init__(self):
        self.ids = []
        self.img_names = []
        self.widths = []
        self.heights = []
        self.fx = []
        self.fy = []
        self.position = torch.empty(0)
        self.rotations = torch.empty(0)

    # Function to load the cameras from a JSON file
    def load_from_json(self, json_file):
        with open(json_file, 'r') as f:
            camera_data = json.load(f)

        positions_list = []
        rotations_list = []

        for frame in camera_data:
            self.ids.append(frame['id'])
            self.img_names.append(frame['img_name'])
            self.widths.append(frame['width'])
            self.heights.append(frame['height'])
            self.fx.append(frame['fx'])
            self.fy.append(frame['fy'])

            position_tensor = torch.tensor(frame['position'])
            rotation_tensor = torch.tensor(frame['rotation'])

            positions_list.append(position_tensor)
            rotations_list.append(rotation_tensor)

        self.position = torch.tensor(np.array(positions_list), dtype=torch.double, device='cuda')
        self.rotations = torch.tensor(np.array(rotations_list), dtype=torch.double, device='cuda')

    def get_camera(self, idx):
        return {
            'id': self.ids[idx],
            'img_name': self.img_names[idx],
            'width': self.widths[idx],
            'height': self.heights[idx],
            'fx': self.fx[idx],
            'fy': self.fy[idx],
            'position': self.position[idx],
            'rotation': self.rotations[idx]
        }

    def print_cameras(self):
        for idx in range(len(self.ids)):
            print(f"Camera {idx}:")
            print(f"  id: {self.ids[idx]}")
            print(f"  img_name: {self.img_names[idx]}")
            print(f"  width: {self.widths[idx]}")
            print(f"  height: {self.heights[idx]}")
            print(f"  fx: {self.fx[idx]}")
            print(f"  fy: {self.fy[idx]}")
            print(f"  position: {self.position[idx]}")
            print(f"  rotation: {self.rotations[idx]}")
            print()

    def save_to_json(self, json_file):
        cameras_data = []
        for idx in range(len(self.ids)):
            camera_frame = {
                'id': self.ids[idx],
                'img_name': self.img_names[idx],
                'width': self.widths[idx],
                'height': self.heights[idx],
                'position': self.position[idx].tolist(),
                'rotation': self.rotations[idx].tolist(),
                'fx': self.fx[idx],
                'fy': self.fy[idx],
            }
            cameras_data.append(camera_frame)

        with open(json_file, 'w') as f:
            json.dump(cameras_data, f, indent=4)

    def transform_cameras_model(self, transformation_matrix, scale):
        rotation_matrix = transformation_matrix[:3, :3]
        translate = transformation_matrix[:3, 3]

        centroid = torch.mean(self.position, dim=0)
        translated_points = self.position - centroid
        scaled_points = translated_points * scale
        self.position = scaled_points - centroid
        self.position = torch.matmul(self.position, rotation_matrix.T)
        self.position += translate
        self.rotations = torch.matmul(rotation_matrix, self.rotations)
