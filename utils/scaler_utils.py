import argparse
import json
import os


def load_transformation(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    transformation_matrix = data.get('transform', None)
    scale = data.get('scale', None)

    if transformation_matrix is None:
        raise ValueError("Transformation matrix not found in the JSON file.")
    if scale is None:
        raise ValueError("Scale value not found in the JSON file.")

    return transformation_matrix, scale


def check_file(filepath, extension):
    if not os.path.isfile(filepath):
        raise argparse.ArgumentTypeError(f"File '{filepath}' does not exist.")
    if not filepath.lower().endswith(extension):
        raise argparse.ArgumentTypeError(f"File '{filepath}' must have a '{extension}' extension.")
    return filepath
