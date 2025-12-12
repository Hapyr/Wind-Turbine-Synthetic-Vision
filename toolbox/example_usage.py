import blenderproc as bproc
import os
import sys

import wind_turbine_synthetic_vision.helper as helper
from wind_turbine_synthetic_vision.generator import DatasetGenerator

output_path_root, output_paths = helper.get_output_paths("example", os.path.dirname(__file__))

blender_scene_path = os.path.join(os.path.dirname(__file__), "./scene/example_100_wts.blend")
background_images_path = os.path.join(os.path.dirname(__file__), "background")
config_file_path = os.path.join(os.path.dirname(__file__), "example_config.yaml")

generator = DatasetGenerator(
    number_images=10,
    yolo_scene_file_path_abs=blender_scene_path,
    output_paths=(output_path_root, output_paths),
    background_images_path_abs=background_images_path,
    config_file_path=config_file_path,
)

generator.generate()
