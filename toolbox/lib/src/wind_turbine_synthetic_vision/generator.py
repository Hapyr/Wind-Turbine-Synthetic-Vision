import random
import math
import os
from typing import List, Tuple, Callable

import numpy as np
import blenderproc as bproc
import yaml
import bpy
import bpy_extras
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from . import helper
from .config_validator import validate_config


class DatasetGenerator:
    def __init__(
        self,
        number_images: int,
        yolo_scene_file_path_abs: str,
        output_paths: Tuple[str, helper.OutputPaths],
        background_images_path_abs: str,
        config_file_path: str = None,
        color_codes: List[Tuple[int, int, int]] = None,
    ):
        if config_file_path is None:
            config_file_path = os.path.join(os.path.dirname(__file__), "config.yaml")

        try:
            self.config = validate_config(config_file_path)
            print(f"Configuration loaded and validated from: {config_file_path}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Configuration file not found: {config_file_path}") from e
        except Exception as e:
            raise ValueError(f"Error validating configuration: {e}") from e

        self.PLACEMENT = self.config["PLACEMENT"]
        self.HOUSING_ROTATION = self.config["HOUSING_ROTATION"]
        self.ROTOR_ROTATION = self.config["ROTOR_ROTATION"]
        self.CAMERA = self.config["CAMERA"]
        self.WT_PARAMETERS = self.config["WT_PARAMETERS"]
        self.POSTPROCESSING = self.config["POSTPROCESSING"]
        self.SKY_TEXTURE = self.config["SKY_TEXTURE"]
        self.MATERIAL = self.config["MATERIAL"]

        self.path_scene = yolo_scene_file_path_abs
        self.number_images = number_images
        self.output_path_root, self.output_paths = output_paths
        self.background_images_path_abs = background_images_path_abs
        self.random_background = self.config["RANDOM_BACKGROUND"]

        bproc.renderer.set_render_devices("OPTIX")
        bproc.init()

        bpy.ops.wm.open_mainfile(filepath=self.path_scene)
        bpy.context.view_layer.update()

        bpy.context.scene.cycles.feature_set = "EXPERIMENTAL"
        bpy.context.scene.render.engine = "CYCLES"

        self.camera = bpy.data.objects["Camera"].data

        self.camera.sensor_fit = "VERTICAL"
        self.camera.sensor_height = self.CAMERA["SENSOR_HEIGHT"]

        bproc.camera.set_resolution(self.CAMERA["X_RES"], self.CAMERA["Y_RES"])

        self.color_codes = color_codes or helper.default_color_codes
        self.turbine_map = None

        self.sky_texture_node = None
        for n in bpy.data.worlds["World"].node_tree.nodes:
            if n.name == "Sky Texture":
                self.sky_texture_node = n
        assert self.sky_texture_node is not None, "Sky Texture node not found"

    def generate(self, outputpaths: helper.OutputPaths = None):
        output_paths = outputpaths or self.output_paths

        # save the config file in the output folder
        with open(os.path.join(self.output_path_root, "config.yaml"), "w") as f:
            yaml.dump(self.config, f)

        for i in range(self.number_images):
            print(f"Start image genration {i + 1}/{self.number_images}")
            bproc.utility.reset_keyframes()

            # create folder structure to save the images, keypoints visualizations and the keypoint text file
            path_image, path_keypoints, path_image_keypoints = self.generate_folder_structure(output_paths, i)

            helper.randomize_sky_texture(self.sky_texture_node, self.SKY_TEXTURE)

            self.set_camera(
                height_min=self.CAMERA["MIN_HEIGHT"],
                height_max=self.CAMERA["MAX_HEIGHT"],
                pitch_centered=self.CAMERA["PITCH_CENTERED"],
                angle_bound=self.CAMERA["ANGLE_BOUND"],
                housing_height=self.WT_PARAMETERS["HOUSING_HEIGHT"],
            )

            wea_selection, xy_weas_placed = helper.generate_wind_turbine_set(
                number_of_wts=self.PLACEMENT["NUMBER_OF_WTS"],
                centered_wt=self.PLACEMENT["CENTERED_WT"],
                boundary_left_angle=self.PLACEMENT["BOUNDARY_LEFT_ANGLE"],
                boundary_right_angle=self.PLACEMENT["BOUNDARY_RIGHT_ANGLE"],
                min_distance_between_wts=self.PLACEMENT["MIN_DISTANCE_BETWEEN_WTS"],
                max_distance=self.PLACEMENT["MAX_DISTANCE"],
            )

            if not self.random_background:
                helper.position_objects_in_scene(
                    self.PLACEMENT["BOUNDARY_LEFT_ANGLE"],
                    self.PLACEMENT["BOUNDARY_RIGHT_ANGLE"],
                    1000,
                    xy_weas_placed,
                    self.PLACEMENT["MIN_DISTANCE_BETWEEN_WTS"],
                )

            helper.randomization_material_properties(wea_selection, self.MATERIAL)

            helper.rotate_housing(
                wea_selection,
                mean_angle=self.HOUSING_ROTATION["MEAN"],
                std_deviation=self.HOUSING_ROTATION["STD_DEV"],
                normal_distributed=self.HOUSING_ROTATION["NORMAL_DISTRIBUTED_SET"],
                fixed=self.HOUSING_ROTATION["FIXED"],
                min_angle=self.HOUSING_ROTATION["ROTATION_ANGLE_HOUSING_MIN"],
                max_angle=self.HOUSING_ROTATION["ROTATION_ANGLE_HOUSING_MAX"],
            )

            helper.rotate_rotor(
                wea_selection,
                fixed=self.ROTOR_ROTATION["FIXED"],
                min_angle=self.ROTOR_ROTATION["ROTATION_ANGLE_ROTOR_MIN"],
                max_angle=self.ROTOR_ROTATION["ROTATION_ANGLE_ROTOR_MAX"],
            )

            scaling_factors = helper.scale_shaft(
                wea_selection,
                scale_random=self.WT_PARAMETERS["RANDOMIZE_SHAFT_SCALING_FACTOR"],
                scaling_factor_min=self.WT_PARAMETERS["SHAFT_SCALING_FACTOR_MIN"],
                scaling_factor_max=self.WT_PARAMETERS["SHAFT_SCALING_FACTOR_MAX"],
            )

            # create yolo text file witht the labels
            keypoints = self.create_yolo_text_file(
                path_keypoints,
                wea_selection,
                bpy.context.scene.camera,
                bpy.context.scene,
                scaling_factors,
            )

            if self.random_background:
                # only the components of the front wind turbine are made visible
                for obj in bpy.context.scene.objects:
                    obj.hide_render = True
                for wea in wea_selection:
                    wea.housing.hide_render = False
                    wea.tower.hide_render = False
                    wea.rotor.hide_render = False

            helper.set_category_ids(wea_selection)

            bproc.renderer.set_output_format(enable_transparency=self.random_background)
            bproc.renderer.set_noise_threshold(0.01)

            # render RGB image
            data = bproc.renderer.render()

            # foreground to pil image
            foreground_img = Image.fromarray(data["colors"][0]).convert("RGBA")

            # apply random hue shift to foreground
            if random.random() < self.POSTPROCESSING["HUE_SHIFT"]["THRESHOLD_FORGROUND"]:
                foreground_img = helper.shift_hue_random(foreground_img, self.POSTPROCESSING["HUE_SHIFT"])

            scene_render_image = None
            if self.random_background:
                # randomly select a background image from the background folder
                background_img = os.path.join(
                    self.background_images_path_abs,
                    random.choice(os.listdir(self.background_images_path_abs)),
                )
                background_img = Image.open(background_img).convert("RGBA")
                background_img = background_img.resize((self.CAMERA["X_RES"], self.CAMERA["Y_RES"]))

                # replace the background image with a random noise image
                if random.random() < self.POSTPROCESSING["NOISE_BACKGROUND_THRESHOLD"]:
                    background_img = np.random.uniform(0, 256, (self.CAMERA["X_RES"], self.CAMERA["Y_RES"], 3)).astype(
                        np.uint8
                    )
                    background_img = Image.fromarray(background_img).convert("RGBA")
                    background_img = background_img.resize((self.CAMERA["X_RES"], self.CAMERA["Y_RES"]))

                # apply random hue shift to background
                if random.random() < self.POSTPROCESSING["HUE_SHIFT"]["THRESHOLD_BACKGROUND"]:
                    background_img = helper.shift_hue_random(background_img, self.POSTPROCESSING["HUE_SHIFT"])

                # paste foreground on background
                background_img.paste(foreground_img, mask=foreground_img)
                scene_render_image = np.array(background_img)
            else:
                scene_render_image = np.array(foreground_img)

            assert scene_render_image is not None, "Scene render image is None"

            # add random noise add some random level
            if random.random() < self.POSTPROCESSING["GAUSSIAN_NOISE_ARTEFACT"]["THRESHOLD"]:
                sigma = random.randint(1, self.POSTPROCESSING["GAUSSIAN_NOISE_ARTEFACT"]["SIGMA_MAX"])
                scene_render_image = helper.add_gaussian_noise(scene_render_image, sigma=sigma)

            # save image with low JPEG quality, random compression quality between 1 and 100
            if random.random() < self.POSTPROCESSING["COMPRESSION"]["THRESHOLD"]:
                compression_quality = random.randint(
                    self.POSTPROCESSING["COMPRESSION"]["QUALITY_MIN"],
                    self.POSTPROCESSING["COMPRESSION"]["QUALITY_MAX"],
                )
                _, encoded_img = cv2.imencode(
                    ".jpg",
                    scene_render_image,
                    [cv2.IMWRITE_JPEG_QUALITY, compression_quality],
                )
                scene_render_image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)

            # save the image and the keypoints
            cv2.imwrite(path_image, cv2.cvtColor(scene_render_image, cv2.COLOR_RGBA2BGRA))
            self.draw_keypoints(keypoints, path_image, path_image_keypoints, self.color_codes)

    def generate_from_map(self, outputpaths: helper.OutputPaths = None):
        output_paths = outputpaths or self.output_paths

        # save the config file in the output folder
        with open(os.path.join(self.output_path_root, "config.yaml"), "w") as f:
            yaml.dump(self.config, f)

        for i in range(self.number_images):
            print(f"Start image genration {i + 1}/{self.number_images}")
            bproc.utility.reset_keyframes()

            # create folder structure to save the images, keypoints visualizations and the keypoint text file
            path_image, path_keypoints, path_image_keypoints = self.generate_folder_structure(output_paths, i)

            helper.randomize_sky_texture(self.sky_texture_node, self.SKY_TEXTURE)

            if self.turbine_map is None:
                self.turbine_map = np.array(helper.get_relative_wea_map(54.34, 7.64, 9)) * 500
                # Create scatter plot of turbine positions
                plt.figure(figsize=(8, 8))
                plt.scatter(self.turbine_map[:, 0], self.turbine_map[:, 1], c="blue", alpha=0.6)
                plt.title("Wind Turbine Positions")
                plt.xlabel("X Position (m)")
                plt.ylabel("Y Position (m)")
                plt.grid(True)
                plt.savefig(os.path.join(self.output_path_root, "turbine_positions.png"))
                plt.close()

            wea_selection = helper.generate_wind_turbine_set_from_map(
                centered_wt=self.PLACEMENT["CENTERED_WT"],
                boundary_left_angle=self.PLACEMENT["BOUNDARY_LEFT_ANGLE"],
                boundary_right_angle=self.PLACEMENT["BOUNDARY_RIGHT_ANGLE"],
                min_distance_between_wts=self.PLACEMENT["MIN_DISTANCE_BETWEEN_WTS"],
                max_distance=self.PLACEMENT["MAX_DISTANCE"],
                expnsn_ref=self.PLACEMENT["EXPANSION_REF"],
                min_distance_cam_wt=self.PLACEMENT["MIN_DIST_CAM_WT"],
                turbine_map=self.turbine_map,
            )

            helper.randomization_material_properties(wea_selection, self.MATERIAL)

            helper.rotate_housing(
                wea_selection,
                mean_angle=self.HOUSING_ROTATION["MEAN"],
                std_deviation=self.HOUSING_ROTATION["STD_DEV"],
                normal_distributed=self.HOUSING_ROTATION["NORMAL_DISTRIBUTED_SET"],
                fixed=self.HOUSING_ROTATION["FIXED"],
                min_angle=self.HOUSING_ROTATION["ROTATION_ANGLE_HOUSING_MIN"],
                max_angle=self.HOUSING_ROTATION["ROTATION_ANGLE_HOUSING_MAX"],
            )

            helper.rotate_rotor(
                wea_selection,
                fixed=self.ROTOR_ROTATION["FIXED"],
                min_angle=self.ROTOR_ROTATION["ROTATION_ANGLE_ROTOR_MIN"],
                max_angle=self.ROTOR_ROTATION["ROTATION_ANGLE_ROTOR_MAX"],
            )

            scaling_factors = helper.scale_shaft(
                wea_selection,
                scale_random=self.WT_PARAMETERS["RANDOMIZE_SHAFT_SCALING_FACTOR"],
                scaling_factor_min=self.WT_PARAMETERS["SHAFT_SCALING_FACTOR_MIN"],
                scaling_factor_max=self.WT_PARAMETERS["SHAFT_SCALING_FACTOR_MAX"],
            )

            self.set_camera(
                height_min=self.CAMERA["MIN_HEIGHT"],
                height_max=self.CAMERA["MAX_HEIGHT"],
                pitch_centered=self.CAMERA["PITCH_CENTERED"],
                angle_bound=self.CAMERA["ANGLE_BOUND"],
                housing_height=self.WT_PARAMETERS["HOUSING_HEIGHT"],
            )

            # create yolo text file with the labels
            keypoints = self.create_yolo_text_file(
                path_keypoints,
                wea_selection,
                bpy.context.scene.camera,
                bpy.context.scene,
                scaling_factors,
            )

            if self.random_background:
                # only the components of the front wind turbine are made visible
                for obj in bpy.context.scene.objects:
                    obj.hide_render = True
                for wea in wea_selection:
                    wea.housing.hide_render = False
                    wea.tower.hide_render = False
                    wea.rotor.hide_render = False

            helper.set_category_ids(wea_selection)

            bproc.renderer.set_output_format(enable_transparency=self.random_background)
            bproc.renderer.set_noise_threshold(0.01)

            # render RGB image
            data = bproc.renderer.render()

            # foreground to pil image
            foreground_img = Image.fromarray(data["colors"][0]).convert("RGBA")

            # apply random hue shift to foreground
            if random.random() < self.POSTPROCESSING["HUE_SHIFT"]["THRESHOLD_FORGROUND"]:
                foreground_img = helper.shift_hue_random(foreground_img, self.POSTPROCESSING["HUE_SHIFT"])

            scene_render_image = None
            if self.random_background:
                # randomly select a background image from the background folder
                background_img = os.path.join(
                    self.background_images_path_abs,
                    random.choice(os.listdir(self.background_images_path_abs)),
                )
                background_img = Image.open(background_img).convert("RGBA")
                background_img = background_img.resize((self.CAMERA["X_RES"], self.CAMERA["Y_RES"]))

                # replace the background image with a random noise image
                if random.random() < self.POSTPROCESSING["NOISE_BACKGROUND_THRESHOLD"]:
                    background_img = np.random.uniform(0, 256, (self.CAMERA["X_RES"], self.CAMERA["Y_RES"], 3)).astype(
                        np.uint8
                    )
                    background_img = Image.fromarray(background_img).convert("RGBA")
                    background_img = background_img.resize((self.CAMERA["X_RES"], self.CAMERA["Y_RES"]))

                # apply random hue shift to background
                if random.random() < self.POSTPROCESSING["HUE_SHIFT"]["THRESHOLD_BACKGROUND"]:
                    background_img = helper.shift_hue_random(background_img, self.POSTPROCESSING["HUE_SHIFT"])

                # paste foreground on background
                background_img.paste(foreground_img, mask=foreground_img)
                scene_render_image = np.array(background_img)
            else:
                scene_render_image = np.array(foreground_img)

            assert scene_render_image is not None, "Scene render image is None"

            # add random noise add some random level
            if random.random() < self.POSTPROCESSING["GAUSSIAN_NOISE_ARTEFACT"]["THRESHOLD"]:
                sigma = random.randint(1, self.POSTPROCESSING["GAUSSIAN_NOISE_ARTEFACT"]["SIGMA_MAX"])
                scene_render_image = helper.add_gaussian_noise(scene_render_image, sigma=sigma)

            # save image with low JPEG quality, random compression quality between 1 and 100
            if random.random() < self.POSTPROCESSING["COMPRESSION"]["THRESHOLD"]:
                compression_quality = random.randint(
                    self.POSTPROCESSING["COMPRESSION"]["QUALITY_MIN"],
                    self.POSTPROCESSING["COMPRESSION"]["QUALITY_MAX"],
                )
                _, encoded_img = cv2.imencode(
                    ".jpg",
                    scene_render_image,
                    [cv2.IMWRITE_JPEG_QUALITY, compression_quality],
                )
                scene_render_image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)

            # save the image and the keypoints
            cv2.imwrite(path_image, cv2.cvtColor(scene_render_image, cv2.COLOR_RGBA2BGRA))
            self.draw_keypoints(keypoints, path_image, path_image_keypoints, self.color_codes)

    def set_camera(
        self,
        height_min: int = 55,
        height_max: int = 90,
        pitch_centered: bool = False,
        angle_bound: int = 5,
        housing_height: int = 50,
    ):
        """The distance, height and viewing angle of the camera can be set. The distance of the camera to the wind turbine is along the negative y-axis.
        The height changes the z-axis of the camera. The x-axis is always set to zero.

        Args:
            height_min (int, optional): minimal height of the camera in perspective to the wind turbine. Defaults to 55.
            height_max (int, optional): maximal height of the camera in perspective to the wind turbine. Defaults to 90.
            pitch_centered (bool, optional): whether or not the camera should be aimed at the wind turbine housing. The height of the housing must also be provided. Defaults to False.
            angle_bound (int, optional): if pitch_centered is false, this angle is used as the minimum and maximum limit for all yaw, pitch and roll angles in degree. Defaults to 5.
            housing_height (int, optional): if pitch_centered is true, is used to adjust the camera pitch angle to look at the housing. Defaults to 50.
        """

        if not self.CAMERA["FIXED_CAMERA_LENS"]:
            distance, lens = self.rejection_sample(
                self.CAMERA["CAMERA_DISTANCE_MIN"],
                self.CAMERA["CAMERA_DISTANCE_MAX"],
                3.0,
                55.0,
            )
            self.camera.lens = lens
        else:
            distance = random.uniform(
                self.CAMERA["CAMERA_DISTANCE_MIN"],
                self.CAMERA["CAMERA_DISTANCE_MAX"],
            )
            self.camera.lens = self.CAMERA["LENS_MM"]

        height = random.uniform(height_min, height_max)

        if pitch_centered:
            delta = height - housing_height
            angle_x = math.pi / 2 - math.atan(delta / distance)  # pitch angle of the camera
            angle_y = math.radians(np.random.normal(loc=0.0, scale=3)) + math.radians(
                random.uniform(-angle_bound, angle_bound)
            )  # roll angle of the camera
            angle_z = math.radians(random.uniform(-angle_bound, angle_bound))  # yaw angle of the camera rotation
        else:
            angle_x = math.pi / 2 + math.radians(random.uniform(-angle_bound, angle_bound))
            angle_y = math.radians(random.uniform(-angle_bound, angle_bound))
            angle_z = math.radians(random.uniform(-angle_bound, angle_bound))

        position = [0, -distance, height]
        euler_rotation = [angle_x, angle_y, angle_z]

        matrix_world = bproc.math.build_transformation_mat(position, euler_rotation)
        bproc.camera.add_camera_pose(matrix_world)

        return position, euler_rotation

    def draw_keypoints(
        self,
        keypoints: List[List[float]],
        input_image_path: str,
        output_image_path: str,
        color_codes: List[Tuple[int, int, int]] = None,
    ):
        """Draw the keypoints on the image for visualization only.

        Args:
            keypoints (List[List[float]]): The keypoints to draw.
            input_image_path (str): The path to the input image.
            output_image_path (str): The path to the output image.
            color_codes (List[Tuple[int, int, int]], optional): The color codes to use for the keypoints. Defaults to None.
        """
        image_new = cv2.imread(input_image_path)

        colors = color_codes or [
            (255, 0, 0),  # Red - shaft bottom
            (0, 255, 0),  # Green - shaft top
            (0, 0, 255),  # Blue - housing back
            (255, 255, 0),  # Yellow - rotor middle
            (255, 0, 255),  # Magenta - rotor tip 1 (top)
            (0, 255, 255),  # Cyan - rotor tip 2 (right)
            (255, 128, 0),  # Orange - rotor tip 3 (left)
        ]

        for keypoint in keypoints:
            pos_start = (
                int(self.CAMERA["X_RES"] * (keypoint[1] - keypoint[3] / 2)),
                int(self.CAMERA["Y_RES"] * (keypoint[2] - keypoint[4] / 2)),
            )
            pos_end = (
                int(pos_start[0] + keypoint[3] * self.CAMERA["X_RES"]),
                int(pos_start[1] + keypoint[4] * self.CAMERA["Y_RES"]),
            )

            image_new = cv2.rectangle(image_new, pos_start, pos_end, (255, 0, 0), 1)

            for i in range(int(len(keypoint[5:]) / 2)):
                image_new = cv2.circle(
                    image_new,
                    (
                        int(keypoint[5 + 2 * i] * self.CAMERA["X_RES"]),
                        int(keypoint[6 + 2 * i] * self.CAMERA["Y_RES"]),
                    ),
                    2,
                    colors[i],
                    -1,
                )

        cv2.imwrite(output_image_path, image_new)

    def rejection_sample(self, d_min: float, d_max: float, l_min: float, l_max: float):
        """Rejection sampling for camera distance and sensor lens.
        The distance and lens are sampled from a uniform distribution.
        The lens is then checked if it is too large for the distance, which mean that that elements of the wind turbine is outside the image.
        If it is, the function is called again, until the lens is in the acceptable range.
        This means that the distance and the aperture are equally distributed within the valid range.

        Args:
            d_min (float): Minimal distance of the camera to the wind turbine.
            d_max (float): Maximal distance of the camera to the wind turbine.
            l_min (float): Minimal lens of the camera.
            l_max (float): Maximal lens of the camera.
        """
        distance = np.random.uniform(d_min, d_max)
        lens = np.random.uniform(l_min, l_max)

        l_max_deala = (abs(distance) / self.WT_PARAMETERS["TURBINE_RADIUS"]) * (self.CAMERA["SENSOR_HEIGHT"] / 2.0)

        if lens > l_max_deala:
            return self.rejection_sample(d_min, d_max, l_min, l_max)

        return distance, lens

    def generate_folder_structure(self, output_paths: helper.OutputPaths, index: int):
        """Generate the folder structure for the wind turbine synthetic vision dataset.
        Implement training/test split and set output path for the image and keypoints.

        Args:
            output_path (str): The path to the output folder.
        """

        if random.random() < self.POSTPROCESSING["TRAIN_VALID_RATIO"]:
            path_image = os.path.join(output_paths["training"]["path_images"], f"{index:05d}.png")
            path_keypoints = os.path.join(output_paths["training"]["path_keypoints"], f"{index:05d}.txt")
            path_image_keypoints = os.path.join(output_paths["training"]["path_images_keypoints"], f"{index:05d}.png")
        else:
            path_image = os.path.join(output_paths["validation"]["path_images"], f"{index:05d}.png")
            path_keypoints = os.path.join(output_paths["validation"]["path_keypoints"], f"{index:05d}.txt")
            path_image_keypoints = os.path.join(output_paths["validation"]["path_images_keypoints"], f"{index:05d}.png")

        return path_image, path_keypoints, path_image_keypoints

    def create_yolo_text_file(
        self,
        output_path,
        wea_selection,
        camera,
        scene,
        rotor_scaling_factors: List[float] = None,
    ):
        """Create a YOLO text file for the wind turbine.

        Args:
            output_path (str): The path to the output file.
            wea_selection (list): The list of WEAs to be included in the text file.
            camera (dict): The camera parameters.
            scene (dict): The scene parameters.
            rotor_scaling_factors (list, optional): The scaling factors for the rotors. Defaults to None.

        """
        return_keypoints = []

        if rotor_scaling_factors is None:
            rotor_scaling_factors = [1.0] * len(wea_selection)

        if len(rotor_scaling_factors) != len(wea_selection):
            raise ValueError(
                f"The number of rotor scaling factors must be equal to the number of WEAs. scaling factors: {len(rotor_scaling_factors)}, WEAs: {len(wea_selection)}"
            )

        with open(output_path, "a") as file:
            bpy.context.view_layer.update()

            for wea, rotor_scaling_factor in zip(wea_selection, rotor_scaling_factors):
                key_points_all = []

                key_points_all.append(0)
                Werte = []
                x_Werte = []
                y_Werte = []

                co_2d_housing_back = bpy_extras.object_utils.world_to_camera_view(
                    scene, camera, wea.kp_housing_back.matrix_world.translation
                )
                co_2d_tower_top = bpy_extras.object_utils.world_to_camera_view(
                    scene, camera, wea.kp_tower_top.matrix_world.translation
                )
                co_2d_tower_bottom = bpy_extras.object_utils.world_to_camera_view(
                    scene, camera, wea.kp_tower_bottom.matrix_world.translation
                )
                co_2d_rotor_middle = bpy_extras.object_utils.world_to_camera_view(
                    scene, camera, wea.kp_housing_front.matrix_world.translation
                )
                co_2d_rotor_tip_1 = bpy_extras.object_utils.world_to_camera_view(
                    scene, camera, wea.kp_tip_1.matrix_world.translation
                )
                co_2d_rotor_tip_2 = bpy_extras.object_utils.world_to_camera_view(
                    scene, camera, wea.kp_tip_2.matrix_world.translation
                )
                co_2d_rotor_tip_3 = bpy_extras.object_utils.world_to_camera_view(
                    scene, camera, wea.kp_tip_3.matrix_world.translation
                )

                render_size = (
                    bpy.context.scene.render.resolution_x,
                    bpy.context.scene.render.resolution_y,
                )

                def f_0_0(m_x, m_y, x):
                    return x * m_y / m_x

                def f_0_1(m_x, m_y, x):
                    return (-x * (1 - m_y) / m_x) + 1

                def f_1_0(m_x, m_y, x):
                    return x * m_y / (m_x - 1) + m_y / (1 - m_x)

                def f_1_1(m_x, m_y, x):
                    return (x * (1 - m_y) / (1 - m_x)) + (m_y - m_x) / (1 - m_x)

                def compare_tip(rotor_middle, rotor_tip):
                    c_0_0 = f_0_0(
                        rotor_middle[0],
                        rotor_middle[1],
                        rotor_tip[0],
                    )
                    c_0_1 = f_0_1(
                        rotor_middle[0],
                        rotor_middle[1],
                        rotor_tip[0],
                    )
                    c_1_0 = f_1_0(
                        rotor_middle[0],
                        rotor_middle[1],
                        rotor_tip[0],
                    )
                    c_1_1 = f_1_1(
                        rotor_middle[0],
                        rotor_middle[1],
                        rotor_tip[0],
                    )
                    x_res = 0
                    y_res = 0

                    if c_0_0 > rotor_tip[1] and c_1_0 > rotor_tip[1] and rotor_tip[1] < 0:
                        y_res = 0
                        x_res = rotor_middle[0] - rotor_middle[1] * (rotor_middle[0] - rotor_tip[0]) / (
                            rotor_middle[1] - rotor_tip[1]
                        )
                    elif c_0_1 < rotor_tip[1] and c_1_1 < rotor_tip[1]:
                        y_res = 1
                        x_res = rotor_middle[0] + (1 - rotor_middle[1]) * (rotor_tip[0] - rotor_middle[0]) / (
                            rotor_tip[1] - rotor_middle[1]
                        )
                    elif c_0_0 < rotor_tip[1] and c_0_1 > rotor_tip[1]:
                        y_res = rotor_middle[1] - rotor_middle[0] * (rotor_tip[1] - rotor_middle[1]) / (
                            rotor_tip[0] - rotor_middle[0]
                        )
                        x_res = 0
                    elif c_1_0 < rotor_tip[1] and c_1_1 > rotor_tip[1]:
                        y_res = rotor_middle[1] + (1 - rotor_middle[0]) * (rotor_tip[1] - rotor_middle[1]) / (
                            rotor_tip[0] - rotor_middle[0]
                        )
                        x_res = 1

                    return x_res, y_res

                if co_2d_tower_bottom[1] < 0:
                    scaler = co_2d_tower_bottom[1] / (co_2d_tower_top[1] - co_2d_tower_bottom[1])
                    co_2d_tower_bottom[1] = 0
                    co_2d_tower_bottom[0] = co_2d_tower_bottom[0] - scaler * (
                        co_2d_tower_top[0] - co_2d_tower_bottom[0]
                    )

                if (
                    co_2d_rotor_tip_1[0] < 0
                    or co_2d_rotor_tip_1[0] > 1
                    or co_2d_rotor_tip_1[1] < 0
                    or co_2d_rotor_tip_1[1] > 1
                ):
                    co_2d_rotor_tip_1[0], co_2d_rotor_tip_1[1] = compare_tip(co_2d_rotor_middle, co_2d_rotor_tip_1)

                if (
                    co_2d_rotor_tip_2[0] < 0
                    or co_2d_rotor_tip_2[0] > 1
                    or co_2d_rotor_tip_2[1] < 0
                    or co_2d_rotor_tip_2[1] > 1
                ):
                    co_2d_rotor_tip_2[0], co_2d_rotor_tip_2[1] = compare_tip(co_2d_rotor_middle, co_2d_rotor_tip_2)

                if (
                    co_2d_rotor_tip_3[0] < 0
                    or co_2d_rotor_tip_3[0] > 1
                    or co_2d_rotor_tip_3[1] < 0
                    or co_2d_rotor_tip_3[1] > 1
                ):
                    co_2d_rotor_tip_3[0], co_2d_rotor_tip_3[1] = compare_tip(co_2d_rotor_middle, co_2d_rotor_tip_3)

                # create a list of rotor tips and shuffle them
                rotor_tips = [
                    co_2d_rotor_tip_1,
                    co_2d_rotor_tip_2,
                    co_2d_rotor_tip_3,
                ]
                random.shuffle(rotor_tips)

                co_2d_all = [
                    co_2d_housing_back,
                    co_2d_tower_top,
                    co_2d_tower_bottom,
                    co_2d_rotor_middle,
                    *rotor_tips,  # unpack the shuffled rotor tips
                ]
                for co_2d in co_2d_all:
                    pixel_x = round(co_2d[0] * (render_size[0] - 1))
                    pixel_y = round((1 - co_2d[1]) * (render_size[1] - 1))

                    if pixel_x >= self.CAMERA["X_RES"] or pixel_x < 0 or pixel_y >= self.CAMERA["Y_RES"] or pixel_y < 0:
                        Werte.append(0)
                        Werte.append(0)
                    else:
                        Werte.append(pixel_x / self.CAMERA["X_RES"])
                        Werte.append(pixel_y / self.CAMERA["Y_RES"])

                    if pixel_x < 0 and pixel_y < 0:
                        x_Werte.append(0)
                        y_Werte.append(0)

                    elif pixel_x < 0 and 0 <= pixel_y <= self.CAMERA["Y_RES"]:
                        x_Werte.append(0)
                        y_Werte.append(pixel_y / self.CAMERA["Y_RES"])

                    elif pixel_x < 0 and pixel_y > self.CAMERA["Y_RES"]:
                        x_Werte.append(0)
                        y_Werte.append(1)

                    elif 0 <= pixel_x <= self.CAMERA["X_RES"] and pixel_y < 0:
                        x_Werte.append(pixel_x / self.CAMERA["X_RES"])
                        y_Werte.append(0)

                    elif 0 <= pixel_x <= self.CAMERA["X_RES"] and pixel_y > self.CAMERA["Y_RES"]:
                        x_Werte.append(pixel_x / self.CAMERA["X_RES"])
                        y_Werte.append(1)

                    elif 0 <= pixel_x <= self.CAMERA["X_RES"] and 0 <= pixel_y <= self.CAMERA["Y_RES"]:
                        x_Werte.append(pixel_x / self.CAMERA["X_RES"])
                        y_Werte.append(pixel_y / self.CAMERA["Y_RES"])

                    elif pixel_x > self.CAMERA["X_RES"] and 0 <= pixel_y <= self.CAMERA["Y_RES"]:
                        x_Werte.append(1)
                        y_Werte.append(pixel_y / self.CAMERA["Y_RES"])

                    elif pixel_x > self.CAMERA["X_RES"] and pixel_y > self.CAMERA["Y_RES"]:
                        x_Werte.append(1)
                        y_Werte.append(1)

                    elif pixel_x > self.CAMERA["X_RES"] and pixel_y < 0:
                        x_Werte.append(1)
                        y_Werte.append(0)

                max_x = max(x_Werte)
                min_x = min(x_Werte)
                max_y = max(y_Werte)
                min_y = min(y_Werte)

                mid_x = (max_x + min_x) / 2
                mid_y = (max_y + min_y) / 2
                height = max_y - min_y
                width = max_x - min_x

                # only write the bounding box is within the image
                if (
                    mid_x > 0
                    and mid_x < 1
                    and mid_y > 0
                    and mid_y < 1
                    and width > 0
                    and width < 1
                    and height > 0
                    and height < 1
                ):
                    file.write(f"0 {mid_x} {mid_y} {width} {height} ")
                    key_points_all.append(mid_x)
                    key_points_all.append(mid_y)
                    key_points_all.append(width)
                    key_points_all.append(height)

                    for Wert in Werte:
                        file.write(f"{Wert} ")
                        key_points_all.append(Wert)

                    file.write("\n")
                    return_keypoints.append(key_points_all)

            return return_keypoints
