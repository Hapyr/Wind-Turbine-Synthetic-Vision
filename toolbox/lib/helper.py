import bpy
import random
import math
import numpy as np
from PIL import Image
import os
import datetime
from typing import TypedDict, NamedTuple, List
import mathutils
import bpy_types
import requests
import json


default_color_codes = [
    (255, 0, 0),  # Red - shaft bottom
    (0, 255, 0),  # Green - shaft top
    (0, 0, 255),  # Blue - housing back
    (255, 255, 0),  # Yellow - rotor middle
    (255, 0, 255),  # Magenta - rotor tip 1 (top)
    (0, 255, 255),  # Cyan - rotor tip 2 (right)
    (255, 128, 0),  # Orange - rotor tip 3 (left)
]


class OutputPathsDict(TypedDict):
    """Directory paths for a single dataset split (training, visualization, validation)."""

    path_images: str
    path_images_keypoints: str
    path_keypoints: str


class OutputPaths(TypedDict):
    """Complete directory structure for training and validation datasets."""

    training: OutputPathsDict
    validation: OutputPathsDict


class WTObjectSet(NamedTuple):
    """A collection of objects representing a wind turbine."""

    obj_all: bpy_types.Collection
    tower: bpy_types.Object
    housing: bpy_types.Object
    rotor: bpy_types.Object
    kp_housing_back: bpy_types.Object
    kp_housing_front: bpy_types.Object
    kp_tower_top: bpy_types.Object
    kp_tower_bottom: bpy_types.Object
    kp_tip_1: bpy_types.Object
    kp_tip_2: bpy_types.Object
    kp_tip_3: bpy_types.Object

    def set_xy_position(self, x: float, y: float):
        """Set the x and y coordinates of the wind turbine.

        Args:
            x (float): The x coordinate of the wind turbine.
            y (float): The y coordinate of the wind turbine.
        """
        self.tower.location[0] = x
        self.tower.location[1] = y


def sample_distribution(distribution: str, param_1: float, param_2: float):
    """Sample a value from a specified distribution.

    Args:
        distribution: Distribution type ("normal" or "uniform").
        param_1: For normal: mean, for uniform: lower bound.
        param_2: For normal: standard deviation, for uniform: upper bound.

    Returns:
        float: Sampled value from the distribution.

    Raises:
        ValueError: If distribution type is not supported.
    """
    if distribution == "normal":
        return random.normalvariate(param_1, param_2)
    elif distribution == "uniform":
        return random.uniform(param_1, param_2)
    else:
        raise ValueError(f"Invalid distribution: {distribution}")


def set_category_ids(wea_selection: List[WTObjectSet]):
    """
    Set the category_id property for the objects in the wea_selection list to 1 and for all other objects to 0.

    Args:
        wea_selection (List[WTObjectSet]): List of WTObjectSet objects.
    """
    prop_name = "category_id"

    for obj in bpy.data.objects:
        obj[prop_name] = 0

    for obj in wea_selection:
        obj.tower[prop_name] = 1
        obj.housing[prop_name] = 1
        obj.rotor[prop_name] = 1

    bpy.context.scene.world[prop_name] = 0


def randomize_sky_texture(sky_texture: bpy.types.ShaderNodeTexSky, config: dict):
    """Randomize sky texture parameters based on configuration.

    Args:
        sky_texture: Blender sky texture node to modify.
        config: Dictionary containing distribution parameters for each sky property.
    """

    sky_texture.sky_type = config["TYPE"]
    sky_texture.sun_disc = True

    sky_texture.sun_intensity = np.clip(
        sample_distribution(
            config["SUN_INTENSITY"]["DISTRIBUTION"],
            config["SUN_INTENSITY"]["ARG_1"],
            config["SUN_INTENSITY"]["ARG_2"],
        ),
        0.0,
        1000.0,
    )
    sky_texture.sun_size = np.clip(
        sample_distribution(
            config["SUN_SIZE"]["DISTRIBUTION"],
            config["SUN_SIZE"]["ARG_1"],
            config["SUN_SIZE"]["ARG_2"],
        ),
        0.0,
        1.5708,
    )
    sky_texture.sun_elevation = sample_distribution(
        config["SUN_ELEVATION"]["DISTRIBUTION"],
        config["SUN_ELEVATION"]["ARG_1"],
        config["SUN_ELEVATION"]["ARG_2"],
    )
    sky_texture.sun_rotation = sample_distribution(
        config["SUN_ROTATION"]["DISTRIBUTION"],
        config["SUN_ROTATION"]["ARG_1"],
        config["SUN_ROTATION"]["ARG_2"],
    )
    sky_texture.air_density = np.clip(
        sample_distribution(
            config["AIR_DENSITY"]["DISTRIBUTION"],
            config["AIR_DENSITY"]["ARG_1"],
            config["AIR_DENSITY"]["ARG_2"],
        ),
        0.0,
        10.0,
    )
    sky_texture.dust_density = np.clip(
        sample_distribution(
            config["DUST_DENSITY"]["DISTRIBUTION"],
            config["DUST_DENSITY"]["ARG_1"],
            config["DUST_DENSITY"]["ARG_2"],
        ),
        0.0,
        10.0,
    )
    sky_texture.ozone_density = np.clip(
        sample_distribution(
            config["OZONE_DENSITY"]["DISTRIBUTION"],
            config["OZONE_DENSITY"]["ARG_1"],
            config["OZONE_DENSITY"]["ARG_2"],
        ),
        0.0,
        10.0,
    )
    sky_texture.turbidity = np.clip(
        sample_distribution(
            config["TURBIDITY"]["DISTRIBUTION"],
            config["TURBIDITY"]["ARG_1"],
            config["TURBIDITY"]["ARG_2"],
        ),
        0.0,
        10.0,
    )
    sky_texture.altitude = np.clip(
        sample_distribution(
            config["SUN_ALTITUDE"]["DISTRIBUTION"],
            config["SUN_ALTITUDE"]["ARG_1"],
            config["SUN_ALTITUDE"]["ARG_2"],
        ),
        0.0,
        10000.0,
    )


def shift_hue_random(
    image,
    config: dict,
):
    """Shift the hue of a PIL RGBA image.
    Args:
        image (PIL.Image.Image): The input image.
    Returns:
        PIL.Image.Image: The shifted image.
    """

    # Extract alpha channel from original RGBA image before converting to HSV
    r, g, b, a = image.split()

    # Convert RGB channels to HSV (without alpha)
    rgb_image = Image.merge("RGB", (r, g, b))
    hsv_image = rgb_image.convert("HSV")
    h, s, v = hsv_image.split()

    hue_shift = sample_distribution(
        config["HUE_SHIFT"]["DISTRIBUTION"],
        config["HUE_SHIFT"]["ARG_1"],
        config["HUE_SHIFT"]["ARG_2"],
    )
    sat_shift = sample_distribution(
        config["SATURATION_SHIFT"]["DISTRIBUTION"],
        config["SATURATION_SHIFT"]["ARG_1"],
        config["SATURATION_SHIFT"]["ARG_2"],
    )
    val_shift = sample_distribution(
        config["VALUE_SHIFT"]["DISTRIBUTION"],
        config["VALUE_SHIFT"]["ARG_1"],
        config["VALUE_SHIFT"]["ARG_2"],
    )

    hue_shift = hue_shift * 255 / 360
    sat_shift = sat_shift * 255 / 100
    val_shift = val_shift * 255 / 100

    # Apply adjustments with numpy
    h_np = np.clip(np.array(h) + hue_shift, 0, 255)  # Hue clamping
    s_np = np.clip(np.array(s) + sat_shift, 0, 255)  # Saturation clamping
    v_np = np.clip(np.array(v) + val_shift, 0, 255)  # Value clamping

    # Convert back to PIL Images
    h_new = Image.fromarray(h_np.astype(np.uint8), "L")
    s_new = Image.fromarray(s_np.astype(np.uint8), "L")
    v_new = Image.fromarray(v_np.astype(np.uint8), "L")

    # Rebuild and convert back to RGB
    hsv_adj = Image.merge("HSV", (h_new, s_new, v_new))
    rgb_adj = hsv_adj.convert("RGB")

    # Split new RGB and merge with original Alpha
    r_new, g_new, b_new = rgb_adj.split()
    return Image.merge("RGBA", (r_new, g_new, b_new, a))


def rotate_rotor(
    wea_selection: List[WTObjectSet],
    fixed: bool = False,
    min_angle: int = 0,
    max_angle: int = 119,
):
    """Rotate the rotor of the wind turbines (WEAs) from the given list of WTObjectSet objects.

    Args:
        wea_selection (List[WTObjectSet]): List of WTObjectSet objects.
        fixed (bool, optional): If True, the rotor is not rotated and set to the min_angle. Defaults to False.
        min_angle (int, optional): smallest angle that can be sampled from the distribution in degrees. Defaults to 0.
        max_angle (int, optional): largest angle that can be sampled from the distribution in degrees. Defaults to 119.
    """
    for wea in wea_selection:
        if not fixed:
            angle = random.uniform(min_angle, max_angle)
        else:
            angle = min_angle
        wea.rotor.rotation_euler[1] = math.radians(angle)


def scale_shaft(
    wea_selection: List[WTObjectSet],
    scale_random: bool = True,
    scaling_factor_min: float = 0.0,
    scaling_factor_max: float = 0.0,
):
    """Scale the shaft of the wind turbines (WEAs) from the given list of WTObjectSet objects.
    This allows to have different proportions between the blades and the height of the wind turbine.

    Args:
        wea_selection (List[WTObjectSet]): List of WTObjectSet objects.
        scale_random (bool, optional): If True, the shaft is scaled randomly. Defaults to True.
        factor (float, optional): The factor by which the shaft is scaled when scale_random is False. Defaults to 0.0.

    Returns:
        List[float]: List of scaling factors.
    """
    scaling_factors = []
    for wea in wea_selection:
        if scale_random:
            factor = random.uniform(scaling_factor_min, scaling_factor_max)
            wea.rotor.delta_scale = mathutils.Vector((1.0 * factor, 1.0 * factor, 1.0))
        else:
            factor = scaling_factor_min
            wea.rotor.delta_scale = mathutils.Vector((1.0 * factor, 1.0 * factor, 1.0))
        scaling_factors.append(factor)

    return scaling_factors


def rotate_housing(
    wea_selection: List[WTObjectSet],
    mean_angle: float = 0.0,
    std_deviation: float = 15.0,
    normal_distributed: bool = True,
    fixed: bool = False,
    min_angle: float = 0.0,
    max_angle: float = 360.0,
):
    """Rotate the housing of the wind turbine (WEA) from the given list of WTObjectSet objects.
    The normal distribution for the rotation can be useful if there are several wind turbines on the scene and they should all point in the wind direction,
    with slight deviations.
    With fixed=True, the function will set the angle to the mean_angle. It can still be normally distributed around this fixed angle.

    Args:
        wea_selection (List[WTObjectSet]): List of WTObjectSet objects.
        mean_angle (int, optional): When normal_distributed this is the mean of the distribution. Defaults to 0.
        std_deviation (_type_, optional): When normal_distributed this is the standard deviation of the distribution. Defaults to 45/3.
        normal_distributed (bool, optional): when false, the function rotate each object uniform randomly. Defaults to True.
        fixed (bool, optional): If True, the function will not rotate the objects. Defaults to False.
        min_angle (int, optional): Minimum angle of the distribution. Defaults to 0.
        max_angle (int, optional): Maximum angle of the distribution. Defaults to 360.
    """

    if not fixed:
        mean_angle = random.uniform(min_angle, max_angle)

    for wea in wea_selection:
        if normal_distributed:
            angle = random.gauss(mean_angle, std_deviation)
        else:
            if not fixed:
                angle = random.uniform(min_angle, max_angle)
            else:
                angle = mean_angle
        wea.housing.rotation_euler[2] = math.radians(angle)


def generate_wind_turbine_set_from_map(
    centered_wt: bool = True,
    boundary_left_angle: int = 0,
    boundary_right_angle: int = 0,
    min_distance_between_wts: int = 60,
    max_distance: int = 100,
    expnsn_ref: float = 0.0,
    min_distance_cam_wt: float = 20.0,
    turbine_map=None,
):
    """Position the wind turbines randomly within a specific area. The wind turbines are positioned in the x and y directions.

    Args:
        number_of_wts (int, optional): Number of wind turbines to be generated. Defaults to 1.
        centered_wt (bool, optional): If True, one wind turbines is centered in the camera view. Defaults to True.
        boundary_left_angle (int, optional): Minimum angle of the left boundary in degrees. The angle is measured from the positive y-axis in counterclockwise direction. Defaults to 0.
        boundary_right_angle (int, optional): Maximum angle of the right boundary in degrees. The angle is measured from the positive y-axis in clockwise direction. Defaults to 100.
        min_distance_between_wts (int, optional): Minimum distance between wind turbines. Defaults to 0.
        max_distance (int, optional): Maximum distance from the center of the scene to the wind turbines. Defaults to 100.
        turbine_map (vector, non optional): The map of the wind park.
    """

    def get_xy_sample(
        distance_max: int = 100,
        boundary_left_angle: int = 0,
        boundary_right_angle: int = 0,
    ):
        """Get a random position for a wind turbine.
        The position is sampled in cylindrical coordinates, by sampling angle and distance uniformly.
        The xy-position is then calculated from the cylindrical coordinates.

        Args:
            distance_max (int, optional): Maximum distance from (0,0) to the wind turbines. Defaults to 100.
            boundary_left_angle (int, optional): Minimum angle of the left boundary in degrees. The angle is measured from the positive y-axis in counterclockwise direction. Defaults to 0.
            boundary_right_angle (int, optional): Maximum angle of the right boundary in degrees. The angle is measured from the positive y-axis in clockwise direction. Defaults to 0.

        Returns:
            tuple: A tuple containing the x and y coordinates of the wind turbine.
        """
        distance = random.uniform(0, distance_max)
        angle_deg = random.uniform(0, boundary_left_angle + boundary_right_angle) + 90 - boundary_right_angle
        angle_rad = math.radians(angle_deg)

        return distance * math.cos(angle_rad), distance * math.sin(angle_rad)

    def is_in_range(pos_x, pos_y, min_distance_between_wts, xy_placed):
        """Check if a position is in range of another wind turbine.

        Args:
            pos_x (float): x-coordinate of the position to check.
            pos_y (float): y-coordinate of the position to check.
            min_distance_between_wts (float): Minimum distance between wind turbines.
            xy_placed (list): List of already placed wind turbines.

        Returns:
            bool: True if the position is in range of another wind turbine, False otherwise.
        """
        for x, y in xy_placed:
            if math.sqrt((pos_x - x) ** 2 + (pos_y - y) ** 2) < min_distance_between_wts:
                return True
        return False

    def cross_product(p, q, r):
        """Do cross product for the vectors pq and pr with points p, q, and r given."""
        a = q - p
        b = r - p
        cross_2d = a[0] * b[1] - a[1] * b[0]
        len_a = np.sqrt(np.power(a[0], 2) + np.power(a[1], 2))
        len_b = np.sqrt(np.power(b[0], 2) + np.power(b[1], 2))
        return cross_2d, len_a, len_b

    def dot_product(p, q, r):
        """Do dot product for the vectors pq and pr with points p, q, and r given."""
        a = q - p
        b = r - p
        dot_prod = a[0] * b[0] + a[1] * b[1]
        len_a = np.sqrt(np.power(a[0], 2) + np.power(a[1], 2))
        len_b = np.sqrt(np.power(b[0], 2) + np.power(b[1], 2))
        return dot_prod, len_a, len_b

    def cmpt_expnsn_ang(r, p, q):
        """Compute the angle of the expansion vector. The expansion vector is the vector that when added to a
        given vertex, establishes the new vertex of the new and expanded polygon."""
        rp = p - r
        pq = q - p
        ang_1 = math.atan2(rp[1], rp[0]) - math.pi / 2
        ang_2 = math.atan2(pq[1], pq[0]) - math.pi / 2
        expnsn_ang = (ang_2 - ang_1) / 2 + ang_1
        dot_prod, len_a, len_b = dot_product(p, q, r)
        dot_product_angle = math.acos(dot_prod / (len_a * len_b))
        expnsn_ang = ang_1 + (math.pi - dot_product_angle) / 2
        return expnsn_ang

    map_matrix_size = turbine_map.shape
    if map_matrix_size[0] == 2 and map_matrix_size[1] >= 4:
        turbine_map = turbine_map.T
    elif not (map_matrix_size[0] >= 4 and map_matrix_size[1] == 2) and not (
        map_matrix_size[0] == 2 and map_matrix_size[1] >= 4
    ):
        raise SyntaxError(
            f"The dimension of the windturbine map is not correct. It should be either nx2 or 2xn and currently is: {map_matrix_size}."
        )
    map_matrix_len = turbine_map.shape[0]

    # get all object from the blender scene that start with 'wt_' (wind turbine collections)
    wt_collections = []
    for col in bpy.data.collections:
        col.hide_render = True
        if col.name.lower().startswith("wt_"):
            wt_collections.append(col)

    if map_matrix_len > len(wt_collections):
        raise ValueError(
            f"number_of_wts is larger than the number of wt collections: {map_matrix_len} > {len(wt_collections)}"
        )

    # select a random number of wind turbine collections
    collections_selected = random.sample(wt_collections, map_matrix_len)

    # list of already placed wind turbines as WTObjectSet list
    return_objects = []

    # place the wind turbines based on the map
    # find the polygon around the wind park
    ind_p = np.argmin(turbine_map[:, 1])  # find the lowest point
    ind_p_init = ind_p
    poly_list = []
    poly_list.append(ind_p)
    finish_while = 0
    while not (finish_while):
        ind_q = (ind_p + 1) % map_matrix_len
        for ind_r in range(map_matrix_len):
            p = turbine_map[ind_p, :]
            q = turbine_map[ind_q, :]
            r = turbine_map[ind_r, :]
            cross_2d, len_a, len_b = cross_product(p, q, r)
            if cross_2d == 0:
                if len_b >= len_a:
                    ind_q = ind_r
            elif cross_2d < 0:
                ind_q = ind_r
        ind_p = ind_q
        poly_list.append(ind_p)
        if ind_p == ind_p_init:
            finish_while = 1

    # expand the polygon
    len_poly = len(poly_list) - 1
    expnsn_poly_vrtcs = np.zeros((len_poly, 2))
    for i in range(len_poly):
        p = turbine_map[poly_list[i], :]
        q = turbine_map[poly_list[(i + 1) % len_poly], :]
        r = turbine_map[poly_list[(i - 1) % len_poly], :]
        dot_prod, len_a, len_b = dot_product(p, q, r)
        vrtc_ang = math.acos(dot_prod / (len_a * len_b))  # calculate the angle between given vectors
        expnsn_mag = expnsn_ref / math.sin(
            vrtc_ang / 2
        )  # based on the expnsn_ref, calculate the magnitude of the expansion vector
        expnsn_ang = cmpt_expnsn_ang(r, p, q)  # calculate the angle of the expansion vector
        vrtc_new = p + expnsn_mag * np.array([math.cos(expnsn_ang), math.sin(expnsn_ang)])  # calculate the new vertex
        expnsn_poly_vrtcs[i, :] = vrtc_new

    # find a single point uniformly randomly within the polygon
    # first, divide the polygon into triangles
    n_tri = len_poly - 2  # number of the created triangles
    ind_triangles = np.zeros((n_tri, 3), dtype=int)

    for j in range(n_tri):
        ind_triangles[j, :] = [0, j + 1, j + 2]

    # calculate the area of each resulting triangle
    area = np.zeros((1, n_tri))
    ind_p0 = ind_triangles[0, 0]
    ind_q0 = ind_triangles[0, 1]
    ind_r0 = ind_triangles[0, 2]
    cross_2d, len_a, len_b = cross_product(
        expnsn_poly_vrtcs[ind_p0], expnsn_poly_vrtcs[ind_q0], expnsn_poly_vrtcs[ind_r0]
    )
    area[0, 0] = abs(cross_2d) / 2
    for i in range(1, n_tri):
        p = expnsn_poly_vrtcs[ind_triangles[i, 0]]
        q = expnsn_poly_vrtcs[ind_triangles[i, 1]]
        r = expnsn_poly_vrtcs[ind_triangles[i, 2]]
        cross_2d, len_a, len_b = cross_product(p, q, r)
        area[0, i] = area[0, i - 1] + abs(cross_2d) / 2
    area_total = int(area[0, n_tri - 1])

    # select a triangle randomly based on the areas
    rand_area = random.uniform(0, area_total)
    rand_tri = np.searchsorted(area[0, :], rand_area)
    # find a random point within the randomly selected triangle
    p = expnsn_poly_vrtcs[ind_triangles[rand_tri, 0]]
    q = expnsn_poly_vrtcs[ind_triangles[rand_tri, 1]]
    r = expnsn_poly_vrtcs[ind_triangles[rand_tri, 2]]
    tri_vec_1 = q - p
    tri_vec_2 = r - p
    rand_1 = random.uniform(0, 1)
    rand_2 = random.uniform(0, 1)
    if (rand_1 + rand_2) > 1:
        rand_1 = 1 - rand_1
        rand_2 = 1 - rand_2
    camera_position = p + rand_1 * tri_vec_1 + rand_2 * tri_vec_2

    # find the closest wind turbine, assigning it the wind turbine of interest (wtoi)
    cmr_wt_dis = abs(turbine_map - camera_position)
    dist_list = np.zeros((1, map_matrix_len))
    for i in range(map_matrix_len):
        dist_list[0, i] = np.sqrt(np.power(cmr_wt_dis[i, 0], 2) + np.power(cmr_wt_dis[i, 1], 2))
    ind_wtoi = np.argmin(dist_list)
    # secure a minimum distance to the closest wind turbine
    dist_wtoi = dist_list[0, ind_wtoi]
    if dist_wtoi < min_distance_cam_wt:
        if dist_wtoi == 0:
            search_ang = random.uniform(0, 360) / 180 * math.pi
        else:
            wt2cam = camera_position - turbine_map[ind_wtoi, :]
            search_ang = math.atan2(wt2cam[1], wt2cam[0])
        camera_position = (min_distance_cam_wt - dist_wtoi) * wt2cam / (
            np.sqrt(np.power(wt2cam[0], 2)) + np.power(wt2cam[1], 2)
        ) + turbine_map[ind_wtoi, :]
    cam2wt = turbine_map[ind_wtoi, :] - camera_position
    camera_angle = math.atan2(cam2wt[1], cam2wt[0])
    print(camera_angle / math.pi * 180)

    # move and rotating the park so the the camera is at [0,0] and looks up
    rot_ang = math.pi / 2 - camera_angle
    turbine_map_new = turbine_map - camera_position
    turbine_map_new = np.matmul(
        np.array([[np.cos(rot_ang), -np.sin(rot_ang)], [np.sin(rot_ang), np.cos(rot_ang)]]), turbine_map_new.T
    )
    turbine_map_new = turbine_map_new.T

    for i, col in enumerate(collections_selected):
        col.hide_render = False
        # create a WTObjectSet object for each wind turbine collection
        wea_point = WTObjectSet(
            obj_all=col.objects,
            tower=next((obj for obj in col.objects if obj.name.startswith("tower")), None),
            housing=next((obj for obj in col.objects if obj.name.startswith("housing")), None),
            rotor=next((obj for obj in col.objects if obj.name.startswith("rotor")), None),
            kp_housing_back=next(
                (obj for obj in col.objects if obj.name.startswith("kp_housing_back")),
                None,
            ),
            kp_housing_front=next(
                (obj for obj in col.objects if obj.name.startswith("kp_housing_front")),
                None,
            ),
            kp_tower_top=next(
                (obj for obj in col.objects if obj.name.startswith("kp_tower_top")),
                None,
            ),
            kp_tower_bottom=next(
                (obj for obj in col.objects if obj.name.startswith("kp_tower_bottom")),
                None,
            ),
            kp_tip_1=next((obj for obj in col.objects if obj.name.startswith("kp_tip_1")), None),
            kp_tip_2=next((obj for obj in col.objects if obj.name.startswith("kp_tip_2")), None),
            kp_tip_3=next((obj for obj in col.objects if obj.name.startswith("kp_tip_3")), None),
        )

        pos_x = turbine_map_new[i, 0]
        pos_y = turbine_map_new[i, 1]
        wea_point.set_xy_position(pos_x, pos_y)

        return_objects.append(wea_point)

    return return_objects


def get_camera_horizontal_fov():
    """Get the horizontal FoV of the camera."""
    # get the camera horizontal FoV
    camera_calc = bpy.context.scene.camera

    if camera_calc and camera_calc.type == "CAMERA":
        camera_data = camera_calc.data

        render = bpy.context.scene.render
        aspect_ratio = render.resolution_x / render.resolution_y

        horizontal_fov = 2 * math.atan(math.tan(camera_data.angle / 2) * aspect_ratio)

        return math.degrees(horizontal_fov)
    else:
        raise ValueError("No active camera in the scene found.")


def get_xy_sample(
    distance_max: int = 100,
    boundary_left_angle: int = 0,
    boundary_right_angle: int = 0,
):
    """Get a random position for a wind turbine.
    The position is sampled in cylindrical coordinates, by sampling angle and distance uniformly.
    The xy-position is then calculated from the cylindrical coordinates.

    Args:
        distance_max (int, optional): Maximum distance from (0,0) to the wind turbines. Defaults to 100.
        boundary_left_angle (int, optional): Minimum angle of the left boundary in degrees. The angle is measured from the positive y-axis in counterclockwise direction. Defaults to 0.
        boundary_right_angle (int, optional): Maximum angle of the right boundary in degrees. The angle is measured from the positive y-axis in clockwise direction. Defaults to 0.

    Returns:
        tuple: A tuple containing the x and y coordinates of the wind turbine.
    """
    distance = random.uniform(0, distance_max)
    angle_deg = random.uniform(0, boundary_left_angle + boundary_right_angle) + 90 - boundary_right_angle
    angle_rad = math.radians(angle_deg)

    return distance * math.cos(angle_rad), distance * math.sin(angle_rad)


def position_objects_in_scene(
    boundary_left_angle: int,
    boundary_right_angle: int,
    distance_max: int = 1000,
    xy_points: List = [],
    object_distance: int = 100,
):
    """Randomly positions the objects collections in the scene.
    Only objects that start with 'obj_' are considered.

    Args:
        boundary_left_angle (int, optional): Minimum angle of the left boundary in degrees. The angle is measured from the positive y-axis in counterclockwise direction. Defaults to 0.
        boundary_right_angle (int, optional): Maximum angle of the right boundary in degrees. The angle is measured from the positive y-axis in clockwise direction. Defaults to 100.
        distance_max (int, optional): Maximum distance from the center of the scene to the objects. Defaults to 1000.
    """

    object_collections = [col for col in bpy.data.collections if col.name.lower().startswith("obj_")]

    # get the camera horizontal FoV
    horizontal_fov_degrees = get_camera_horizontal_fov()
    if horizontal_fov_degrees / 2 < boundary_left_angle:
        boundary_left_angle = horizontal_fov_degrees / 2
    if horizontal_fov_degrees / 2 < boundary_right_angle:
        boundary_right_angle = horizontal_fov_degrees / 2

    for col in object_collections:
        col.hide_render = False

        collision = True
        counter = 0
        while collision:
            collision = False
            loc_x, loc_y = get_xy_sample(distance_max, boundary_left_angle, boundary_right_angle)
            for x, y in xy_points:
                if math.sqrt((x - loc_x) ** 2 + (y - loc_y) ** 2) < object_distance:
                    collision = True
                    break
            if counter > 200:
                col.hide_render = True
                print(
                    "WARNING: Too many tries to place an object inside the given boundary. The object is not visualized now."
                )
                break
            counter += 1

        for obj in col.objects:
            if obj.parent is None:
                obj.location = (loc_x, loc_y, obj.location[2])


def generate_wind_turbine_set(
    number_of_wts: int = 1,
    centered_wt: bool = True,
    boundary_left_angle: int = 0,
    boundary_right_angle: int = 0,
    min_distance_between_wts: int = 60,
    max_distance: int = 100,
):
    """Position the wind turbines randomly within a specific area. The wind turbines are positioned in the x and y directions.

    Args:
        number_of_wts (int, optinal): Number of wind turbines to be generated. Defaults to 1.
        centered_wt (bool, optional): If True, one wind turbines is centered in the camera view. Defaults to True.
        boundary_left_angle (int, optional): Minimum angle of the left boundary in degrees. The angle is measured from the positive y-axis in counterclockwise direction. Defaults to 0.
        boundary_right_angle (int, optional): Maximum angle of the right boundary in degrees. The angle is measured from the positive y-axis in clockwise direction. Defaults to 100.
        min_distance_between_wts (int, optional): Minimum distance between wind turbines. Defaults to 0.
        max_distance (int, optional): Maximum distance from the center of the scene to the wind turbines. Defaults to 100.
    """

    def is_in_range(pos_x, pos_y, min_distance_between_wts, xy_placed):
        """Check if a position is in range of another wind turbine.

        Args:
            pos_x (float): x-coordinate of the position to check.
            pos_y (float): y-coordinate of the position to check.
            min_distance_between_wts (float): Minimum distance between wind turbines.
            xy_placed (list): List of already placed wind turbines.

        Returns:
            bool: True if the position is in range of another wind turbine, False otherwise.
        """
        for x, y in xy_placed:
            if math.sqrt((pos_x - x) ** 2 + (pos_y - y) ** 2) < min_distance_between_wts:
                return True
        return False

    # get the camera horizontal FoV
    horizontal_fov_degrees = get_camera_horizontal_fov()
    if horizontal_fov_degrees / 2 < boundary_left_angle:
        boundary_left_angle = horizontal_fov_degrees / 2
    if horizontal_fov_degrees / 2 < boundary_right_angle:
        boundary_right_angle = horizontal_fov_degrees / 2

    # get all object from the blender scene that start with 'wt_' (wind turbine collections)
    wt_collections = []
    for col in bpy.data.collections:
        col.hide_render = True
        if col.name.lower().startswith("wt_"):
            wt_collections.append(col)

    if number_of_wts > len(wt_collections):
        raise ValueError(
            f"number_of_wts is larger than the number of wt collections: {number_of_wts} > {len(wt_collections)}"
        )

    # select a random number of wind turbine collections
    collections_selected = random.sample(wt_collections, number_of_wts)

    # list of already placed wind turbines
    xy_placed = []
    # list of already placed wind turbines as WTObjectSet list
    return_objects = []

    for i, col in enumerate(collections_selected):
        # create a WTObjectSet object for each wind turbine collection
        col.hide_render = False
        wea_point = WTObjectSet(
            obj_all=col.objects,
            tower=next((obj for obj in col.objects if obj.name.startswith("tower")), None),
            housing=next((obj for obj in col.objects if obj.name.startswith("housing")), None),
            rotor=next((obj for obj in col.objects if obj.name.startswith("rotor")), None),
            kp_housing_back=next(
                (obj for obj in col.objects if obj.name.startswith("kp_housing_back")),
                None,
            ),
            kp_housing_front=next(
                (obj for obj in col.objects if obj.name.startswith("kp_housing_front")),
                None,
            ),
            kp_tower_top=next(
                (obj for obj in col.objects if obj.name.startswith("kp_tower_top")),
                None,
            ),
            kp_tower_bottom=next(
                (obj for obj in col.objects if obj.name.startswith("kp_tower_bottom")),
                None,
            ),
            kp_tip_1=next((obj for obj in col.objects if obj.name.startswith("kp_tip_1")), None),
            kp_tip_2=next((obj for obj in col.objects if obj.name.startswith("kp_tip_2")), None),
            kp_tip_3=next((obj for obj in col.objects if obj.name.startswith("kp_tip_3")), None),
        )

        # get a random position for the wind turbine
        pos_x, pos_y = 0, 0
        if not (centered_wt and i == 0):
            pos_x, pos_y = get_xy_sample(max_distance, boundary_left_angle, boundary_right_angle)

        iterations = 0
        while is_in_range(pos_x, pos_y, min_distance_between_wts, xy_placed):
            if iterations >= 100:
                raise RuntimeError("Could not find valid wind turbine position after 100 attempts")
            pos_x, pos_y = get_xy_sample(max_distance, boundary_left_angle, boundary_right_angle)
            iterations += 1

        wea_point.set_xy_position(pos_x, pos_y)
        xy_placed.append((pos_x, pos_y))

        return_objects.append(wea_point)

    return return_objects, xy_placed


def randomization_material_properties(wea_selection: List[WTObjectSet], config: dict):
    """
    Randomize the material properties metallic and roughness of the wind turbines (WEAs) from the given list of WTObjectSet objects.
    The modification is the same for each part of the wind turbine, but different for each wind turbine.

    Args:
        wea_selection (List[WTObjectSet]): List of WTObjectSet objects.
        config (dict): Configuration dictionary.
    """

    metallic = np.clip(
        sample_distribution(
            config["METALLIC"]["DISTRIBUTION"],
            config["METALLIC"]["ARG_1"],
            config["METALLIC"]["ARG_2"],
        ),
        0,
        1,
    )
    roughness = np.clip(
        sample_distribution(
            config["ROUGHNESS"]["DISTRIBUTION"],
            config["ROUGHNESS"]["ARG_1"],
            config["ROUGHNESS"]["ARG_2"],
        ),
        0,
        1,
    )

    for wea in wea_selection:
        for material in wea.rotor.data.materials:
            material.metallic = metallic
            material.roughness = roughness

        for material in wea.housing.data.materials:
            material.metallic = metallic
            material.roughness = roughness

        for material in wea.tower.data.materials:
            material.metallic = metallic
            material.roughness = roughness


def add_gaussian_noise(image: np.array, mean=0, sigma=25) -> np.array:
    """Add Gaussian noise to an image.

    Args:
        image: Input image (numpy array)
        mean: Mean of Gaussian distribution
        sigma: Standard deviation of Gaussian distribution

    Returns:
        Noisy image (numpy array)
    """

    image = image.astype(np.float32) / 255.0

    # Generate Gaussian noise
    noise = np.random.normal(mean, sigma / 255.0, image.shape).astype(np.float32)

    # Add noise to image
    noisy_image = image + noise

    # Clip values to [0, 1] range and convert back to uint8
    noisy_image = np.clip(noisy_image, 0, 1)
    noisy_image = (noisy_image * 255).astype(np.uint8)

    return noisy_image


def get_output_paths(pre_path_str: str, base_path: str) -> OutputPaths:
    """Create and return output directory structure for image generation.

    This function creates a timestamped directory structure within a 'data' subdirectory
    of the provided base path. The structure includes separate folders for training and
    validation datasets, each containing subdirectories for images, labels, and label
    visualizations.

    Args:
        pre_path_str (str): Prefix string to be used in the generated folder name.
                           This will be combined with a timestamp to create a unique
                           directory name.
        base_path (str): Base directory path where the 'data' folder and subsequent
                        directory structure will be created.

    Returns:
        tuple: A tuple containing:
            - str: Absolute path to the newly created root directory
            - OutputPaths: TypedDict containing training and validation paths with
                          subdirectories for:
                          - path_images: Directory for storing image files
                          - path_images_keypoints: Directory for label visualization files
                          - path_keypoints: Directory for label/annotation files

    Directory Structure Created:
        base_path/
        └── data/
            └── {pre_path_str}_{timestamp}/
                ├── images/
                │   ├── train/
                │   └── val/
                ├── label_vis/
                │   ├── train/
                │   └── val/
                └── labels/
                    ├── train/
                    └── val/

    Raises:
        ValueError: If a directory with the generated name already exists.

    """

    new_folder_name_uuid = pre_path_str + "_" + str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    new_folder_path_abs = os.path.join(base_path, "data/" + new_folder_name_uuid)

    path_images = os.path.join(new_folder_path_abs, "images")
    path_images_keypoints = os.path.join(new_folder_path_abs, "label_vis")
    path_keypoints = os.path.join(new_folder_path_abs, "labels")

    path_images_train = os.path.join(path_images, "train")
    path_images_keypoints_train = os.path.join(path_images_keypoints, "train")
    path_keypoints_train = os.path.join(path_keypoints, "train")

    path_images_test = os.path.join(path_images, "val")
    path_images_keypoints_test = os.path.join(path_images_keypoints, "val")
    path_keypoints_test = os.path.join(path_keypoints, "val")

    if not os.path.exists(new_folder_path_abs):
        os.makedirs(new_folder_path_abs)
        os.makedirs(path_images)
        os.makedirs(path_images_keypoints)
        os.makedirs(path_keypoints)
        os.makedirs(path_images_train)
        os.makedirs(path_images_keypoints_train)
        os.makedirs(path_keypoints_train)
        os.makedirs(path_images_test)
        os.makedirs(path_images_keypoints_test)
        os.makedirs(path_keypoints_test)
    else:
        raise ValueError("Pathname already exists.")

    return new_folder_path_abs, OutputPaths(
        training=OutputPathsDict(
            path_images=path_images_train,
            path_images_keypoints=path_images_keypoints_train,
            path_keypoints=path_keypoints_train,
        ),
        validation=OutputPathsDict(
            path_images=path_images_test,
            path_images_keypoints=path_images_keypoints_test,
            path_keypoints=path_keypoints_test,
        ),
    )


def get_windturbines(min_lat, min_lon, max_lat, max_lon, timeout=30):
    """
    Query wind turbines from OpenStreetMap using the Overpass API.

    Args:
        min_lat (float): Minimum latitude of the bounding box
        min_lon (float): Minimum longitude of the bounding box
        max_lat (float): Maximum latitude of the bounding box
        max_lon (float): Maximum longitude of the bounding box
        timeout (int): Query timeout in seconds (default: 30)

    Returns:
        dict: JSON response from the Overpass API
    """

    # Overpass API endpoint
    url = "https://overpass-api.de/api/interpreter"

    # Build the Overpass QL query with variable coordinates
    query = f"""
    [out:json][timeout:{timeout}];
    (
        node['generator:source'=wind]({min_lat},{min_lon},{max_lat},{max_lon});
        way['generator:source'=wind]({min_lat},{min_lon},{max_lat},{max_lon});
        relation['generator:source'=wind]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out body geom;
    """

    # Parameters for the GET request
    params = {"data": query}

    try:
        # Make the GET request
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Return the JSON response
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        return None


def get_relative_wea_map(lat: int, lon: int, padding: int):
    lat_offset = padding / 111
    lon_offset = padding / (111 * math.cos(math.radians(lat)))

    min_lat = lat - lat_offset
    max_lat = lat + lat_offset
    min_lon = lon - lon_offset
    max_lon = lon + lon_offset

    print(f"Querying wind turbines in bounding box:")
    print(f"  Min: {min_lat}, {min_lon}")
    print(f"  Max: {max_lat}, {max_lon}")

    # Get wind turbine data
    data = get_windturbines(min_lat, min_lon, max_lat, max_lon)

    positions = []
    if data:
        print(f"\nFound {len(data.get('elements', []))} wind turbine elements")
        if len(data.get("elements", [])) > 0:
            first_wea = data.get("elements", [])[0]
            x_ref, y_ref = first_wea["lat"], first_wea["lon"]

            for element in data.get("elements", []):
                if "lat" in element and "lon" in element:
                    positions.append(
                        [111 * (element["lat"] - x_ref), (111 * math.cos(math.radians(lat))) * (element["lon"] - y_ref)]
                    )
                else:
                    print(f"No lat and lon found: {element}")
    return positions
