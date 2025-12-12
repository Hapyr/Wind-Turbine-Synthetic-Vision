# Wind Turbine Synthetic Vision Toolbox

This repository provides a BlenderProc addon for generating synthetic wind turbine datasets, including rendered images and annotated files.
The image rendering can be done with the following main options: 
- Flexible background
 - Random background images: when this option is chosen, each image receives a random background. The background images must be provided. 
 - Rendered ocean background: when this option is selected, wind turbines are placed on an off-shore wind park.

- Win turbine placement
 - Random wind turbine placement in front of the camera: with this option, the desired number of wind turbines will be placed randomly in front of the camera.
 - Camera placement based on a map: when this option is selected and a map with the local coordinates of wind turbines is provided, the camera will be placed randomly within the provided map, always pointed at the closest wind turbine. 

- Configuration pipeline
 - This includes scene setup, camera parameters, and post-processing effects.
 - A configuration file is responsible for the customization of these features. 


# Installation and Usage
## Installation
**We recommend using a virtual environment to avoid conflicts with system packages.**

You can create and activate a virtual environment with:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

In order to use the synthetic vision toolbox, BlenderProc must be installed, which can be achieved by simply using the following pip command:
```bash
pip install blenderproc==2.8.0
```
The first execution of BlenderProc will automatically download the correct version of Blender as well. The dependencies are listed in the `requirements.txt` file. 
Since Blender and BlenderProc use their own integrated python interpreter independently from the system, this library must be installed with the pip command within the lib directory:
```bash
blenderproc pip install -e .
```
In case this command did not work, you could use the following command instead:
```
/home/<user_name>/blender/blender-4.2.1-linux-x64/4.2/python/bin/python3.11 -m pip install -e .
```
This is also required for any additional python dependencies.

# Preparation
For the synthetic vision toolbox to work, a folder named "background" must be created in the toolbox folder. You can fill this folder with the images you wish to use as the background. A script (`background_dataset_download.sh`) is provided to download a set of background images from a kaggle dataset (13GB space required for the download).

To test if the installation was successful, the `example_dev.py` script is provided to generate a test dataset. Successful execution of the example indicates that the toolbox and the rest of the pipeline work properly.
```bash
blenderproc run example_dev.py
```

As also implemented in the example file, the toolbox is included in the python script by the following lines:
```python
import wind_turbine_synthetic_vision.helper as helper
from wind_turbine_synthetic_vision.generator import DatasetGenerator
```


## Development

**This section is for development purposes only.** Since BlenderProc uses its own Python interpreter, for development of the library the imports need to be added to the import paths and a pip install of the library is not needed.

As shown in the `example_dev.py` script,these paths must be manually imported when developing:
```python
sys.path.insert(0, "lib/src")
try:
    import wind_turbine_synthetic_vision.helper as helper
    from wind_turbine_synthetic_vision.generator import DatasetGenerator
except ImportError as e:
    print(f'No Import: {e}')
```
This is not required for the usage of the library, when no library manipulation is intended.

Similar to the example file, a script can be executed with the following command:
```bash
blenderproc <file_name>.py
```

## Blender Scene
BlenderProc version 2.8.0 uses Blender 4.2.1. Therefore the provided scene files need to align with this version.

The Blender scene is stored in the `scene` folder.
Multiple wind turbines can be placed in the scene, as demonstrated in the `example_wt_set.blend` file.
Each wind turbine is stored in a unique blender collection. 

Only collections that fulfill the following naming convention and hierarchy are considered as wind turbines and can be used for automatic placement and labeling:

- WT collection: `wt_collection_*`
    - Tower object: `tower*`
        - Housing object: `housing*`
            - Rotor object: `rotor*`
                - Keypoint Housing Front object: `kp_housing_front*`
                - Keypoint Tip 1 object: `kp_tip_1*`
                - Keypoint Tip 2 object: `kp_tip_2*`
                - Keypoint Tip 3 object: `kp_tip_3*`
            - Keypoint Housing Back object: `kp_housing_back*`
        - Keypoint Tower Bottom object: `kp_tower_bottom*`
        - Keypoint Tower Top object: `kp_tower_top*`
            
The `*` in the naming convention needs to be replaced with a unique identifier for each wind turbine.
The keypoints have no visual representation and are only used for placement and labeling.
We provide an example file `example_wt_set.blend` to show the required naming convention.


## Background Images

As mentioned previously, wind turbines can be rendered and placed either on a variety of background images or on a rendered ocean back ground. The background images must be stored in the `background` folder.
A script (`background_dataset_download.sh`) is provided to download a set of background images from a kaggle dataset. This download serves only as suggestion and is not mandatory for the functionality of the toolbox. If a random background should be used or the ocean should be rendered as the background can be defined with the configuration file when "RANDOM_BACKGROUND" is set to either "True" or "False".

## Generate from MAP

As mentioned briefly, the toolbox can either place the wind turbines randomly in front of the camera or place the camera randomly within the boundaries of a provided wind park map. 
TODO1: add map, excel file and folder and describe here.
As shown in the "example_dev.py" file, the toolbox uses random backgrounds with the "generate()" command, and it renders the ocean background when "generate_from_map()" command is used. 


## Configuration Parameters
Configuration file for generating synthetic wind turbine datasets using Blender. Controls scene setup, camera parameters, and post-processing effects.

For the correct indentation and example values of all parameters, please refer to the default configuration file located at:
`lib/src/wind_turbine_synthetic_vision/config.yaml`

### Basic Settings
| Parameter | Type | Description |
|-----------|------|-------------|
| `RANDOM_BACKGROUND` | boolean | Use random background images instead of blender scene elements (e.g. sky or ocean)  |
| `RANDOM_OBJECTS` | boolean | In addition to wind turbines, objects can be rendered and randomly positioned. Only objects that begin with “obj_” in the Blender scene are taken into account. |
---


### Wind Turbine Placement HERE PLEASE ALSO ADD THE VARIABLES USED FOR FROM THE MPA FUNCTION TODO2
`PLACEMENT`
| Sub-Parameter | Type | Description |
|-----------|------|-------------|
| `NUMBER_OF_WTS` | integer | Number of wind turbines in scene |
| `CENTERED_WT` | boolean | If true, one wind turbine is placed at (0,0) |
| `BOUNDARY_LEFT_ANGLE` | integer | Left viewing boundary (degrees from +Y axis) |
| `BOUNDARY_RIGHT_ANGLE` | integer | Right viewing boundary (degrees from +Y axis) | TODO3 explain what this means
| `MIN_DISTANCE_BETWEEN_WTS` | integer | Minimum distance between turbines |
| `MAX_DISTANCE` | integer | Maximum distance from (0,0) | TODO EXOLAIN MORE
---

### Housing Rotation
`HOUSING_ROTATION`
| Sub-Parameter | Type | Description |
|-----------|------|-------------|
| `FIXED` | boolean | Use the mean rotation angle on all wind turbines for every image. It still can get normal distributed around this fixed value |
| `NORMAL_DISTRIBUTED_SET` | boolean | Use normal distribution vs uniform |
| `MEAN` | float | Mean rotation angle (degrees) |
| `STD_DEV` | float | Standard deviation for normal distribution |
| `ROTATION_ANGLE_HOUSING_MIN` | integer | Min rotation angle for uniform sampling (degrees) |
| `ROTATION_ANGLE_HOUSING_MAX` | integer | Max rotation angle for uniform sampling (degrees) |
---

### Rotor Rotation
`ROTOR_ROTATION`
| Sub-Parameter | Type | Description |
|-----------|------|-------------|
| `FIXED` | boolean | Use fixed rotor position. If true, `ROTATION_ANGLE_ROTOR_MIN` is used. If false, the angle is sampled from the uniform distribution. |
| `ROTATION_ANGLE_ROTOR_MIN` | integer | Min rotor angle (degrees) |
| `ROTATION_ANGLE_ROTOR_MAX` | integer | Max rotor angle (degrees) |
---

### Wind Turbine Parameters
`WT_PARAMETERS`
| Sub-Parameter | Type | Description |
|-----------|------|-------------|
| `TURBINE_RADIUS` | float | Blade radius (meters). Has no visual effect. Used for lens calculation.  |
| `HOUSING_HEIGHT` | integer | Tower height (meters). Has no visual effect. Used for Camera pitch angle calculation. |
| `RANDOMIZE_SHAFT_SCALING_FACTOR` | boolean | Randomize rotor size scaling. |
| `SHAFT_SCALING_FACTOR_MIN` | float | Min rotor scale factor |
| `SHAFT_SCALING_FACTOR_MAX` | float | Max rotor scale factor |
---

### Camera Settings
`CAMERA`
| Sub-Parameter | Type | Description |
|-----------|------|-------------|
| `FIXED_CAMERA_LENS` | boolean | Use fixed focal length |
| `LENS_MM` | float | Focal length (mm) |
| `SENSOR_HEIGHT` | float | Camera sensor height (mm) |
| `CAMERA_DISTANCE_MIN` | integer | Min distance of camera to (0,0) |
| `CAMERA_DISTANCE_MAX` | integer | Max distance of camera to (0,0) |
| `PITCH_CENTERED` | boolean | Center camera pitch on turbine |
| `ANGLE_BOUND` | integer | Camera roll, pitch, yaw variation, negative and positive. (degrees) |
| `X_RES` | integer | Image width (pixels) |
| `Y_RES` | integer | Image height (pixels) |
| `MIN_HEIGHT` | integer | Min camera height |
| `MAX_HEIGHT` | integer | Max camera height |
---

### Post-Processing Effects
`POSTPROCESSING`
| Sub-Parameter | Type | Description |
|-----------|------|-------------|
| `NOISE_BACKGROUND_THRESHOLD` | float | Probability of replacing background image with random noise |
| `TRAIN_VALID_RATIO` | float | Training/validation split ratio |
| `GAUSSIAN_NOISE_ARTEFACT` |  | More details below |
| `COMPRESSION` |  | More details below |
| `HUE_SHIFT` |  | More details below |

**Gaussian Noise Artefacts** 
| Sub-Parameter | Type | Description |
|-----------|------|-------------|
| `THRESHOLD` | float | Probability of adding Gaussian noise |
| `SIGMA_MAX` | integer | Max noise intensity |

**JPEG Compression**
| Sub-Parameter | Type | Description |
|-----------|------|-------------|
| `THRESHOLD` | float | Probability of JPEG compression |
| `QUALITY_MIN` | integer | Min JPEG quality |
| `QUALITY_MAX` | integer | Max JPEG quality |


**Color Shifts**
| Sub-Parameter | Type | Description |
|-----------|------|-------------|
| `THRESHOLD_FOREGROUND` | float | Foreground hue shift probability |
| `THRESHOLD_BACKGROUND` | float | Background hue shift probability |
| `HUE_SHIFT` |  | More details below |
| `SATURATION_SHIFT` |  | More details below |
| `VALUE_SHIFT` |  | More details below |


Several more parameter can be set to shift the hue of the image as seen below.
Hue, Saturation and Value can be shifted independently.

**Hue Shift** `HUE_SHIFT` (degrees 0-360)
- `DISTRIBUTION`: normal / uniform
- `ARG_1`: Mean value / min
- `ARG_2`: Standard deviation / max

**Saturation Shift** `SATURATION_SHIFT` (percent 0-100)
- `DISTRIBUTION`: normal / uniform
- `ARG_1`: Mean value / min 
- `ARG_2`: Standard deviation / max

**Value Shift** `VALUE_SHIFT` (brightness percent 0-100)
- `DISTRIBUTION`: normal / uniform
- `ARG_1`: Mean value
- `ARG_2`: Standard deviation
---

### Material Properties
`MATERIAL`
| Sub-Parameter | Type | Description |
|-----------|------|-------------|
| `METALLIC` |  | More details below |
| `ROUGHNESS` |  | More details below |

**Metallic** (0-1 range)
- `DISTRIBUTION`: normal / uniform
- `ARG_1`: Mean value / min 
- `ARG_2`: Standard deviation / max

**Roughness** (0-1 range)
- `DISTRIBUTION`: normal / uniform
- `ARG_1`: Mean value / min 
- `ARG_2`: Standard deviation / max
---

### Sky Texture (Nishita Atmospheric Model)
`SKY_TEXTURE`
| Sub-Parameter | Type | Description |
|-----------|------|-------------|
| `TYPE` | string | Sky model: REETHAM, HOSEK_WILKIE, NISHITA |
| `SUN_INTENSITY` | 0-1000 | Sun brightness |
| `SUN_ALTITUDE` | 0-3000m | Sun altitude above sea level |
| `SUN_SIZE` | 0-1.5708 | Sun angular size |
| `SUN_ELEVATION` | 0-1.5708 rad | Sun elevation angle (0=horizon, 1.5708=zenith) |
| `SUN_ROTATION` | 0-360° | Sun azimuth angle |
| `AIR_DENSITY` | 0-10 | Atmospheric density (1.0 = Earth standard) |
| `DUST_DENSITY` | 0-10 | Atmospheric dust particles |
| `OZONE_DENSITY` | 0-10 | Ozone layer density |
| `TURBIDITY` | 0-10 | Atmospheric turbidity (1.0 = clear sky) |

For each parameter except `TYPE` the following parameters need to be set:
- `DISTRIBUTION`: normal / uniform
- `ARG_1`: Mean value / min 
- `ARG_2`: Standard deviation / max
