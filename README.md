# Wind Turbine Synthetic Vision

## Description
This repository contains the code used to generate synthetic images of wind turbines, as well as the trained YOLOv11 object detection model based on this synthetic dataset.

## Project Status
This is the camera-ready version of the code for the publication of the ICMV paper “Wind Turbine Feature Detection Using Deep Learning and Synthetic Data” by Arash Shahirpour, Jakob Gebler, Manuel Sanders, and Tim Reuscher.

Future development will include:
- Integration into a Docker environment
- Simplified parameterization and configurability
- A production-grade release

## Dependencies
- [BlenderProc](https://github.com/DLR-RM/BlenderProc) (GPLv3)
- [YOLOv11](link-to-the-fork-or-original-if-available) (AGPL-3.0)
- NumPy, Pillow, OpenCV, and other standard Python libraries

## Installation

It is recommended to use a virtual environment to manage dependencies.

1.  **Create and activate the virtual environment:**
    ```bash
    python3.11 -m venv ./venv
    source ./venv/bin/activate
    ```

2.  **Install the required package:**
    ```bash
    pip install blenderproc==2.7
    ```

3.  **Prepare background images:**
    Create a directory named `background` in the `toolbox` directory and populate it with random images (e.g., landscapes) to be used for background randomization.
    * Recommended source: [LHQ-1024 Dataset](https://www.kaggle.com/datasets/dimensi0n/lhq-1024)

## Usage

Run the main processing script within the `toolbox` directory:

```bash
blenderproc run main.py
```

**Important: BlenderProc 2.7 automatically installs Blender 3.5.1. Ensure your scene files were created with Blender 3.5.1 to align with this version.**

## Authors and Acknowledgment
- [Arash Shahirpour](https://github.com/arash-sp)  
- [Jakob Gebler](https://github.com/Hapyr)  
- Manuel Sanders  
With contributions from Tim Reuscher.
Institute of Automatic Control – RWTH Aachen University


## License
This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

It includes and modifies components from YOLOv11 (AGPL-3.0), and uses BlenderProc (GPLv3), both of which enforce free software licensing.  
See the [`LICENSE`](./LICENSE) file for full terms.

© 2025 Arash Shahirpour, Jakob Gebler, Manuel Sanders, RWTH Aachen University.

