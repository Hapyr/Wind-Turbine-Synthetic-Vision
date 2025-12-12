# Wind Turbine Synthetic Vision
# Copyright (C) 2025 Arash Shahirpour, Jakob Gebler, Manuel Sanders
# Institute of Automatic Control - RWTH Aachen University

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import torch
from ultralytics import YOLO
import os

from ultralytics.models.yolo.pose import PoseTrainer
torch.cuda.empty_cache()


config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "config_train.yaml"))

args = dict(model="pretrained/yolo11s-pose.pt", data=config_path, epochs=150)
trainer = PoseTrainer(overrides=args)

trainer.train()
