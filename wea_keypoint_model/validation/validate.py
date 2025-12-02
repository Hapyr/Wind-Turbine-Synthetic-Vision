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

from ultralytics import YOLO

model = YOLO("/home/jakob/Dokumente/IRT/modele_final/val_data/model-s/best.pt")

results = model.val(plots=True, data="config.yaml")

print(f"Box mAP50: {results.box.map50}")
print(f"Box mAP50-95: {results.box.map}")
print(f"Pose mAP50: {results.pose.map50}")
print(f"Pose mAP50-95: {results.pose.map}")
