# Wind Turbine Detection - Interactive Frontend

A real-time web application for detecting wind turbines from webcam feed using a PyTorch model running entirely in the browser.

## Quick Start

### 1. Convert Your PyTorch Model to ONNX

First, you need to convert your PyTorch `.pt` model to ONNX format:

```bash
# Install required packages
pip install torch onnx

# Convert your model
python convert_to_onnx.py --model your_model.pt --output wind_turbine_model.onnx
```

If you have a custom model architecture, you may need to modify `convert_to_onnx.py` to properly load your model.

### 2. Run the Web Application

```bash
# Using Python
python main.py

# Then open http://localhost:8000 in your browser
```

## Customization

### Adjusting Model Input/Output

The application assumes a standard object detection model format. You may need to adjust the preprocessing and post-processing functions in `app.js`:

1. **Input Preprocessing** (`preprocessImage` function):
   - Default: 480x640 RGB image (width x height), normalized to [0, 1]
   - Modify `MODEL_INPUT_WIDTH` and `MODEL_INPUT_HEIGHT` constants at the top of `app.js` if your model expects different dimensions
   - Modify if your model expects different normalization

2. **Output Processing** (`processDetections` function):
   - Default: YOLO-style output [x_center, y_center, width, height, confidence, ...]
   - Modify based on your model's output format

### Changing Detection Confidence Threshold

In `app.js`, find the line:

```javascript
if (confidence > 0.5) {
```

Change `0.5` to your desired threshold (0.0 to 1.0).

## Browser Compatibility

- Chrome/Edge: ✓ Full support
- Firefox: ✓ Full support
- Safari: ✓ Full support (iOS 11+)
- Mobile browsers: ✓ Full support


