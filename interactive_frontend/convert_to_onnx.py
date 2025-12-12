#!/usr/bin/env python3
"""
Convert PyTorch .pt model to ONNX format for browser inference.

Usage:
    python convert_to_onnx.py --model model.pt --output model.onnx
    
Optional arguments:
    --input-size WIDTH HEIGHT  (default: 480 640)
    --opset-version VERSION    (default: 14)
"""

import argparse
import torch
import torch.onnx

def convert_to_onnx(model_path, output_path, input_size=(480, 640), opset_version=14):
    """
    Convert PyTorch model to ONNX format.
    
    Args:
        model_path: Path to the .pt model file
        output_path: Path where to save the .onnx file
        input_size: Tuple of (height, width) for input image
        opset_version: ONNX opset version
    """
    print(f"Loading PyTorch model from {model_path}...")
    
    # Load the model
    try:
        # Try loading as a complete model
        model = torch.load(model_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying to load as state dict...")
        # If it fails, it might be just the state dict
        # You'll need to initialize your model architecture here
        raise NotImplementedError(
            "Please modify this script to load your specific model architecture"
        )
    
    # If model is a dict with 'model' key (common in training checkpoints)
    if isinstance(model, dict):
        if 'model' in model:
            model = model['model']
        elif 'state_dict' in model:
            print("Found state_dict in checkpoint. Please initialize your model architecture.")
            raise NotImplementedError(
                "Please modify this script to load your specific model architecture"
            )
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
    
    print(f"Converting to ONNX with input size: {input_size}...")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✓ Model successfully converted to ONNX: {output_path}")
    
    # Verify the model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid")
        
        # Print model info
        print("\nModel Information:")
        print(f"  Inputs: {[inp.name for inp in onnx_model.graph.input]}")
        print(f"  Outputs: {[out.name for out in onnx_model.graph.output]}")
        
    except ImportError:
        print("Note: Install 'onnx' package to verify the exported model")
    except Exception as e:
        print(f"Warning: Could not verify ONNX model: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='Convert PyTorch model to ONNX format'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to the PyTorch .pt model file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path where to save the .onnx file'
    )
    parser.add_argument(
        '--input-size',
        type=int,
        nargs=2,
        default=[480, 640],
        help='Input size (width height) for the model (default: 480 640)'
    )
    parser.add_argument(
        '--opset-version',
        type=int,
        default=14,
        help='ONNX opset version (default: 14)'
    )
    
    args = parser.parse_args()
    
    convert_to_onnx(
        args.model,
        args.output,
        tuple(args.input_size),
        args.opset_version
    )

if __name__ == '__main__':
    main()

