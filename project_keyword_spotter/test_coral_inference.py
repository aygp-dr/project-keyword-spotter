#!/usr/bin/env python3

import os
import time
import numpy as np
import argparse
from PIL import Image

def download_test_files():
    """Download test files if they don't exist."""
    # Get the project directory
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_dir, 'models')
    
    # Define file paths
    model_file = os.path.join(models_dir, 'mobilenet_v2_1.0_224_quant_edgetpu.tflite')
    label_file = os.path.join(models_dir, 'imagenet_labels.txt')
    image_file = os.path.join(models_dir, 'parrot.jpg')
    
    # Create directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Download test files if they don't exist
    if not os.path.exists(model_file):
        print(f"Downloading model to {model_file}...")
        os.system(f"wget -q https://github.com/google-coral/test_data/raw/master/mobilenet_v2_1.0_224_quant_edgetpu.tflite -O {model_file}")
    
    if not os.path.exists(label_file):
        print(f"Downloading labels to {label_file}...")
        os.system(f"wget -q https://github.com/google-coral/test_data/raw/master/imagenet_labels.txt -O {label_file}")
    
    if not os.path.exists(image_file):
        print(f"Downloading test image to {image_file}...")
        os.system(f"wget -q https://github.com/google-coral/test_data/raw/master/parrot.jpg -O {image_file}")
    
    return model_file, label_file, image_file

def run_inference(model_file, label_file, image_file):
    """Run inference on the Edge TPU."""
    try:
        # Import the TensorFlow Lite runtime
        from tflite_runtime.interpreter import Interpreter
        from tflite_runtime.interpreter import load_delegate
        
        print("Loading Edge TPU delegate...")
        # Load the delegate
        delegates = [load_delegate('libedgetpu.so.1')]
        
        print(f"Loading model: {model_file}")
        # Create the interpreter with the Edge TPU delegate
        interpreter = Interpreter(
            model_path=model_file,
            experimental_delegates=delegates)
        
        # Allocate tensors
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Check the type of the input tensor
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]
        
        # Load the image
        print(f"Loading image: {image_file}")
        image = Image.open(image_file).convert('RGB').resize((width, height))
        
        # Pre-process the image
        # Add a batch dimension and convert to float32
        input_data = np.expand_dims(np.array(image), axis=0)
        
        # Check if the model is quantized
        floating_model = input_details[0]['dtype'] == np.float32
        if floating_model:
            input_data = (np.float32(input_data) - 127.5) / 127.5
        
        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        print("Running inference on Edge TPU...")
        start_time = time.time()
        interpreter.invoke()
        inference_time = time.time() - start_time
        
        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        results = np.squeeze(output_data)
        
        # Load the labels
        with open(label_file, 'r') as f:
            labels = {i: line.strip() for i, line in enumerate(f.readlines())}
        
        # Get the top 5 predictions
        top_categories = results.argsort()[-5:][::-1]
        
        print(f"\nInference completed in {inference_time*1000:.2f}ms")
        print("\nTop 5 predictions:")
        for i, idx in enumerate(top_categories):
            if idx < len(labels):
                print(f"  {i+1}: {labels[idx]} ({results[idx]*100:.2f}%)")
            else:
                print(f"  {i+1}: Unknown category {idx} ({results[idx]*100:.2f}%)")
        
        return True
    
    except Exception as e:
        print(f"Error running inference: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Test Coral Edge TPU with image classification')
    parser.add_argument('--model', help='Path to the TFLite model')
    parser.add_argument('--labels', help='Path to the labels file')
    parser.add_argument('--image', help='Path to the test image')
    args = parser.parse_args()
    
    if args.model and args.labels and args.image:
        model_file = args.model
        label_file = args.labels
        image_file = args.image
    else:
        # Download test files if needed
        model_file, label_file, image_file = download_test_files()
    
    # Run inference
    success = run_inference(model_file, label_file, image_file)
    
    if success:
        print("\n✅ Coral USB Accelerator is working correctly!")
    else:
        print("\n❌ Test failed. Please check your setup and Edge TPU runtime installation.")

if __name__ == "__main__":
    main()
