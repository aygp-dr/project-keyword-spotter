#!/usr/bin/env python3

import subprocess
import os
import sys
import re

def check_coral_device():
    """Check if Coral USB Accelerator is connected and accessible."""
    # Check using lsusb for the Global Unichip Corp device (Coral)
    try:
        result = subprocess.run(['lsusb'], capture_output=True, text=True)
        if '1a6e:089a' in result.stdout:
            # Extract bus and device numbers for detailed information
            match = re.search(r'Bus (\d+) Device (\d+): ID 1a6e:089a', result.stdout)
            if match:
                bus_num, dev_num = match.groups()
                device_path = f'/dev/bus/usb/{bus_num}/{dev_num}'
                print(f"✓ Coral USB Accelerator found: Bus {bus_num}, Device {dev_num}")
                print(f"  Device path: {device_path}")
                
                # Check if the device is accessible
                if os.path.exists(device_path):
                    print(f"✓ Device is accessible at system level")
                else:
                    print(f"✗ Device path {device_path} not found")
                    
            else:
                print("✓ Coral USB Accelerator found in USB devices")
            return True
        else:
            print("✗ Coral USB Accelerator not found in USB devices")
            return False
    except Exception as e:
        print(f"Error checking USB devices: {e}")
        return False

def check_edge_tpu_runtime():
    """Check if Edge TPU runtime is installed and accessible."""
    try:
        # Check if libedgetpu.so exists
        lib_paths = [
            '/usr/lib/libedgetpu.so.1',
            '/usr/local/lib/libedgetpu.so.1',
            '/lib/libedgetpu.so.1'
        ]
        
        for path in lib_paths:
            if os.path.exists(path):
                print(f"✓ Edge TPU library found at {path}")
                return True
        
        print("✗ Edge TPU library (libedgetpu.so.1) not found")
        return False
    except Exception as e:
        print(f"Error checking Edge TPU runtime: {e}")
        return False

def check_tflite_runtime():
    """Check if TFLite Runtime with Edge TPU support is available."""
    try:
        import tflite_runtime
        print(f"✓ TFLite Runtime version {tflite_runtime.__version__} is installed")
        
        # Check for Edge TPU support
        try:
            from tflite_runtime.interpreter import load_delegate
            print("✓ TFLite Runtime has delegate support (needed for Edge TPU)")
            
            # Try to load the Edge TPU delegate
            try:
                delegate = load_delegate('libedgetpu.so.1')
                print("✓ Successfully loaded Edge TPU delegate!")
                return True
            except Exception as e:
                print(f"✗ Failed to load Edge TPU delegate: {e}")
                return False
                
        except (ImportError, AttributeError):
            print("✗ TFLite Runtime doesn't have proper Edge TPU support")
            return False
    except ImportError:
        print("✗ TFLite Runtime is not installed")
        print("  To install: pip install tflite-runtime")
        return False

if __name__ == "__main__":
    print("Checking for Coral USB Accelerator...\n")
    
    # Check if device is connected
    device_connected = check_coral_device()
    print("")
    
    # Check if runtime is installed
    runtime_installed = check_edge_tpu_runtime()
    print("")
    
    # Check if TFLite runtime is properly set up
    tflite_setup = check_tflite_runtime()
    
    if device_connected and runtime_installed and tflite_setup:
        print("\n✅ Success! Your Coral USB Accelerator is fully set up and ready to use!")
    elif not device_connected:
        print("\n❌ Coral USB Accelerator is not detected.")
        print("Please check your USB connections and power supply.")
    elif not runtime_installed:
        print("\n❌ Edge TPU runtime library is not installed.")
        print("Please install it with:")
        print("  echo 'deb https://packages.cloud.google.com/apt coral-edgetpu-stable main' | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list")
        print("  curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -")
        print("  sudo apt-get update")
        print("  sudo apt-get install libedgetpu1-std")
    elif not tflite_setup:
        print("\n❌ TensorFlow Lite runtime is not properly set up for Edge TPU.")
        print("Please install the correct version with:")
        print("  pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_aarch64.whl")
    
    # Exit with appropriate status code
    sys.exit(0 if (device_connected and runtime_installed and tflite_setup) else 1)
