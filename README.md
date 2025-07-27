# Deploying YOLOv7 on Kria KV260 for Real-Time License Plate Detection

This repository contains the source code for the project "Deploying a 
GPU-Accelerated YOLOv7 Model on Kria KV260," a comprehensive tutorial 
featured on Hackster.io.

This project is a collaboration between CUEE MDAP (Chulalongkorn University) 
and Design Gateway Co., Ltd.

## About This Project

This project provides a complete workflow for taking a GPU-trained YOLOv7 model, 
optimizing it with the Vitis AI v2.5 toolchain, and deploying it for 
high-performance, real-time inference on the AMD Kria KV260 Vision AI Starter Kit.

### Collaboration Partners

This work is a proud collaboration between:

- **CUEE MDAP**: The Multimedia Data Analytics and Processing (MDAP) Research Unit 
  from the Faculty of Engineering, Chulalongkorn University.
- **Design Gateway Co., Ltd.**: A premier Deep Tech company in Thailand specializing 
  in IP Core development for FPGAs and ASICs.

## Quick Start Guide

For detailed explanations, please refer to the 
https://www.hackster.io/cuee-mdap/deploying-a-gpu-accelerated-yolov7-model-on-kria-kv260-b26aa0 

### 1. Setup Vitis AI 2.5 Environment (Host PC)

```bash
# Clone the Vitis AI repository
git clone https://github.com/Xilinx/Vitis-AI.git
cd Vitis-AI

# Checkout the correct version for this project
git checkout v2.5

# Build the Docker image (use -t gpu for GPU support) ./docker_build.sh

# Run the Vitis AI 2.5 container
./docker_run.sh xilinx/vitis-ai-cpu:2.5
```

### 2. Quantize & Compile the Model

Inside the Vitis AI container, use the provided scripts to prepare the model.

**Calibrate the model:**
```bash
python yolov7_quantize.py \
    --mode calib \
    --build_dir /path/to/trained_model \
    --quant_mode calib
```

**Generate the quantized model:**
```bash
python yolov7_quantize.py \
    --mode test \
    --build_dir /path/to/trained_model \
    --quant_mode test
```

**Compile for the KV260 DPU:**
```bash
vai_c_xir \
    --xmodel /path/to/quantized_model/YOLOv7TinySmallPlus_int.xmodel \
    --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \
    --net_name yolov7_kv260 \
    --output_dir ./compiled_model
```

### 3. Deploy and Run on KV260

1. Copy the compiled `yolov7_kv260.xmodel` and the application script 
   `kv260_license_plate_detector.py` to the board using `scp`.

2. Connect to the board via SSH:
   ```bash
   ssh root@<KV260_IP_ADDRESS>
   ```

3. Run the application:
   ```bash
   python3 kv260_license_plate_detector.py \
       --model yolov7_kv260.xmodel \
       --camera
   ```

## File Structure

```
├── kv260_license_plate_detector.py  # Application to run on the KV260
├── yolov7_train.py                  # Script for model training
│   yolov7_quantize.py               # Script for model quantization
├── kv260-detection-result.png       # Example result image
└── README.md
```

## License

This project is licensed under the MIT License.
