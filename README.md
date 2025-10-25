# Edge Computing Workshop Kigali 2025

![Cover Image](./asset/header-image.jpg)

Slide - [Link](https://docs.google.com/presentation/d/1E56lccdfK-RObEJX98kK9nOLPP6bARF_jRshlebHVts/edit?usp=sharing) 


### Introduction to Edge Impulse: 8 - 11
![Edge Impulse Image](./asset/edge-impulse-cover.png)


### Getting Started with Edge Impulse: 12 - 18 

![Welcome Image](./asset/welcome.png)

### Collecting Data 
<!-- ![Welcome Image](./asset/welcome.png) -->
- Image Classification 
- Object Detection 

### Building Model 
<!-- ![Welcome Image](./asset/welcome.png) -->

### Deploying on Edge Devices and Laptop 
<!-- ![Welcome Image](./asset/welcome.png) -->

- Create a python virtual environment 
```
python -m venv edge-env
```
- Activate the environment
```
george@Georges-MacBook-Pro edge-computing-workshop-kigali % source edge-env/bin/activate
(edge-env) george@Georges-MacBook-Pro edge-computing-workshop-kigali % 
```

- Install the requirements
```
pip install -r requirements.txt
```

```
(edge-env) george@Georges-MacBook-Pro image-classification % python3 camera_infer_h5.py -m models/model.h5 -l label.txt --width 320 --height 3
20 
```

### Working with Visual Language Models(Liquid AI - VLM) 
<!-- ![Welcome Image](./asset/welcome.png) -->

### Deploying VLM on Edge Device 
<!-- ![Welcome Image](./asset/welcome.png) -->


```
(base) george@Georges-MacBook-Pro models % edge-impulse-linux-runner

```

```
(base) george@Georges-MacBook-Pro models % edge-impulse-linux-runner --model-file edge-computing-workshop-2025-image-classification-mac-arm64-v4.eim 
[RUN] Starting the image classifier for Edge Impulse Experts / edge-computing-workshop-2025-image-classification (v4)
[RUN] Parameters image size 320x320 px (3 channels) classes [ 'Esp32', 'Jetson', 'Stm32' ]
[RUN] Thresholds: 42.min_score=0.001 (override via --thresholds <value>)
[RUN] Connected to camera MacBook Pro Camera

Want to see a feed of the camera and live classification in your browser? Go to http://192.168.1.106:4912

Want to use predictions in your application? Open a websocket to ws://192.168.1.106:4912

classifyRes 5ms. { Esp32: 0.2861, Jetson: 0.4963, Stm32: 0.2176 }
classifyRes 4ms. { Esp32: 0.3131, Jetson: 0.4614, Stm32: 0.2256 }
classifyRes 4ms. { Esp32: 0.3156, Jetson: 0.455, Stm32: 0.2294 }
classifyRes 3ms. { Esp32: 0.3205, Jetson: 0.4442, Stm32: 0.2353 }
classifyRes 3ms. { Esp32: 0.3231, Jetson: 0.4399, Stm32: 0.237 }
classifyRes 3ms. { Esp32: 0.326, Jetson: 0.4358, Stm32: 0.2382 }
classifyRes 3ms. { Esp32: 0.3256, Jetson: 0.4337, Stm32: 0.2407 }
classifyRes 3ms. { Esp32: 0.324, Jetson: 0.4297, Stm32: 0.2463 }
classifyRes 3ms. { Esp32: 0.3227, Jetson: 0.4289, Stm32: 0.2483 }
classifyRes 3ms. { Esp32: 0.3203, Jetson: 0.4291, Stm32: 0.2505 }
```

```
python3 camera_infer_h5.py -m models/model.h5 -l label.txt --width 320 --height 320
```

```
python camera_infer_tflite.py --model models/ei-edge-computing-workshop-2025-image-classification-classifier-tensorflow-lite-float32-model.3.lite --labels labels.txt --camera 0 --top_k 2
```

```
python camera_infer_tflite.py --model models/ei-edge-computing-workshop-2025-image-classification-classifier-tensorflow-lite-int8-quantized-model.3.lite --labels labels.txt --camera 0 --top_k 3
```

Quit
```
^CTraceback (most recent call last):
  File "/Users/george/Documents/github/edge-computing-workshop-kigali/image-classification/camera_infer_tflite.py", line 162, in <module>
    main()
  File "/Users/george/Documents/github/edge-computing-workshop-kigali/image-classification/camera_infer_tflite.py", line 154, in main
    key = cv2.waitKey(1) & 0xFF
          ^^^^^^^^^^^^^^
KeyboardInterrupt
^C
```

```
python batch_infer_images.py --model models/ei-edge-computing-workshop-2025-image-classification-classifier-tensorflow-lite-int8-quantized-model.3.lite --labels labels.txt --images_dir ./sample-directory
```

## Reference 
- [Edge Impulse]()
- [Edge Impulse with Tensorrt Jetson](https://docs.edgeimpulse.com/tools/libraries/sdks/inference/linux/cpp#tensorrt)
- [Image Classification]()
- [Edge Impulse Linux CLI](https://docs.edgeimpulse.com/tools/clis/edge-impulse-linux-cli#edge-impulse-linux-runner)
- [Edge Impulse Linux SDKs](https://docs.edgeimpulse.com/tools/libraries/sdks/inference/linux)
- [Edge Impulse NVIDIA Jetson](https://docs.edgeimpulse.com/hardware/boards/nvidia-jetson)
- [venv — Creation of virtual environments](https://docs.python.org/3/library/venv.html)
- [Object Detection]()
- [Visual Language Model - VLM]()
- [Inference with Qualcomm AI Accelerator on Particle Tachyon](https://www.hackster.io/naveenbskumar/inference-with-qualcomm-ai-accelerator-on-particle-tachyon-1c8888?f=1)
