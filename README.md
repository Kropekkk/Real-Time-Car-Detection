# Real-Time-Car-Detection

Simple Real Time Car Detection app using feed from Youtube livestream to count vehicles.

![RTcar](https://github.com/Kropekkk/Real-Time-Car-Detection/RTcar.gif)

## Dependencies
This project uses custom trained [MobileNet-SSD model](https://github.com/tensorflow/models/blob/master/research/object_detection/configs/tf2/ssd_mobilenet_v2_320x320_coco17_tpu-8.config) (to collect data run ```collect.py``` )

* Python 3.9.13
* OpenCV
* Tensorflow (Tensorflow GPU if using Nvidia GPU)
* VidGear (provides support for pipelining live video-frames from YouTube)

## Usage

1. Create virtual environment ```python -m venv enviro```
2. Activat the virtual environment```.\enviro\Scripts\activate```
3. Install dependencies ```pip install -r requirements.txt```
4. Run main.py using ```python main.py```

## Performance

I tested my custom trained model using NVIDIA GeForce GTX 1070, CUDA 11.2.0. The results are given below

![Results](https://github.com/Kropekkk/Real-Time-Car-Detection/fpsResults.png)
