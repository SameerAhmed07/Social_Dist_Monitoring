# Social Distancing Monitoring

## Overview

This project implements a real-time Social Distancing Monitoring using computer vision (opencv) and deep learning techniques. It uses a camera to detect people and measure the distance between them, alerting people when social distancing norms are violated.

## Project Design

![Social Distancing Detector Design](https://github.com/SameerAhmed07/Social_Dist_Monitoring/blob/main/social%20distance%20design.png)

The image above illustrates the high-level design and workflow of our Social Distancing Monitoring.

## Features

- Real-time person detection using YOLOv3
- Distance calculation between detected individuals
- Visual alerts for social distancing violations
- Audio warnings using text-to-speech

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- imutils
- pyttsx3
- YOLOv3 weights and configuration files

## Installation

1. Clone the repository:
   ```
   git clone git@github.com:SameerAhmed07/Social_Dist_Monitoring.git
   cd social-distancing-detector
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Download YOLOv3 weights and configuration files:
   - Download `yolov3.weights` from the official YOLO website
   - Place `yolov3.weights`, `yolov3.cfg`, and `coco.names` in the project directory

## Usage

1. Ensure your webcam is connected and functioning.

2. Run the main script:
   ```
   python main.py
   ```

3. The application will start using your default webcam. It will display the video feed with bounding boxes around detected people and alerts for social distancing violations.

4. Press 'q' to quit the application.

## Configuration

You can adjust the following parameters in `main.py`:

- `confidence`: Adjust the confidence threshold for person detection (default is 0.1)
- `distance_between_objects`: Modify the distance threshold for social distancing alerts (default is 220 pixels)

## How It Works

1. The script uses YOLOv3 to detect people in each frame of the video feed.
2. It calculates the distances between all detected individuals.
3. If the distance between any two people is less than the specified threshold, it draws red bounding boxes and displays a "Red Alert" message.
4. For people maintaining proper distance, green bounding boxes are drawn.
5. When a violation is detected, an audio warning is played using text-to-speech.

## Limitations

- Performance may vary depending on the hardware specifications.

## Contributing

Contributions to improve the project are welcome.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

- [YOLO: Real-Time Object Detection](https://docs.ultralytics.com/)
- [OpenCV community](https://docs.opencv.org/4.x/index.html)
- [pyttsx3](https://pyttsx3.readthedocs.io/en/latest/)
  

