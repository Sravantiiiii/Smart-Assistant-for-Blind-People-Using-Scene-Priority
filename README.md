# Smart Assistant for Blind People Using Scene Priority

## Project Overview

Smart Assistant for Blind People Using Scene Priority is an AI-powered assistive system designed to help visually impaired people understand their surrounding environment.

The system combines object detection, depth estimation, scene understanding, and voice-based assistance to provide useful information about nearby objects and their relative positions.

## Features

- Real-time object detection using YOLOv11
- Depth estimation using Depth Anything V2
- Scene understanding using an LLM
- Voice-based assistance using text-to-speech
- Detection of objects in the surrounding environment
- Scene priority to provide relevant information to the user

## Technologies Used

- Python
- YOLOv11
- Depth Anything V2
- Groq API
- Large Language Model (LLM)
- Text-to-Speech
- OpenCV
- COCO Dataset

## Project Workflow

1. The camera captures the surrounding environment.
2. YOLOv11 detects objects in the scene.
3. Depth Anything V2 estimates the depth of the detected objects.
4. The system combines object and depth information.
5. The scene information is processed to identify important objects.
6. The LLM helps generate meaningful scene descriptions.
7. The output is converted into speech to assist the user.

## Project Structure

| File | Description |
|---|---|
| `main.py` | Main program for running the system |
| `stereo_depth_yolo11.py` | Object detection and stereo depth processing |
| `llmIntegration.py` | LLM integration for scene understanding |
| `stereo_calibrate.py` | Stereo camera calibration |
| `extract_frames.py` | Extracts frames from video |
| `camtest.py` | Camera testing |
| `testing.py` | Testing components of the project |
| `test1.py` | Additional testing |
| `requirements.txt` | Python dependencies required for the project |
| `installation.txt` | Installation and setup instructions |
| `left.mp4` | Sample left-camera video |
| `right.mp4` | Sample right-camera video |

## Dataset

The project uses the COCO dataset and pretrained computer vision models for object detection.

## Installation

1. Clone this repository.
2. Install the required Python packages from `requirements.txt`.
3. Follow the instructions provided in `installation.txt`.
4. Configure the required API credentials.
5. Run the main Python program.

## Applications

This project can be used as an assistive technology concept for helping visually impaired users gain better awareness of their surroundings through AI-based visual and audio assistance.

## Future Scope

- Improve object detection accuracy
- Improve depth estimation in different environments
- Add more intelligent scene prioritization
- Improve real-time performance
- Support additional assistive features and devices

## Authors

Developed as a major academic project in the field of Artificial Intelligence and Machine Learning.
