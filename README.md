# Face Recognition System - CV Final Project

This is a program that uses image processing and pattern recognition techniques to detect and recognize faces. Created by Tim Fitzpatrick and Nick Newton.

## Built With

- [Python](https://www.python.org/)
- [OpenCV](https://opencv.org/)

## How to run

Follow these steps to set up and run a local instance of this project on your machine.

### Prerequisites

Make sure Python and openCV are installed.

- Python installed with Homebrew
  ```sh
  brew install python
  ```
- OpenCV and virtual environment
  ```sh
  conda create --name name_your_env python==3.6.10
  conda install -c conda-forge opencv
  conda install -c anaconda numpy
  conda install -c conda-forge matplotlib
  ```

### Installation

Steps to use the face recognition system:

1. Clone the repo
   ```sh
   git clone git@github.com:tfitzpatrick0/CV-FaceRecognition.git
   ```
2. Move into the `CV-FaceRecognition` directoy
   ```sh
   cd CV-FaceRecognition
   ```
3. Run scripts

Use flags to specify which image sets to use:

      -t for training set
      -v for validation set
      -u for unknown set

```sh
python detection.py [This will go through filesystem and detect/adjust faces]
python trainer.py [This will create a model using lbph]
python recognition.py [This will go through each image and check it against the model to recognize the image and verify accuracy]
```

4. Utilizing Real-Time Detection and Recognition

Run realtime.py This takes about 100 pictures of you face and puts it into a folder with your name inside of the unknown file.
This is then added to the trained model and once trained, a new window should open to recognize your face in real-time.

```sh
python realtime.py
```
