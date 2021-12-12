# CV3 Nick Newton, Tim Fitzpatrick

# About The Project

This project is a tool to detect and recognize faces

## Built With
List of dependencies for project:

* [Python](https://www.python.org/)
* [OpenCV](https://opencv.org/)

## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

Make sure Python and openCV are installed.
* Python installed with Homebrew
  ```sh
  brew install python
  ```
* OpenCV and virtual environment
  ```sh
  conda create --name name_your_env python==3.6.10
  conda install -c conda-forge opencv
  conda install -c anaconda numpy 
  conda install -c conda-forge matplotlib
  ```

### Installation

Steps to get your project running:
1. Clone the repo
   ```sh
   git clone git@github.com:nnewton2/cv_face_recognition.git
   ```
3. Enter `Delivery3` Directory
   ```sh
   cd Delivery3
   ```
4. Run scripts
   ```sh
   python detection.py [This will go through filesystem and detect/adjust faces]
   python trainer.py [This will create a model using lbph]
   python recognition.py [This will go through each image and check it against the model to recognize the image and verify accuracy]
   ```
