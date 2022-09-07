# Research about DL with ophthalmology

Contains a several group of algs used to segment parts of retina. For now, the used of archtectures like U-Net are under test and improvements, for leverage results under some metrics.

# Repo structure

This repo was created to structure some scripts, where is separate by different responsibilities. 

# Venv environment

- `python3 -m venv venv` → creates a virtual environment
- `venv\Scripts\activate.bat` → activates the environment (windows)
- `source venv/bin/activate` → activates the environment (mac)
- `pip install -r requirements.txt` → for install all dependences
- `pip freeze > requirements.txt` → when the development was to finish

# TODOs

# Dependencies

- TensorFlow
- Keras
- PIL
- OpenCv
- Numpy
- Matplotlib
- segmentation models (https://github.com/qubvel/segmentation_models)
- albumentations