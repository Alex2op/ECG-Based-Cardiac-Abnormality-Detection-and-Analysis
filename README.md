# ECG-Based Cardiac Abnormality Detection and Analysis

## Overview

This project uses Python, signal processing, and machine learning to analyze electrocardiogram (ECG) signals. It classifies ECG beat patterns using the MIT-BIH Arrhythmia Database and provides ECG visualization, model predictions, classification metrics, and heart-rate estimation.

## Features

- ECG signal visualization
- ECG record and annotation processing
- Machine learning-based ECG beat classification
- Comparison of predicted and actual beat labels
- Model training and validation reports
- Heart-rate estimation
- Basic experimental risk-status output

## Technologies Used

- Python 3.12
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn
- WFDB

## Dataset

The project uses the MIT-BIH Arrhythmia Database available through PhysioNet.

Dataset link: https://physionet.org/content/mitdb/1.0.0/

The ECG records must be available using the data-loading method implemented in the source code.

## Installation

1. Install Python 3.12.
2. Clone or download this repository.
3. Open a terminal in the project directory.
4. Install the required dependencies:

   `python -m pip install -r requirements.txt`

5. Run the program:

   `python main.py`

Replace `main.py` with the actual Python filename if necessary.

## Limitations

This project is an academic machine learning prototype. Its performance depends on the dataset, preprocessing, class distribution, and evaluation method. Reported test results do not establish clinical reliability.

The experimental risk-status output and heart-rate estimate are not intended for medical diagnosis or treatment decisions.

## Disclaimer

This project is intended for educational and research purposes only. It is not a medical device and does not replace evaluation by a qualified healthcare professional.
