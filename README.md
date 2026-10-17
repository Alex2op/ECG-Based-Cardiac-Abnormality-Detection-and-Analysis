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

This project uses the MIT-BIH Arrhythmia Database provided by PhysioNet.

**Dataset link:** https://physionet.org/content/mitdb/1.0.0/

### Dataset Setup

1. Download the required ECG records from PhysioNet.
2. Create a folder named `data` in the project root directory.
3. Place the required record files inside the `data` folder.
4. Ensure each record has its corresponding files, including `.dat`, `.hea`, and `.atr` where required by the code.

The project directory should look like this:

```text
ECG-Based-Cardiac-Abnormality-Detection-and-Analysis/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    ├── 100.dat
    ├── 100.hea
    ├── 100.atr
    └── ...
```

The filenames shown above are examples. Download the records required by the program and preserve their original filenames.

**Note:** The ECG dataset files are not included in this repository. Users must obtain the records separately from PhysioNet.

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
