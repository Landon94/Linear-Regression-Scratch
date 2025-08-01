# Linear-Regression-Scratch


## Overview
This project uses linear regression to predict student performance based on several features:
- Hours Studied
- Previous Scores
- Extracurricular Activities
- Sleep Hours
- Sample Question Papers Practiced

> **Note**: the data the model was trained on is synthetic
<br>

## Visualization of Features vs Performance
<img width="1500" height="900" alt="Image" src="https://github.com/user-attachments/assets/3395d1a9-3a9d-4d56-8db7-9bbc9bc5c23b" />
The graph shows linear relationships between hours studied and previous scores with performance while other features show more scattered distributions.

<br><br><br>

## Visualizatoin of Featues vs Performance with Normalization
<img width="1500" height="900" alt="Image" src="https://github.com/user-attachments/assets/d4125c1e-7f3f-41d4-aa8a-a679eaf58568" />
The graph shows the same data after appling z-score normalization where each feature now has a mean of 0 and a standard deviation of 1. Normalization helps to prevent large valued features from over influencing the model and improve efficiency.

<br>

---

## Evaluation

Model trained with the following parameters:
- Learning Rate = 0.0027
- Iterations = 25000
- Lambda = 0.001

<br>
<img width="640" height="480" alt="Image" src="https://github.com/user-attachments/assets/7760df67-3f63-4d5e-9e36-5d3e0e8287d6" />

<br><br>

Performance Metrics on Test Set:

- **Mean Squared Error (MSE)**: 4.21

- **R² Score**: 0.99

> MSE measures the averages squared difference between the predictions and actual y values.

> R² measeures the portion of variance in y that can be explained by x.

<br>


<img width="640" height="480" alt="Image" src="https://github.com/user-attachments/assets/6853e952-2dfb-4bb6-af33-7f941a4b2806" />

---

## Acknowledgements
Dataset: https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression



