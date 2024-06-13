# Personality Detection

This repository contains the implementation of a personality detection model that uses a machine learning approach to classify individuals into different personality types based on their responses to a questionnaire. The project leverages a Gradient Boosting Classifier to achieve accurate personality predictions.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Personality detection is a valuable tool in psychology, HR, marketing, and other fields. This project aims to classify individuals' personality types based on questionnaire responses using machine learning techniques. The Gradient Boosting Classifier is employed to achieve high accuracy in predictions.

## Features

- Preprocess the dataset by encoding categorical variables and scaling numerical features.
- Split the dataset into training and testing sets.
- Train a Gradient Boosting Classifier with hyperparameter tuning using GridSearchCV.
- Evaluate the model's performance using accuracy score.
- Save the best model for future use.

## Installation

To run this project, you need to have Python and the required libraries installed. You can install the necessary packages using the following command:

```bash
pip install numpy pandas scikit-learn joblib
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/personality-detection.git
   cd personality-detection
   ```

2. Place your dataset in the desired directory. Update the path in the script if necessary.

3. Run the script:
   ```bash
   python personality_detection.py
   ```

## Model Training and Evaluation

The model training involves the following steps:
1. Loading and preprocessing the dataset.
2. Encoding categorical variables and scaling numerical features.
3. Splitting the data into training and testing sets.
4. Training a Gradient Boosting Classifier using GridSearchCV for hyperparameter tuning.
5. Evaluating the model's performance using the accuracy score.
6. Saving the best-performing model for future use.

The script will output the best hyperparameters found and the accuracy of the model on the test data.

## Contributing

Contributions are welcome! If you have any ideas or improvements, feel free to submit a pull request. Please ensure your contributions adhere to the repository's guidelines.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to explore the code and use the provided script to detect personalities from questionnaire responses. If you encounter any issues or have suggestions for improvement, please open an issue or submit a pull request.
