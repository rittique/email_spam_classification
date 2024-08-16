# Spam Classifier

This Jupyter notebook (`spam_classifier.ipynb`) implements a machine learning pipeline to classify emails as spam or not spam. The pipeline includes data loading, preprocessing, feature extraction, and model training/testing.

## Contents

1. **Loading Libraries**: 
   - The notebook starts by importing the necessary libraries, including Pandas, NumPy, NLTK, Scikit-learn, and others used for data processing, feature extraction, and model training.

2. **Data Loading**:
   - The notebook assumes that the email data is stored in a directory called `data/`. The first file in this directory is loaded into a Pandas DataFrame.

3. **Helper Functions**:
   - Several helper functions are defined to preprocess the email data:
     - `extract_subject_and_email`: Splits the emails into `subjects` and `email_bodies`.
     - `email_check`: Identifies whether an email body contains an email address.
     - `clean_text`: Cleans and preprocesses text data by removing URLs, emails, phone numbers, punctuation, and stopwords, and then applies lemmatization.
     - `complete_preprocessing`: Combines all preprocessing steps and applies TF-IDF vectorization to the text data.

4. **Model Preparation**:
   - The preprocessed data is prepared for model input by combining the TF-IDF vectors of the `subjects` and `email_bodies` along with additional features.

5. **Model Training/Testing**:
   - The notebook includes code for training various machine learning models such as RandomForestClassifier, LogisticRegression, and SVM, and evaluating their performance.

## How to Run

1. **Environment Setup**:
   - Ensure you have the necessary Python libraries installed. You can do this by running:
     ```bash
     pip install -r requirement.txt
     ```
   - Additionally, download the required NLTK data by running:
     ```python
     import nltk
     nltk.download('stopwords')
     nltk.download('wordnet')
     ```

2. **Data Preparation**:
   - Place your email dataset (CSV format) inside a directory named `data/`. The notebook will automatically load the first file found in this directory.

3. **Running the Notebook**:
   - Open the notebook in Jupyter and run all cells sequentially. The notebook will preprocess the data, train the model, and provide classification results.

4. **Model Evaluation**:
   - After running the notebook, you will see the classification report and accuracy metrics for the models used.

## Notes

- Ensure that your dataset is in a format compatible with the notebook, with at least one column containing the raw email text.
- Modify the `data_path` variable if your dataset is stored in a different location.
