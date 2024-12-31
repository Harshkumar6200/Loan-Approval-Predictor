## Loan Approval Predictor

#### Description
The Loan Approval Predictor is a project designed to assist users in determining whether their loan application is likely to be approved or not. By providing various personal and financial information, users can receive a prediction regarding the approval status of their loan application.

#### Functionality
- **Input Parameters**: Users are prompted to input the following information:
  1. Number of dependents
  2. Education level (graduated or not)
  3. Self-employed status (Yes or No)
  4. Annual income
  5. Loan amount
  6. Loan term
  7. Cibil Score (credit score)
  8. Value of residential assets
  9. Value of commercial assets
  10. Value of luxury assets
  11. Value of bank assets
  12. Choice of classification algorithm (Random Forest Classifier, Bagging Classifier, Gradient Boosting, Decision Tree Classifier)

- **Prediction**: Based on the user's input, the model predicts whether the loan will be approved or not.

#### Implementation Steps
1. **Data Preprocessing**: Clean and preprocess the dataset to handle missing values and categorical variables.
2. **Splitting Data**: Divide the dataset into independent variables (X) and the target variable (y).
3. **Scaling Data**: Standardize the numerical features using the Standard Scaler.
4. **Train-Test Split**: Split the dataset into training and testing sets.
5. **Balancing Data**: Implement Random Oversampling to balance the imbalanced dataset.
6. **Model Training**: Train the classification models (Random Forest Classifier, Bagging Classifier, Gradient Boosting, Decision Tree Classifier) using the training data.
7. **Generating Predictions**: Generate predictions for all classification models using the testing data.
8. **Model Evaluation**: Evaluate the performance of each model using relevant metrics.
9. **Saving Models**: Save the best performing models using the pickle library for future use.
10. **Deployment**: Deploy the project using Streamlit to provide an interactive interface for users.

#### Usage
1. Run the application.
2. Input the required information as prompted.
3. Select the desired classification algorithm.
4. Receive the loan approval prediction.

#### Technologies Used
- Python
- Pandas
- Scikit-learn
- Streamlit
- Pickle

#### Note
This project is intended for informational purposes only and should not be considered as financial advice. The predictions provided are based solely on the data provided by the user and the performance of the selected classification algorithm. Users should consult with financial professionals for personalized advice regarding loan applications.

URL: https://loan-approval-predictor-k5happghix66kcgwn95ca86.streamlit.app/
