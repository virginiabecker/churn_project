import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import preprocess_data
from src.model_training import train_decision_tree, train_xgboost_model
from xgboost import XGBClassifier

class TestModelTraining(unittest.TestCase):
    def setUp(self):
        self.raw_data = pd.DataFrame({
            'gender': ['Female', 'Male'],
            'SeniorCitizen': [0, 1],
            'Partner': ['Yes', 'No'],
            'Dependents': ['No', 'No'],
            'tenure': [1, 34],
            'PhoneService': ['No', 'Yes'],
            'MultipleLines': ['No phone service', 'No'],
            'OnlineSecurity': ['No', 'Yes'],
            'OnlineBackup': ['Yes', 'No'],
            'DeviceProtection': ['No', 'Yes'],
            'TechSupport': ['No', 'No'],
            'StreamingTV': ['No', 'No'],
            'StreamingMovies': ['No', 'No'],
            'Contract': ['Month-to-month', 'One year'],
            'PaperlessBilling': ['Yes', 'No'],
            'PaymentMethod': ['Electronic check', 'Mailed check'],
            'MonthlyCharges': [29.85, 56.95],
            'TotalCharges': ['29.85', '1889.5'],
            'Churn': ['No', 'No']
        })
        X, y = preprocess_data(self.raw_data.copy())
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def test_train_decision_tree(self):
        model = train_decision_tree(self.X_train, self.y_train)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, DecisionTreeClassifier)

    def test_train_xgboost_model(self):
        model = train_xgboost_model(self.X_train, self.y_train)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, XGBClassifier)

if __name__ == '__main__':
    unittest.main()