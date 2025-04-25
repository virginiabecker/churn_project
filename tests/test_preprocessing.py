import unittest
import pandas as pd
from src.preprocessing import preprocess_data, split_data

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.raw_data = pd.DataFrame({
            'customerID': ['1', '2', '3'],
            'gender': ['Female', 'Male', 'Female'],
            'SeniorCitizen': [0, 1, 0],
            'Partner': ['Yes', 'No', 'No'],
            'Dependents': ['No', 'No', 'No'],
            'tenure': [1, 34, 2],
            'PhoneService': ['No', 'Yes', 'Yes'],
            'MultipleLines': ['No phone service', 'No', 'No'],
            'OnlineSecurity': ['No', 'Yes', 'No'],
            'OnlineBackup': ['Yes', 'No', 'Yes'],
            'DeviceProtection': ['No', 'Yes', 'No'],
            'TechSupport': ['No', 'No', 'No'],
            'StreamingTV': ['No', 'No', 'No'],
            'StreamingMovies': ['No', 'No', 'No'],
            'Contract': ['Month-to-month', 'One year', 'Month-to-month'],
            'PaperlessBilling': ['Yes', 'No', 'Yes'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Mailed check'],
            'MonthlyCharges': [29.85, 56.95, 53.85],
            'TotalCharges': ['29.85', '1889.5', '108.15'],
            'Churn': ['No', 'No', 'Yes']
        })

    def test_preprocess_data(self):
        X, y = preprocess_data(self.raw_data.copy())
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(y.name, 'Churn')
        self.assertEqual(X.shape[0], self.raw_data.shape[0])
        self.assertNotIn('customerID', X.columns)
        self.assertIn('gender_Male', X.columns)
        self.assertIn('MultipleLines_Yes', X.columns)
        self.assertIn('QtdServicos', X.columns)
        self.assertEqual(X['TotalCharges'].isnull().sum(), 0)
        self.assertEqual(y.dtype, 'int64')
        self.assertEqual(y.tolist(), [0, 0, 1])

    def test_split_data(self):
        X_processed, y_processed = preprocess_data(self.raw_data.copy())
        X_train, X_test, y_train, y_test = split_data(X_processed, y_processed, test_size=0.2, random_state=42, stratify=y_processed)
        self.assertIsInstance(X_train, pd.DataFrame)
        self.assertIsInstance(X_test, pd.DataFrame)
        self.assertIsInstance(y_train, pd.Series)
        self.assertIsInstance(y_test, pd.Series)
        self.assertEqual(X_train.shape[0], int(0.8 * self.raw_data.shape[0]))
        self.assertEqual(X_test.shape[0], int(0.2 * self.raw_data.shape[0]))
        self.assertEqual(y_train.shape[0], int(0.8 * self.raw_data.shape[0]))
        self.assertEqual(y_test.shape[0], int(0.2 * self.raw_data.shape[0]))
        self.assertEqual(y_train.value_counts(normalize=True).round(2).to_dict(), {0: 0.67, 1: 0.33})
        self.assertEqual(y_test.value_counts(normalize=True).round(2).to_dict(), {0: 0.5, 1: 0.5}) # Pequena diferença devido ao tamanho da amostra

if __name__ == '__main__':
    unittest.main()