import unittest
import pandas as pd
from src.data_loader import load_data
import os

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Criar um arquivo CSV de teste temporário
        self.test_data = pd.DataFrame({
            'customerID': ['1', '2'],
            'gender': ['Female', 'Male'],
            'Churn': ['Yes', 'No']
        })
        self.test_csv_path = 'temp_test_churn.csv'
        self.test_data.to_csv(self.test_csv_path, index=False)

    def tearDown(self):
        # Remover o arquivo CSV de teste temporário
        if os.path.exists(self.test_csv_path):
            os.remove(self.test_csv_path)

    def test_load_data_success(self):
        df = load_data(self.test_csv_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (2, 3))
        self.assertEqual(list(df.columns), ['customerID', 'gender', 'Churn'])
        self.assertEqual(df['customerID'].tolist(), ['1', '2'])

    def test_load_data_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_data('non_existent_file.csv')

if __name__ == '__main__':
    unittest.main()