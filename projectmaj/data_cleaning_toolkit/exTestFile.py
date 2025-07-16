import unittest
import pandas as pd
from data_cleaning_toolkit import fill_missing_with_mean, fill_missing_with_median

class TestMissingDataHandling(unittest.TestCase):

    def setUp(self):
        """Create a sample dataframe for testing."""
        self.df = pd.DataFrame({
            'A': [1, 2, None, 4, 5],
            'B': [5, None, 1, 2, 3],
            'C': ['x', 'y', 'z', None, 'w']
        })

    def test_fill_missing_with_mean(self):
        """Test filling missing numerical data with the mean."""
        df_filled = fill_missing_with_mean(self.df.copy(), 'A')
        self.assertEqual(df_filled.isnull().sum()['A'], 0)
        self.assertEqual(df_filled.at[2, 'A'], 3.0)  # Expected mean of column A

    def test_fill_missing_with_median(self):
        """Test filling missing numerical data with the median."""
        df_filled = fill_missing_with_median(self.df.copy(), 'B')
        self.assertEqual(df_filled.isnull().sum()['B'], 0)
        self.assertEqual(df_filled.at[1, 'B'], 2.5)  # Expected median of column B

    def test_fill_missing_with_mean_nonexistent_column(self):
        """Test handling of a non-existent column for mean filling."""
        with self.assertRaises(SystemExit):
            fill_missing_with_mean(self.df.copy(), 'D')  # D does not exist

    def test_fill_missing_with_median_nonexistent_column(self):
        """Test handling of a non-existent column for median filling."""
        with self.assertRaises(SystemExit):
            fill_missing_with_median(self.df.copy(), 'D')  # D does not exist

if __name__ == '__main__':
    unittest.main()
