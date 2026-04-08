import pandas as pd


class DataCleaning:
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic cleaning: remove duplicates, handle missing values.
        """
        df = df.copy()

        # Remove duplicates
        df = df.drop_duplicates()

        # Fill missing values (if any)
        df = df.fillna(0)

        return df