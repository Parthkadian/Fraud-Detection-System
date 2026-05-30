from sklearn.model_selection import train_test_split
from src.utils.config_loader import load_yaml_file


class DataSplitter:
    def __init__(self):
        self.config = load_yaml_file("configs/config.yaml")

    def split(self, df):
        target = self.config["project"]["target_column"]

        X = df.drop(columns=[target])
        y = df[target]

        test_size = self.config["split"]["test_size"]
        valid_size = self.config["split"]["valid_size"]

        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=(test_size + valid_size),
            random_state=42,
            stratify=y
        )

        relative_test_size = test_size / (test_size + valid_size)

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=relative_test_size,
            random_state=42,
            stratify=y_temp
        )

        return X_train, X_val, X_test, y_train, y_val, y_test