import pandas as pd


class DataInfo:

    def __init__(self, csv_path, target_col_index):
        df = pd.read_csv(csv_path)
        all_cols = df.columns
        self._csv_path = csv_path
        self._target_col = all_cols[target_col_index]
        self._feature_cols = [all_cols[i] for i in range(len(all_cols)) if ((i != target_col_index) and "Unnamed" not in all_cols[i])]

    def get_target_col(self):
        return self._target_col

    def get_feature_cols(self):
        return self._feature_cols

    def get_csv_path(self):
        return self._csv_path

    def __str__(self) -> str:
        return str(
            {"csv path": self._csv_path, "target columns": self._target_col, "feature columns": self._feature_cols})
