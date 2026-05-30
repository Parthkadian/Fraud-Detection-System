import pandas as pd
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from src.utils.config_loader import load_yaml_file
from src.monitoring.logger import setup_logger

class DriftMonitor:
    def __init__(self):
        self.config = load_yaml_file("configs/config.yaml")
        self.logger = setup_logger(self.config["paths"]["log_file"])
        self.reference_data_path = self.config["paths"]["raw_data"]
        self.report_dir = "reports"
        os.makedirs(self.report_dir, exist_ok=True)
        
        try:
            # We assume the raw data is the reference for training
            self.reference_df = pd.read_csv(self.reference_data_path)
            # Remove class for drift detection if it exists
            if "Class" in self.reference_df.columns:
                self.reference_df.drop(columns=["Class"], inplace=True)
        except Exception as e:
            self.logger.error(f"Failed to load reference data for drift monitoring: {e}")
            self.reference_df = None

    def generate_drift_report(self, production_data: list[dict], report_name="data_drift.html") -> str:
        """
        Generate Evidently Drift Report comparing production data with reference train data.
        """
        if self.reference_df is None:
            return "Error: Reference data not loaded."

        if not production_data:
            return "Error: No production data provided."

        prod_df = pd.DataFrame(production_data)
        
        # Ensure only overlapping columns are compared
        common_cols = list(set(self.reference_df.columns) & set(prod_df.columns))
        
        if not common_cols:
            return "Error: No common columns to compare."

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=self.reference_df[common_cols], current_data=prod_df[common_cols])
        
        report_path = os.path.join(self.report_dir, report_name)
        report.save_html(report_path)
        
        self.logger.info(f"Drift report generated at {report_path}")
        return report_path
