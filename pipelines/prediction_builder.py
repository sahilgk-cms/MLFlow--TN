import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple
from data.schema import DATE_COL_WEEK_START, TARGET_COL, DISTRICT_COL, SUB_DISTRICT_COL

def build_prediction_data(predictions: pd.Series,
                          X_test_meta: pd.DataFrame,
                          best_cv_score: float,
                          test_score: float,
                          metric_name: str) -> pd.DataFrame:
    prediction_df = pd.DataFrame({
        DATE_COL_WEEK_START: X_test_meta[DATE_COL_WEEK_START],
        'predicted_case_count': predictions,
        'prediction_date': datetime.now().strftime('%Y-%m-%d'),
        f'best_cv_{metric_name}': best_cv_score,
        f"test_{metric_name}": test_score,
        SUB_DISTRICT_COL: X_test_meta[SUB_DISTRICT_COL],
        #DISTRICT_COL: X_test_meta[DISTRICT_COL],
        TARGET_COL: X_test_meta[TARGET_COL]
    })
    return prediction_df



def calc_high_risk_cases(prediction_df: pd.DataFrame,
                         high_risk_limit: int) -> pd.DataFrame:
    prediction_df['cases_rounded'] = prediction_df['predicted_case_count'].round(0) 
    prediction_df['mae'] = abs(prediction_df['cases_rounded'] - prediction_df[TARGET_COL])
    prediction_df['high_risk'] = np.where(prediction_df[TARGET_COL] > high_risk_limit, 1, 0)
    prediction_df['high_risk_pred'] = np.where(prediction_df['cases_rounded'] > high_risk_limit, 1, 0)
    prediction_df['correct'] = (prediction_df['high_risk'] == prediction_df['high_risk_pred']).astype(int)
    return prediction_df


def calc_precision_recall(prediction_df: pd.DataFrame) -> Tuple[float, float]:
    tp = ((prediction_df["high_risk"] == 1) & (prediction_df["high_risk_pred"] == 1)).sum()
    fp = ((prediction_df["high_risk"] == 0) & (prediction_df["high_risk_pred"] == 1)).sum()
    fn = ((prediction_df["high_risk"] == 1) & (prediction_df["high_risk_pred"] == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return precision, recall