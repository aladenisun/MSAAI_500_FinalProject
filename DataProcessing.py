import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import patsy as pt
from scipy import stats

RETAINED_COLS = [
    "tech_company",
    "workplace_resources", "mh_employer_discussion", "medical_coverage",
    "mh_share", "age", "gender"
]

class DataProcessor:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df: pd.DataFrame | None = None

    def load_datafile(self) -> pd.DataFrame:
        """Loads the dataset"""
        self.df = pd.read_csv(self.filepath)
        return self.df

    def keep_relevant_columns(self) -> pd.DataFrame:
        assert self.df is not None, "Call load() first."
        self.df = self.df[RETAINED_COLS].copy()
        return self.df

    def filter_tech(self) -> pd.DataFrame:
        assert self.df is not None, "Call load() first."
        self.df = self.df[self.df["tech_company"] == "Yes"].copy()
        return self.df

    def preprocess_data(self):
        """Handles missing values and filters for tech companies
            Encodes binary variables and scales age """


        # Outcome: High comfort (>=7 on a 1-10 scale)
        # Utilized in Regression Models
        self.df['high_comfort'] = np.where(self.df['mh_share'] >= 7, 1, 0)

        # Encode predictors
        # Treat "I don't know" as a separate category instead of dropping
        self.df['resources_binary'] = self.df['workplace_resources'].map({'Yes':1,'No':0})
        self.df['resources_binary'] = self.df['resources_binary'].fillna(-1)  # -1 for "I don't know"

        self.df['employer_binary'] = self.df['mh_employer_discussion'].map({'Yes':1,'No':0})
        self.df['employer_binary'] = self.df['employer_binary'].fillna(-1)

        self.df['coverage_binary']  = self.df['medical_coverage'].map({'Yes':1,'No':0})
        self.df['coverage_binary']  = self.df['coverage_binary'].fillna(-1)

        # combined support (at least 2 supports)
        self.df['combined_support'] = ((self.df['resources_binary'] +
                                self.df['employer_binary'] +
                                self.df['coverage_binary']) >= 2).astype(int)

        # Encode gender as binary for modeling (Female=1, Male=0, Others=-1)
        self.df['gender_binary'] = self.df['gender'].map({'Female':1, 'Male':0}).fillna(-1)

        # Optional: Normalize age (helps logistic regression stability)
        self.df['age_scaled'] = (self.df['age'] - self.df['age'].mean()) / self.df['age'].std()

        # Drop any missing values
        self.df = self.df.dropna(subset=[
                    "mh_share", "age", "resources_binary", "employer_binary",
                    "coverage_binary", "gender_binary", "age_scaled", "high_comfort"
                ]).copy()


        return self.df