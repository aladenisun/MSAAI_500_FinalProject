import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import patsy as pt
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import chi2_contingency


def preprocess_data():
    df = pd.read_csv("data.csv")

    # Standardize column names (important to avoid case/space issues)
    df.columns = df.columns.str.strip().str.lower()

    # Filter to tech employees only
    df_tech = df[df['tech_company'].str.strip().str.title() == 'Yes']

    # Keep only the two columns we need
    df_filtered = df_tech[['mh_share', 'workplace_resources']].dropna()

    print(df_filtered.head())

    return df