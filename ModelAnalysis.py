from __future__ import annotations
from DataProcessing import pd, np, sns, plt, pt, stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

APA_P = lambda p: f"< .001" if p < 0.001 else f"= {p:.4f}"

@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

# Independent Samples Welch's T-Test
class ModelAnalysis:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def welch_test(self, a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
            md = a.mean() - b.mean()
            se = np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))
            ci = (md - 1.96*se, md + 1.96*se)
            return md, ci[0], ci[1]

    def ttest_report(self, group_col: str, target_col: str, label: str) -> dict:
        a = self.df.loc[self.df[group_col] == 1, target_col].dropna().values
        b = self.df.loc[self.df[group_col] == 0, target_col].dropna().values
        t, p = stats.ttest_ind(a, b, equal_var=False)
        md, lo, hi = self.welch_test(a, b)
        return {
            "Variable": label,
            "t(df)": f"{t:.3f}",   # (Welch's df omitted to avoid clutter; optional to add)
            "p": APA_P(p),
            "Mean Diff": f"{md:.2f}",
            "95% CI": f"[{lo:.2f}, {hi:.2f}]",
            "Significance": "Sig." if p < 0.05 else "Not Sig."
        }

    def ttest_table(self) -> pd.DataFrame:
        rows = []
        rows.append(self.ttest_report("workplace_resources_binary" if "workplace_resources_binary" in self.df.columns else "resources_binary", "mh_share", "Resources Binary"))
        rows.append(self.ttest_report("employer_binary", "mh_share", "Employer Discussion"))
        rows.append(self.ttest_report("coverage_binary", "mh_share", "Coverage Binary"))
        return pd.DataFrame(rows, columns=["Variable", "t(df)", "p", "Mean Diff", "95% CI", "Significance"])


    # LOGISTIC REGRESSION
    # ---------- Provide X/y split ----------
    def get_xy(self) -> tuple[pd.DataFrame, pd.Series]:
        assert self.df is not None, "Call build_features() first."
        X = self.df[[
            "resources_binary", "employer_binary", "coverage_binary",
            "combined_support", "gender_binary", "age_scaled"
        ]].copy()
        y = self.df["high_comfort"].astype(int).copy()
        return X, y

    def train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> Split:
        X, y = self.get_xy()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        return Split(Xtr, Xte, ytr, yte)

    def fit_improved_logistic(self, X: pd.DataFrame, y: pd.Series) -> LogisticRegression:
        model = LogisticRegression(max_iter=1000, solver="liblinear")
        model.fit(X, y)
        return model

    def coef_table(self, model: LogisticRegression, feature_names: list[str]) -> pd.DataFrame:
        coefs = model.coef_[0]
        df = pd.DataFrame({"Predictor": feature_names, "β": coefs, "OR": np.exp(coefs)})
        return df.sort_values("OR", ascending=False).reset_index(drop=True).round(3)

    def statsmodels_logit(self, X: pd.DataFrame, y: pd.Series):
        Xc = sm.add_constant(X)
        logit = sm.Logit(y, Xc)
        result = logit.fit(disp=False)
        summ = result.summary2().tables[1].copy()
        # Rename for APA friendliness
        summ = summ.rename(columns={"Coef.": "β", "Std.Err.": "SE", "P>|z|": "p", "[0.025": "CI Low", "0.975]": "CI High"})
        summ["OR"] = np.exp(summ["β"])
        # Order columns
        summ = summ[["β", "SE", "OR", "CI Low", "CI High", "p"]]
        return summ.round(3), result

    # ---------- MODEL COMPARISON ----------
    def evaluate(self, y_true, y_prob, y_pred) -> dict:
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1 Score": f1_score(y_true, y_pred, zero_division=0),
            "AUC": roc_auc_score(y_true, y_prob)
        }

    def compare_models(self, X_train, X_test, y_train, y_test) -> pd.DataFrame:
        # 1) Baseline logistic (simple logistic on intercept only -> predicts majority class probability)
        # We emulate your baseline behavior by fitting on a single constant column.
        X_train_base = np.ones((len(X_train), 1))
        X_test_base  = np.ones((len(X_test), 1))
        base = LogisticRegression(solver="liblinear")
        base.fit(X_train_base, y_train)
        prob_b = base.predict_proba(X_test_base)[:, 1]
        pred_b = (prob_b >= 0.5).astype(int)
        perf_base = self.evaluate(y_test, prob_b, pred_b)

        # 2) Improved Logistic (all features)
        log_imp = LogisticRegression(max_iter=1000, solver="liblinear")
        log_imp.fit(X_train, y_train)
        prob_l = log_imp.predict_proba(X_test)[:, 1]
        pred_l = (prob_l >= 0.5).astype(int)
        perf_log = self.evaluate(y_test, prob_l, pred_l)

        # 3) Random Forest
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2, random_state=42
        )
        rf.fit(X_train, y_train)
        prob_rf = rf.predict_proba(X_test)[:, 1]
        pred_rf = (prob_rf >= 0.5).astype(int)
        perf_rf = self.evaluate(y_test, prob_rf, pred_rf)

        # 4) Gradient Boosting
        gb = GradientBoostingClassifier(random_state=42)
        gb.fit(X_train, y_train)
        prob_gb = gb.predict_proba(X_test)[:, 1]
        pred_gb = (prob_gb >= 0.5).astype(int)
        perf_gb = self.evaluate(y_test, prob_gb, pred_gb)

        perf = pd.DataFrame.from_records([
            {"Model": "Baseline Logistic"} | perf_base,
            {"Model": "Improved Logistic"} | perf_log,
            {"Model": "Random Forest"} | perf_rf,
            {"Model": "Gradient Boosting"} | perf_gb,
        ])
        # Nice rounding/order
        cols = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "AUC"]
        return perf[cols].round(3).sort_values("AUC", ascending=False).reset_index(drop=True)