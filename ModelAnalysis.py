from __future__ import annotations
from DataProcessing import pd, np, sns, plt, pt, stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_validate

from dataclasses import dataclass
from typing import Dict, Tuple, List

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
    # Provide X/y split
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

    # MODEL COMPARISON
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

    # Fit models
    def fit_all_models(
        self,
        X_train: pd.DataFrame, y_train: pd.Series
    ) -> Dict[str, object]:
        models = {}

        # Baseline logistic (intercept-only emulation via single constant feature)
        base = LogisticRegression(solver="liblinear")
        base.fit(np.ones((len(X_train), 1)), y_train)
        models["Baseline Logistic"] = base

        # Improved logistic
        log_imp = LogisticRegression(max_iter=1000, solver="liblinear")
        log_imp.fit(X_train, y_train)
        models["Improved Logistic"] = log_imp

        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, random_state=42
        )
        rf.fit(X_train, y_train)
        models["Random Forest"] = rf

        # Gradient Boosting
        gb = GradientBoostingClassifier(random_state=42)
        gb.fit(X_train, y_train)
        models["Gradient Boosting"] = gb

        return models

    # Predictions / metrics
    def predict_all(
        self,
        models: Dict[str, object],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: float = 0.5
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        probs, preds = {}, {}
        for name, mdl in models.items():
            if name == "Baseline Logistic":
                # Use intercept-only fitted on constant feature; mirror test-shape
                p = mdl.predict_proba(np.ones((len(X_test), 1)))[:, 1]
            else:
                p = mdl.predict_proba(X_test)[:, 1]
            probs[name] = p
            preds[name] = (p >= threshold).astype(int)
        return probs, preds

    def performance_table(
        self, y_true: pd.Series, probs: Dict[str, np.ndarray], preds: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        rows = []
        for name in probs.keys():
            row = {
                "Model": name,
                "Accuracy": accuracy_score(y_true, preds[name]),
                "Precision": precision_score(y_true, preds[name], zero_division=0),
                "Recall": recall_score(y_true, preds[name], zero_division=0),
                "F1": f1_score(y_true, preds[name], zero_division=0),
                "AUC": roc_auc_score(y_true, probs[name]),
            }
            rows.append(row)
        df = pd.DataFrame(rows).sort_values("AUC", ascending=False)
        return df[["Model", "Accuracy", "Precision", "Recall", "F1", "AUC"]].round(3).reset_index(drop=True)

    # ROC comparison plot
    def plot_roc_comparison(
        self, y_true: pd.Series, probs: Dict[str, np.ndarray],
        title: str = "ROC Curve Comparison"
    ):
        plt.figure(figsize=(6, 6))
        for name, p in probs.items():
            fpr, tpr, _ = roc_curve(y_true, p)
            auc = roc_auc_score(y_true, p)
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()


    # Confusion matrices
    def plot_confusion_matrices(
        self, y_true: pd.Series, preds: Dict[str, np.ndarray]
    ):
        for name, yhat in preds.items():
            cm = confusion_matrix(y_true, yhat)
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Pred: Low","Pred: High"],
                yticklabels=["True: Low","True: High"])

            ax.set_title(f"Confusion Matrix — {name}", fontsize=14, fontweight="bold", pad=15)
            ax.set_xlabel("Predicted Label", fontsize=12)
            ax.set_ylabel("True Label", fontsize=12)
            plt.tight_layout()
            plt.show()

    # Feature importance tables
    def feature_importance_tables(
        self, models: Dict[str, object], feature_names: List[str]
    ) -> pd.DataFrame:
        rows = []
        for name, mdl in models.items():
            if name == "Baseline Logistic":
                continue
            if hasattr(mdl, "feature_importances_"):
                importances = mdl.feature_importances_
            elif hasattr(mdl, "coef_"):
                # absolute standardized weight proxy for importance
                importances = np.abs(mdl.coef_[0])
            else:
                continue
            for feat, imp in zip(feature_names, importances):
                rows.append({"Model": name, "Feature": feat, "Importance": float(imp)})
        fi = pd.DataFrame(rows)
        fi["Importance"] = fi["Importance"].round(4)
        return fi.sort_values(["Model", "Importance"], ascending=[True, False]).reset_index(drop=True)

    # Train vs Test summaries
    def train_test_summary(
        self, models: Dict[str, object],
        X_train: pd.DataFrame, y_train: pd.Series,
        X_test: pd.DataFrame, y_test: pd.Series,
        threshold: float = 0.5
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # TRAIN
        probs_tr, preds_tr = self.predict_all(models, X_train, y_train, threshold)
        train_tbl = self.performance_table(y_train, probs_tr, preds_tr)
        train_tbl = train_tbl.rename_axis(None)

        # TEST
        probs_te, preds_te = self.predict_all(models, X_test, y_test, threshold)
        test_tbl = self.performance_table(y_test, probs_te, preds_te)
        test_tbl = test_tbl.rename_axis(None)

        return train_tbl, test_tbl

    # Cross-validation (AUC + more)
    def cross_validation_summary(
        self, X: pd.DataFrame, y: pd.Series, cv_splits: int = 5, seed: int = 42
    ) -> pd.DataFrame:
        """
        CV for Improved Logistic, Random Forest, Gradient Boosting.
        Baseline: predicts 1 for all in test folds (prevalence baseline).
        """
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)

        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "auc": "roc_auc",
        }

        # Improved Logistic
        log_imp = LogisticRegression(max_iter=1000, solver="liblinear")
        cv_log = cross_validate(log_imp, X, y, cv=skf, scoring=scoring, n_jobs=None)

        # Random Forest
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=42)
        cv_rf = cross_validate(rf, X, y, cv=skf, scoring=scoring, n_jobs=None)

        # Gradient Boosting
        gb = GradientBoostingClassifier(random_state=42)
        cv_gb = cross_validate(gb, X, y, cv=skf, scoring=scoring, n_jobs=None)

        # Baseline Regression
        accs, precs, recs, f1s, aucs = [], [], [], [], []
        for train_idx, test_idx in skf.split(X, y):
            y_te = y.iloc[test_idx]
            y_hat = np.ones_like(y_te)
            accs.append(accuracy_score(y_te, y_hat))
            precs.append(precision_score(y_te, y_hat, zero_division=0))
            recs.append(recall_score(y_te, y_hat, zero_division=0))
            f1s.append(f1_score(y_te, y_hat, zero_division=0))
            # Baseline has no probabilities; AUC undefined — set to np.nan
            aucs.append(np.nan)

        def mean_std(col):
            return f"{np.mean(col):.3f} ± {np.std(col):.3f}"

        rows = [
            {"Model": "Baseline Logistic",
             "Accuracy": mean_std(accs), "Precision": mean_std(precs),
             "Recall": mean_std(recs), "F1": mean_std(f1s), "AUC": "—"},
            {"Model": "Improved Logistic",
             "Accuracy": mean_std(cv_log['test_accuracy']),
             "Precision": mean_std(cv_log['test_precision']),
             "Recall": mean_std(cv_log['test_recall']),
             "F1": mean_std(cv_log['test_f1']),
             "AUC": mean_std(cv_log['test_auc'])},
            {"Model": "Random Forest",
             "Accuracy": mean_std(cv_rf['test_accuracy']),
             "Precision": mean_std(cv_rf['test_precision']),
             "Recall": mean_std(cv_rf['test_recall']),
             "F1": mean_std(cv_rf['test_f1']),
             "AUC": mean_std(cv_rf['test_auc'])},
            {"Model": "Gradient Boosting",
             "Accuracy": mean_std(cv_gb['test_accuracy']),
             "Precision": mean_std(cv_gb['test_precision']),
             "Recall": mean_std(cv_gb['test_recall']),
             "F1": mean_std(cv_gb['test_f1']),
             "AUC": mean_std(cv_gb['test_auc'])},
        ]
        return pd.DataFrame(rows, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "AUC"])