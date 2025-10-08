from DataProcessing import pd, np, sns, plt, pt, smf, stats
from DataProcessing import preprocess_data

# Independent Samples Welch's T-Test
def ttest_resources_vs_comfort(data):
    """
    Performs an independent samples t-test comparing mean comfort sharing
    between employees WITH and WITHOUT access to mental health resources,
    restricted to tech employees.

    Parameters:
        data (DataFrame): Original dataset.

    Returns:
        dict: Results including t-statistic, p-value, confidence interval,
              mean difference, and effect size.
    """

    df = data.copy()
    df.columns = df.columns.str.strip().str.lower()

    df = df[df['workplace_resources'].isin(['Yes', 'No'])]
    df['res_access'] = (df['workplace_resources'] == 'Yes').astype(int)
    df = df[['mh_share', 'res_access']].dropna()

    yes = df.loc[df['res_access'] == 1, 'mh_share']
    no = df.loc[df['res_access'] == 0, 'mh_share']

    # Use the Welch’s method to compute t-test
    t_stat, p_val = stats.ttest_ind(yes, no, equal_var=False)

    # Compute confidence interval and effect size
    s1, s0 = yes.std(ddof=1), no.std(ddof=1)
    n1, n0 = len(yes), len(no)
    se_diff = np.sqrt(s1**2/n1 + s0**2/n0)

    df_num = (s1**2/n1 + s0**2/n0)**2
    df_den = ((s1**2/n1)**2/(n1-1)) + ((s0**2/n0)**2/(n0-1))
    df_welch = df_num / df_den
    t_crit = stats.t.ppf(0.975, df_welch)

    mean_diff = yes.mean() - no.mean()
    ci_low = mean_diff - t_crit * se_diff
    ci_high = mean_diff + t_crit * se_diff

    # Step 4: Print summary
    print("\n--- Independent Samples t-Test (Tech Employees Only) ---")
    print(f"t = {t_stat:.3f},  df ≈ {df_welch:.1f},  p = {p_val:.4f}")
    print(f"Mean difference (Yes - No) = {mean_diff:.3f}")
    print(f"95% CI = [{ci_low:.3f}, {ci_high:.3f}]")
    print("\nInterpretation: "
          "A small to moderate effect suggests that employees with access "
          "to mental health resources tend to report higher comfort sharing levels "
          "if the mean difference is positive.\n")

    # Step 5: Visualization
    plt.figure(figsize=(7,5))
    sns.boxplot(x='res_access', y='mh_share', data=df)
    plt.xticks([0,1], ['No Access', 'Access'])
    plt.xlabel('Access to mental health resources')
    plt.ylabel('Comfort sharing level')
    plt.title('Comfort Sharing by Access to Resources (Tech Employees)')
    plt.tight_layout()
    plt.show()

    return {
        't_stat': t_stat,
        'p_value': p_val,
        'df': df_welch,
        'mean_diff': mean_diff,
        'ci_low': ci_low,
        'ci_high': ci_high,
    }


 # Linear Regression Model
def linear_regression_model(data):
    """
    Fits a simple linear regression model predicting comfort sharing
    from access to mental health resources for tech employees.

    Parameters:
        data : Filtered dataset.

    Returns:
        Regression summary i.e. a statsmodels object
    """

    df = data.copy()
    df.columns = df.columns.str.strip().str.lower()

    df['workplace_resources'] = df['workplace_resources'].str.strip().str.title()
    df = df[df['workplace_resources'].isin(['Yes', 'No'])]
    df['res_access'] = (df['workplace_resources'] == 'Yes').astype(int)
    df = df[['mh_share', 'res_access']].dropna()

    model = smf.ols('mh_share ~ res_access', data=df).fit()
    print("\n--- Linear Regression: Comfort Sharing ~ Access to Resources ---")
    print(model.summary())
    print("\nInterpretation: "
          "The correlation coefficient for 'res_access' represents how much higher the mean "
          "comfort sharing level is for employees with access to mental health resources "
          "versus to those without. A positive, significant beta supports the hypothesis.\n")

    # Regression visualization
    plt.figure(figsize=(7,5))
    sns.regplot(x='res_access', y='mh_share', data=df, logistic=False,
                x_jitter=0.1, scatter_kws={'alpha':0.3})
    plt.xticks([0,1], ['No Access', 'Access'])
    plt.xlabel('Access to mental health resources')
    plt.ylabel('Comfort sharing level')
    plt.title('Regression of Comfort Sharing on Resource Access')
    plt.tight_layout()
    plt.show()

    return model