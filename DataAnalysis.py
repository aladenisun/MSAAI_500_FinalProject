from DataProcessing import pd, np, sns, plt, pt
# ------------------------------------
# Exploratory Data Analysis
# ------------------------------------

def compute_descriptive_stats(data):
   # Load data
    df = data

    # Table 1: Descriptive statistics
    table1 = df.groupby('workplace_resources')['mh_share'].describe().round(2)
    print("\nTable 1. Descriptive statistics by resource access\n")
    print(table1)

    # Figure 1: Distribution of comfort sharing
    #plt.figure(figsize=(7,5))
    #plt.hist(df['mh_share'], bins=10, density=True, alpha=0.7)
    sns.histplot(df['mh_share'], bins=10, stat='density', kde=True, color="steelblue", line_kws={'color' : 'black', 'lw' : 2}, alpha=0.8)
    #plt.title("Figure 1. Distribution of Comfort Sharing Levels")
    plt.xlabel("Comfort Sharing Level")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.show()

    # Figure 2: Resource access counts
    plt.figure(figsize=(6,4))
    sns.countplot(x='workplace_resources', data=df, palette="colorblind")
    #plt.title("Figure 2. Access to Mental Health Resources")
    plt.xlabel("Access to Resources (0 = No, 1 = Yes)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Figure 3: Boxplot by resource access
    plt.figure(figsize=(7,5))
    sns.boxplot(x='workplace_resources', y='mh_share', data=df, palette="colorblind")
    #plt.title("Figure 3. Comfort Sharing by Access to Resources (Boxplot)")
    plt.xlabel("Access to Mental Health Resources")
    plt.ylabel("Comfort Sharing Level")
    plt.tight_layout()
    plt.show()

    # Figure 4: Mean comparison bar plot
    plt.figure(figsize=(7,5))
    mean_values = df.groupby('workplace_resources')['mh_share'].mean().reset_index()
    sns.barplot(x='workplace_resources', y='mh_share', data=mean_values, palette="colorblind")
    #plt.title("Figure 4. Mean Comfort Sharing by Resource Access")
    plt.xlabel("Access to Mental Health Resources")
    plt.ylabel("Mean Comfort Sharing Level")
    plt.tight_layout()
    plt.show()

    # Figure 5: Group means with error bars (SD)
    plt.figure(figsize=(7,5))
    summary = df.groupby('workplace_resources')['mh_share'].agg(['mean','std']).reset_index()
    plt.bar(summary['workplace_resources'], summary['mean'], yerr=summary['std'], capsize=5)
    #plt.title("Figure 5. Comfort Sharing Means with Standard Deviation")
    plt.xlabel("Access to Mental Health Resources (0=No, 1=Yes)")
    plt.ylabel("Mean Comfort Sharing Level")
    plt.tight_layout()
    plt.show()