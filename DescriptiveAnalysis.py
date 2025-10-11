from DataProcessing import pd, np, sns, plt, pt
# ------------------------------------
# Exploratory Data Analysis
# ------------------------------------

class DescriptiveAnalisis:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def descriptive_stats(self) -> pd.DataFrame:
      """ Generate the descriptive statistics"""
      rcols = [
            "mh_share", "age", "high_comfort",
            "resources_binary", "employer_binary", "coverage_binary",
            "combined_support", "gender_binary", "age_scaled"
        ]
      table = self.df[rcols].describe().T
      # Clean up formatting
      table = table.rename(columns={
         "count": "N", "mean": "Mean", "std": "SD", "min": "Min",
         "25%": "Q1", "50%": "Median", "75%": "Q3", "max": "Max"
         })
      # Helpful rounding
      return table.round({
         "N": 0, "Mean": 3, "SD": 3, "Min": 3, "Q1": 3, "Median": 3, "Q3": 3, "Max": 3
         })

    def descriptive_visualization(self):
       """ Creates the descriptive plots and visual representations of the data"""
       # Table 1: Descriptive statistics
       # Gender × Workplace Resources interaction summary
       gender_resource = (
          self.df.groupby(['gender', 'workplace_resources'])
          .agg(
             mean_mh_share = ('mh_share', 'mean'),
             n =('mh_share', 'count'),
             high_comfort_pct=('high_comfort', lambda x: x.mean() * 100)
             ).round(2)
       )
       print(f"{gender_resource}")

       # Function to add percentages on bar plots
       def plot_count_with_pct(x, hue, title):
          ax = sns.countplot(x=x, hue=hue, data=self.df, palette="Set2")
          total = len(self.df)
          for p in ax.patches:
             height = p.get_height()
             ax.annotate(f'{100*height/total:.1f}%',
                         (p.get_x() + p.get_width()/2, height),
                         ha='center', va='bottom', fontsize=9)
          plt.title(title)
          plt.show()

       plot_count_with_pct("workplace_resources","high_comfort","Comfort vs Workplace Resources")
       plot_count_with_pct("mh_employer_discussion","high_comfort","Comfort vs Employer Discussion")
       plot_count_with_pct("medical_coverage","high_comfort","Comfort vs Medical Coverage")


       # Combine plots strictly for report
       sns.set(style="whitegrid", palette="pastel", font_scale=1.1)
       self.df["high_comfort"] = (self.df["mh_share"] >= 7).astype(int)
       # Define the categorical workplace support variables
       support_vars = ["workplace_resources", "mh_employer_discussion", "medical_coverage"]

       plt.figure(figsize=(15, 4))
       for i, var in enumerate(support_vars, 1):
          plt.subplot(1, 3, i)

          # Plot proportion of high comfort (bar height shows frequency of comfort by category)
          sns.countplot(x=self.df[var], hue=self.df["high_comfort"], palette="pastel",)

          plt.title(f"Comfort vs {var.replace('_', ' ').replace('mh', 'M.H').title()}")
          plt.xlabel(var.replace('_', ' ').replace('mh', 'Mental Health').title())
          plt.ylabel("Frequency")
          plt.legend(title="High Comfort", labels=["No (0)", "Yes (1)"])
          plt.xticks(rotation=30)
       plt.tight_layout()
       plt.show()

        # Boxplot comfort levels
       plt.figure(figsize=(6,4))
       sns.boxplot(x="workplace_resources", y="mh_share", data=self.df, hue="workplace_resources", palette="pastel", legend=False)
       plt.ylabel('Mental Health Share')
       plt.xlabel("workplace_resources".replace('_', ' ').replace('mh', 'Mental Health').title())
       plt.title("Comfort Scores by Workplace Resources")
       plt.show()

       # Histogram distribution
       plt.figure(figsize=(6,4))
       sns.histplot(self.df['mh_share'], bins=10, kde=False, color="skyblue", edgecolor="black")
       plt.title("Distribution of Comfort Sharing Scores")
       plt.xlabel("Comfort Sharing Score")
       plt.ylabel("Frequency")
       plt.show()

       # Stratified by gender (extra insight)
       plt.figure(figsize=(6,4))
       sns.barplot(x="gender", y="high_comfort", hue="workplace_resources", data=self.df, errorbar=None, palette="pastel")
       plt.title("High Comfort by Gender & Resources")
       plt.ylabel("High Comfort (%)")
       plt.xlabel("Gender")
       plt.show()

       # Histogram of Age
       plt.figure(figsize=(6,4))
       sns.histplot(self.df['age'], bins=20, color="skyblue", edgecolor='black')
       plt.title("Distribution of Age")
       plt.xlabel("Age")
       plt.ylabel("Frequency")
       plt.show()

       # Average Comfort Score by Age Group
       self.df['age_group'] = pd.cut(self.df['age'], bins=[18,30,40,50,60,80], labels=["18-29","30-39","40-49","50-59","60+"])
       sns.barplot(x="age_group", y="mh_share", data=self.df, color="skyblue", edgecolor='black', errorbar=None)
       plt.title("Average Comfort Score by Age Group")
       plt.ylabel("Mean Comfort Score")
       plt.xlabel("Age Group")
       plt.show()

       numeric_vars = ["mh_share", "age"]
       plt.figure(figsize=(10, 4))
       for i, var in enumerate(numeric_vars, 1):
          plt.subplot(1, 2, i)
          sns.histplot(self.df[var], bins=20, kde=True, color="skyblue", edgecolor='black')
          plt.title(f"Frequency Distribution of {var.replace('_', ' ').replace('mh', 'M.H').title()}")
          plt.xlabel(var.replace('_', ' ').replace('mh', 'Mental Health').title())
          plt.ylabel("Frequency")
       plt.tight_layout()
       plt.show()
