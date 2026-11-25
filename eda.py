
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

sns.set_theme(style="whitegrid")

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "eda_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(PROJECT_DIR / "bank_customer_churn.csv")

print("=" * 60)
print("       BANK CUSTOMER CHURN - EXPLORATORY DATA ANALYSIS")
print("=" * 60)


# --------------------------------------------------
# 2. BASIC DATASET INFORMATION
# --------------------------------------------------

print("\n--- Dataset Overview ---")

print("Dataset shape:", df.shape)

print("\nFirst five records:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe())


# --------------------------------------------------
# 3. CHURN DISTRIBUTION
# --------------------------------------------------

print("\n--- Churn Distribution ---")

churn_counts = df["Exited"].value_counts()
churn_percentage = df["Exited"].value_counts(normalize=True) * 100

retained_count = churn_counts.get(0, 0)
churned_count = churn_counts.get(1, 0)
churn_rate = churn_percentage.get(1, 0)

print("Retained customers:", retained_count)
print("Churned customers:", churned_count)
print("Churn rate:", round(churn_rate, 2), "%")

plt.figure(figsize=(7, 5))

ax = sns.countplot(
    data=df,
    x="Exited",
    hue="Exited",
    order=[0, 1],
    hue_order=[0, 1],
    palette={0: "seagreen", 1: "tomato"},
    legend=False
)

ax.set_xticks([0, 1])
ax.set_xticklabels(["Retained", "Churned"])

# Add customer counts above bars
for container in ax.containers:
    ax.bar_label(container, fmt="%d", padding=3)

plt.title("Customer Churn Distribution")
plt.xlabel("Customer Status")
plt.ylabel("Number of Customers")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "01_churn_distribution.png", dpi=300)
plt.close()


# --------------------------------------------------
# 4. CHURN BY AGE GROUP
# --------------------------------------------------

print("\n--- Churn by Age Group ---")

df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[17, 25, 35, 45, 55, 65, 90],
    labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]
)

age_churn = (
    df.groupby("AgeGroup", observed=False)["Exited"].mean() * 100
)

print(age_churn.round(2))

plt.figure(figsize=(9, 5))

ax = sns.barplot(
    x=age_churn.index,
    y=age_churn.values,
    color="steelblue"
)

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.1f%%",
        padding=3
    )

plt.title("Churn Rate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Churn Rate (%)")
plt.ylim(0, max(age_churn.values) + 10)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "02_churn_by_age.png", dpi=300)
plt.close()


# --------------------------------------------------
# 5. CHURN BY GEOGRAPHY
# --------------------------------------------------

print("\n--- Churn by Geography ---")

geo_churn = (
    df.groupby("Geography")["Exited"].mean() * 100
)

print(geo_churn.round(2))

plt.figure(figsize=(8, 5))

ax = sns.barplot(
    x=geo_churn.index,
    y=geo_churn.values,
    color="cornflowerblue"
)

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.1f%%",
        padding=3
    )

plt.title("Churn Rate by Geography")
plt.xlabel("Geography")
plt.ylabel("Churn Rate (%)")
plt.ylim(0, max(geo_churn.values) + 10)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "03_churn_by_geography.png", dpi=300)
plt.close()


# --------------------------------------------------
# 6. CHURN BY ACTIVE MEMBERSHIP
# --------------------------------------------------

print("\n--- Churn by Active Membership ---")

activity_churn = (
    df.groupby("IsActiveMember")["Exited"].mean() * 100
)

print(activity_churn.round(2))

plt.figure(figsize=(7, 5))

ax = sns.barplot(
    x=activity_churn.index,
    y=activity_churn.values,
    color="mediumpurple"
)

ax.set_xticks([0, 1])
ax.set_xticklabels(["Inactive", "Active"])

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.1f%%",
        padding=3
    )

plt.title("Churn Rate by Membership Activity")
plt.xlabel("Membership Activity")
plt.ylabel("Churn Rate (%)")
plt.ylim(0, max(activity_churn.values) + 10)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "04_churn_by_activity.png", dpi=300)
plt.close()


# --------------------------------------------------
# 7. CHURN BY NUMBER OF PRODUCTS
# --------------------------------------------------

print("\n--- Churn by Number of Products ---")

# Calculate total customers, churned customers and churn rate
product_analysis = df.groupby("NumOfProducts")["Exited"].agg(
    Total_Customers="count",
    Churned_Customers="sum",
    Churn_Rate="mean"
)

product_analysis["Churn_Rate"] = (
    product_analysis["Churn_Rate"] * 100
)

print(product_analysis.round(2).to_string())

plt.figure(figsize=(8, 5))

ax = sns.barplot(
    x=product_analysis.index,
    y=product_analysis["Churn_Rate"],
    color="darkorange"
)

# Display churn rate and customer count
for i, row in enumerate(product_analysis.itertuples()):
    ax.text(
        i,
        row.Churn_Rate + 1,
        f"{row.Churn_Rate:.1f}%\n(n={row.Total_Customers})",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.title("Churn Rate by Number of Banking Products")
plt.xlabel("Number of Banking Products")
plt.ylabel("Churn Rate (%)")
plt.ylim(0, max(product_analysis["Churn_Rate"]) + 12)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "05_churn_by_products.png", dpi=300)
plt.close()


# --------------------------------------------------
# 8. NUMERICAL FEATURE DISTRIBUTIONS
# --------------------------------------------------

print("\n--- Numerical Feature Distributions ---")

numerical_features = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "EstimatedSalary"
]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

axes = axes.flatten()

for i, column in enumerate(numerical_features):

    if column == "Tenure":

        # Tenure is discrete (values 0 to 10)
        sns.histplot(
            data=df,
            x=column,
            kde=False,
            discrete=True,
            ax=axes[i],
            color="steelblue"
        )

    else:

        # KDE is used for continuous numerical variables
        sns.histplot(
            data=df,
            x=column,
            kde=True,
            ax=axes[i],
            color="steelblue"
        )

    axes[i].set_title(f"Distribution of {column}")

axes[-1].axis("off")

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "06_numerical_distributions.png", dpi=300)
plt.close()


# --------------------------------------------------
# 9. OUTLIER ANALYSIS
# --------------------------------------------------

print("\n--- Outlier Analysis ---")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

outlier_features = [
    "CreditScore",
    "Age",
    "Balance"
]

for i, column in enumerate(outlier_features):

    sns.boxplot(
        data=df,
        y=column,
        ax=axes[i],
        color="lightblue"
    )

    axes[i].set_title(f"Outlier Analysis: {column}")

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "07_outlier_analysis.png", dpi=300)
plt.close()


# --------------------------------------------------
# 10. CORRELATION ANALYSIS
# --------------------------------------------------

print("\n--- Correlation Analysis ---")

correlation_features = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Exited"
]

correlation_matrix = df[
    correlation_features
].corr()

print(
    correlation_matrix["Exited"]
    .sort_values(ascending=False)
    .round(4)
)

plt.figure(figsize=(11, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "08_correlation_heatmap.png", dpi=300)
plt.close()


# --------------------------------------------------
# 11. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 60)
print("              EDA COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Total customers:", len(df))
print("Total churned:", df["Exited"].sum())

print(
    "Overall churn rate:",
    round(df["Exited"].mean() * 100, 2),
    "%"
)

print("\nCharts saved in:", OUTPUT_DIR.resolve())

print("=" * 60)
