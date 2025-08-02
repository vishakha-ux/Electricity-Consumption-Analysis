import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import os
os.makedirs("plots", exist_ok=True)

import logging
logging.basicConfig(level=logging.INFO)
logging.info("✅ Dataset loaded successfully.")

df = pd.read_csv("C:\\Users\\DELL\\Downloads\\electricity.csv")
print(df)
print(df.head)
print(df.columns)

# Rename columns for analysis
df = df.rename(columns={
    "Utility.State": "State",
    "Retail.Total.Revenue": "Revenue",
    "Retail.Total.Sales": "Sales"
})

# Remove rows with missing State values
df = df.dropna(subset=["State"])

# 1. Group by State-wise Revenue & Sales
state_summary = df.groupby("State")[["Revenue", "Sales"]].sum()

# Top 10 states by Revenue and Sales
top10_rev = state_summary.sort_values("Revenue", ascending=False).head(10)
top10_sales = state_summary.sort_values("Sales", ascending=False).head(10)

# Print Top 10 States
print("\nTop 10 States by Revenue:\n", top10_rev[["Revenue"]])
print("\nTop 10 States by Sales:\n", top10_sales[["Sales"]])
logging.info("✅ Top 10 revenue and sales states identified.")

# Bar Plot - Top 10 Revenue States
plt.figure(figsize=(10, 6))
sns.barplot(
    x=top10_rev["Revenue"],
    y=top10_rev.index,
    hue=top10_rev.index,
    palette="crest",
    legend=False
)
plt.title("Top 10 States by Retail Total Revenue")
plt.xlabel("Revenue (US $)")
plt.ylabel("State")
plt.tight_layout()
plt.savefig("top_10_revenue_states.png")
plt.show()

# 2.Bar Plot - Top 10 Sales States
plt.figure(figsize=(10, 6))
sns.barplot(
    x=top10_sales["Sales"],
    y=top10_sales.index,
    hue=top10_sales.index,
    palette="mako",
    legend=False
)
plt.title("Top 10 States by Retail Total Sales")
plt.xlabel("Sales (kWh)")
plt.ylabel("State")
plt.tight_layout()
plt.savefig("top_10_sales_states.png")
plt.show()

# Interactive Bar Chart - Revenue (Plotly)
fig = px.bar(
    top10_rev,
    x="Revenue",
    y=top10_rev.index,
    orientation="h",
    title="Top 10 States by Retail Total Revenue (Interactive)",
    labels={"Revenue": "Revenue (US $)", "index": "State"},
    text_auto=".2s"
)
fig.update_layout(yaxis=dict(autorange="reversed"))
fig.show()

# 3 .Revenue vs Sales Correlation & Scatter Plot
correlation = df["Revenue"].corr(df["Sales"])
print(f"\nCorrelation between Revenue and Sales: {correlation:.2f}")

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Sales", y="Revenue", color="teal")
plt.title("Revenue vs Sales (State-wise Data Points)")
plt.xlabel("Sales (kWh)")
plt.ylabel("Revenue (US $)")
plt.grid(True)
plt.tight_layout()
plt.savefig("revenue_vs_sales.png")
plt.show()

# 3.1 📍 Highest Revenue State
top_state, top_revenue = state_summary["Revenue"].idxmax(),state_summary["Revenue"].max()
print(f"📍 Highest Revenue State: {top_state} (${top_revenue:,.2f})")


# 4. ⃣ Clustering: High Sales + Low Revenue vs High Revenue + Low Sales

# Reset index for clustering
state_summary = state_summary.reset_index()

# Standardize Revenue & Sales for clustering
scaler = StandardScaler()
scaled_features = scaler.fit_transform(state_summary[["Revenue", "Sales"]])

# Apply KMeans Clustering (2 clusters)
kmeans = KMeans(n_clusters=2, random_state=42)
state_summary["Cluster"] = kmeans.fit_predict(scaled_features)

# Visualize Clusters using Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=state_summary,
    x="Sales", y="Revenue",
    hue="Cluster", palette="Set2",
    s=120
)
plt.title("Clustering: High Sales/Low Revenue vs High Revenue/Low Sales")
plt.xlabel("Total Sales (kWh)")
plt.ylabel("Total Revenue (US $)")
plt.grid(True)
plt.tight_layout()
plt.show()
logging.info("📊 Clustering complete.")

# Display clustered states
print("\n📊 Clustered State Summary:")
print(state_summary.sort_values("Cluster")[["State", "Sales", "Revenue", "Cluster"]])



# 4.1 🧾 Lowest Performing States (sabase kam rewanue karane wali 5 state) rewanue means bikri
bottom5_revenue = state_summary.sort_values("Revenue").head(5)

bottom5_revenue.plot(kind="barh", y="Revenue", legend=False, color="crimson")
plt.title("Bottom 5 States by Revenue")
plt.xlabel("Revenue")
plt.ylabel("State")
plt.tight_layout()
plt.show()

# 5. Revenue per Unit Sale (Efficiency)   (Kaunse states har unit (kWh) electricity par zyada paisa kama rahe hain?)
state_summary["Revenue_per_Unit"] = state_summary["Revenue"] / state_summary["Sales"]

# 5.1 Total Revenue & Sales Summary
total_revenue = df["Revenue"].sum()
total_sales = df["Sales"].sum()
print(f"Total Revenue: ₹{total_revenue:,.2f}")
print(f"Total Sales (kWh): {total_sales:,.2f}")

# 6. Distribution of Revenue (Boxplot / Histogram)
# Load data
df = pd.read_csv("electricity.csv")

# Clean data (remove nulls in revenue)
df = df[df["Retail.Total.Revenue"].notna()]

# Plot
plt.figure(figsize=(12, 5))

# Histogram + KDE
plt.subplot(1, 2, 1)
sns.histplot(df["Retail.Total.Revenue"], kde=True, color='skyblue')
plt.title("Revenue Distribution (Histogram + KDE)")

# Boxplot (for outliers)
plt.subplot(1, 2, 2)
sns.boxplot(x=df["Retail.Total.Revenue"], color='salmon')
plt.title("Revenue Distribution (Boxplot)")

plt.tight_layout()
plt.savefig("revenue_distribution.png")
plt.show()

# 7.Top 5 states ka Revenue % pie chart
top5 = state_summary.sort_values("Revenue", ascending=False).head(5)

total = top5["Revenue"].sum()
print(f"💰 Total Revenue of Top 5 States: ${total:,.2f}")

# Plotly Pie Chart
fig = px.pie(
    top5,
    values="Revenue",
    names=top5.index,
    title="💡 Top 5 States by Revenue Share",
    color_discrete_sequence=px.colors.sequential.Tealgrn,
    hole=0.3  # optional donut style
)
fig.update_traces(textinfo="percent+label")
fig.show()


# 8 Clustering States (High Sales + Low Revenue vs High Revenue + Low Sales → business insight ke liye)

# Group by State
grouped = df.groupby("Utility.State")[["Retail.Total.Revenue", "Retail.Total.Sales"]].sum().reset_index()

# Prepare data for clustering
X = grouped[["Retail.Total.Revenue", "Retail.Total.Sales"]]

# Apply KMeans Clustering (choose 3 clusters for clear separation)
kmeans = KMeans(n_clusters=3, random_state=42)
grouped["Cluster"] = kmeans.fit_predict(X)

# Visualize Clusters
plt.figure(figsize=(10,6))
sns.scatterplot(data=grouped, x="Retail.Total.Revenue", y="Retail.Total.Sales", hue="Cluster", palette="Set2", s=100)
for i, row in grouped.iterrows():
    plt.text(row["Retail.Total.Revenue"], row["Retail.Total.Sales"], row["Utility.State"], fontsize=8)

plt.title("Clustering of States: Revenue vs Sales")
plt.xlabel("Total Revenue")
plt.ylabel("Total Sales")
plt.grid(True)
plt.tight_layout()
plt.savefig("sales_revenue_clusters.png")
plt.show()

# 9. 🧮 State Contribution to Total Revenue (Har state ka % contribution)

# Group total revenue by State
state_summary = df.groupby("Utility.State")["Retail.Total.Revenue"].sum().reset_index()

# Calculate Total Revenue
total_revenue = state_summary["Retail.Total.Revenue"].sum()

# Add % Contribution column
state_summary["% Contribution"] = (state_summary["Retail.Total.Revenue"] / total_revenue) * 100

# Sort and show top 10 contributors
top_states = state_summary.sort_values(by="% Contribution", ascending=False).head(10)
print(top_states)

# Visualize as horizontal bar chart
plt.figure(figsize=(10,6))
sns.barplot(x=top_states["% Contribution"], y=top_states["Utility.State"], hue=top_states["Utility.State"], palette="viridis")
plt.title("Top 10 States by % Contribution to Total Revenue")
plt.xlabel("Percentage Contribution (%)")
plt.ylabel("State")
plt.tight_layout()
plt.savefig("top_10_contribution_states.png")
plt.show()

# 10. 🔍 Identify Outliers (Revenue & Sales) using IQR method

# Use the correct state_summary that has raw column names
state_outlier_df = df.groupby("Utility.State")[["Retail.Total.Revenue", "Retail.Total.Sales"]].sum().reset_index()

# Rename columns for clarity
state_outlier_df = state_outlier_df.rename(columns={
    "Utility.State": "State",
    "Retail.Total.Revenue": "Revenue",
    "Retail.Total.Sales": "Sales"
})

# IQR function for outlier detection
def detect_outliers(column):
    Q1 = state_outlier_df[column].quantile(0.25)
    Q3 = state_outlier_df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return state_outlier_df[(state_outlier_df[column] < lower) | (state_outlier_df[column] > upper)]

# Detect Revenue & Sales outliers
revenue_outliers = detect_outliers("Revenue")
sales_outliers = detect_outliers("Sales")

# Combine outliers & tag
outliers_combined = pd.concat([revenue_outliers, sales_outliers]).drop_duplicates(subset=["State"])
state_outlier_df["Outlier"] = state_outlier_df["State"].isin(outliers_combined["State"])
state_outlier_df["Outlier_Label"] = state_outlier_df["Outlier"].replace({True: "Outlier", False: "Normal"})

# Plotly scatter chart
fig = px.scatter(
    state_outlier_df,
    x="Sales",
    y="Revenue",
    color="Outlier_Label",
    hover_name="State",
    size_max=20,
    color_discrete_map={"Outlier": "red", "Normal": "green"},
    title="🔍 Revenue vs Sales (with Outliers Highlighted)"
)
fig.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
fig.update_layout(xaxis_title="Sales (kWh)", yaxis_title="Revenue (US $)")
fig.show()
logging.info("🧮 Outlier detection complete.")


state_outlier_df.to_csv("Electricity_State_Summary.csv", index=False)
print("✅ Final state-wise summary CSV saved as 'Electricity_State_Summary.csv'")
logging.info("✅ Final state-wise summary CSV saved.")
