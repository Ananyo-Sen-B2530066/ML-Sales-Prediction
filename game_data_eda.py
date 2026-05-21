# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 20:26:26 2025

@author: arka8
"""

# =====================================
# 🎯 Exploratory Data Analysis (EDA)
# =====================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load dataset ---
df = pd.read_csv("vgsales.csv")

# --- Basic overview ---
print("✅ Dataset loaded successfully!")
print("\nShape of dataset:", df.shape)
print("\nColumns:\n", df.columns.tolist())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Data Types ---")
print(df.dtypes)

# --- Basic stats for numeric columns ---
print("\n--- Summary Statistics ---")
print(df.describe())

# =====================================
# 🔍 Missing Values and Duplicates
# =====================================

missing_percent = (df.isna().sum() / len(df)) * 100
print("\n--- Missing Values (%): ---")
print(missing_percent[missing_percent > 0])

duplicate_rows = df.duplicated().sum()
print(f"\nDuplicate Rows: {duplicate_rows}")

# =====================================
# 📊 Univariate Analysis
# =====================================

# --- Distribution of Global Sales ---
plt.figure(figsize=(8,5))
sns.histplot(df["Global_Sales"], bins=40, kde=True)
plt.title("Distribution of Global Sales")
plt.xlabel("Global Sales (millions)")
plt.ylabel("Frequency")
plt.show()

# --- Log-transformed target ---
plt.figure(figsize=(6,4))
sns.histplot(np.log1p(df["Global_Sales"]), bins=40, kde=True, color="green")
plt.title("Log-Transformed Global Sales Distribution")
plt.xlabel("log(Global Sales + 1)")
plt.show()

# --- Year distribution ---
plt.figure(figsize=(10,5))
sns.histplot(df["Year"].dropna(), bins=30, kde=False, color="coral")
plt.title("Game Release Year Distribution")
plt.xlabel("Year")
plt.ylabel("Number of Games")
plt.show()

# --- Top 10 Genres ---
plt.figure(figsize=(8,5))
df["Genre"].value_counts().head(10).plot(kind="bar", color="teal")
plt.title("Top 10 Genres by Game Count")
plt.ylabel("Number of Games")
plt.show()

# --- Top 10 Publishers ---
plt.figure(figsize=(8,5))
df["Publisher"].value_counts().head(10).plot(kind="bar", color="purple")
plt.title("Top 10 Publishers by Game Count")
plt.ylabel("Number of Games")
plt.show()

# --- Top 10 Platforms ---
plt.figure(figsize=(8,5))
df["Platform"].value_counts().head(10).plot(kind="bar", color="orange")
plt.title("Top 10 Platforms by Game Count")
plt.ylabel("Number of Games")
plt.show()

# =====================================
# 📈 Bivariate Analysis
# =====================================

# --- Global Sales by Genre ---
plt.figure(figsize=(10,5))
sns.boxplot(x="Genre", y="Global_Sales", data=df)
plt.xticks(rotation=45)
plt.title("Global Sales Distribution by Genre")
plt.show()

# --- Global Sales by Platform ---
plt.figure(figsize=(10,5))
top_platforms = df["Platform"].value_counts().nlargest(10).index
sns.boxplot(x="Platform", y="Global_Sales", data=df[df["Platform"].isin(top_platforms)])
plt.xticks(rotation=45)
plt.title("Global Sales Distribution by Platform")
plt.show()

# --- Global Sales by Publisher (Top 10) ---
plt.figure(figsize=(12,5))
top_publishers = df["Publisher"].value_counts().nlargest(10).index
sns.boxplot(x="Publisher", y="Global_Sales", data=df[df["Publisher"].isin(top_publishers)])
plt.xticks(rotation=45)
plt.title("Global Sales Distribution by Publisher")
plt.show()

# --- Correlation between numeric features ---
plt.figure(figsize=(8,6))
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap (Numerical Features)")
plt.show()

# =====================================
# 🧮 Feature Relationships
# =====================================

# Average sales per Genre
genre_sales = df.groupby("Genre")["Global_Sales"].mean().sort_values(ascending=False)
plt.figure(figsize=(8,5))
genre_sales.plot(kind="bar", color="steelblue")
plt.title("Average Global Sales per Genre")
plt.ylabel("Average Global Sales")
plt.show()

# Average sales per Platform
platform_sales = df.groupby("Platform")["Global_Sales"].mean().sort_values(ascending=False)
plt.figure(figsize=(8,5))
platform_sales.head(15).plot(kind="bar", color="slateblue")
plt.title("Average Global Sales per Platform (Top 15)")
plt.ylabel("Average Global Sales")
plt.show()

# =====================================
# 🕹️ Outlier Detection
# =====================================
plt.figure(figsize=(6,4))
sns.boxplot(x=df["Global_Sales"])
plt.title("Outlier Detection in Global Sales")
plt.show()

# =====================================
# ⏳ Trend Analysis: Global Sales Over Years
# =====================================
yearly_sales = df.groupby("Year")["Global_Sales"].sum()
plt.figure(figsize=(10,5))
sns.lineplot(x=yearly_sales.index, y=yearly_sales.values, marker='o')
plt.title("Total Global Sales Over Years")
plt.xlabel("Year")
plt.ylabel("Total Global Sales (millions)")
plt.show()

# =====================================
# 💾 Optional Insights
# =====================================
print("\nHighest selling game:\n", df.loc[df["Global_Sales"].idxmax()])
print("\nTop 5 Years by Total Sales:\n", yearly_sales.sort_values(ascending=False).head(5))
print("\nTop 5 Genres by Average Sales:\n", genre_sales.head(5))
