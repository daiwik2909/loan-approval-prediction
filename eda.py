"""Stage 2 EDA for loan approval data. Usage: python eda.py data/raw/train.csv  (saves charts to reports/figures/)"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/train.csv"
OUT = Path("reports/figures"); OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(path)
TARGET = "loan_status"
NUM = ["person_age","person_income","person_emp_length","loan_amnt",
       "loan_int_rate","loan_percent_income","cb_person_cred_hist_length"]
CAT = ["person_home_ownership","loan_intent","loan_grade","cb_person_default_on_file"]

print("SHAPE", df.shape)
print("MISSING total", int(df.isna().sum().sum()), "| DUPLICATES (excluding id)", int(df.drop(columns="id").duplicated().sum()))
print("\nTARGET\n", df[TARGET].value_counts().to_string(), "\n", df[TARGET].value_counts(normalize=True).round(4).to_string())
print("\nSUMMARY\n", df[NUM].describe().T.round(2).to_string())

# outliers via IQR
print("\nOUTLIERS (IQR rule)")
for c in NUM:
    q1,q3 = df[c].quantile([.25,.75]); i=q3-q1
    n = int(((df[c]<q1-1.5*i)|(df[c]>q3+1.5*i)).sum())
    print(f"  {c:28s} {n:6d} ({n/len(df):.1%})")
print("  age>80:", int((df.person_age>80).sum()), "| emp_length>60:", int((df.person_emp_length>60).sum()),
      "| emp_length>age-14:", int((df.person_emp_length>df.person_age-14).sum()))

# feature-target relationships
print("\nTARGET RATE BY CATEGORY")
for c in CAT:
    print(df.groupby(c)[TARGET].agg(["mean","count"]).round(3).to_string(), "\n")
print("MEAN OF NUMERICS BY TARGET\n", df.groupby(TARGET)[NUM].mean().T.round(2).to_string())
print("\ninterest rate by grade\n", df.groupby("loan_grade").loan_int_rate.mean().round(2).to_string())
chk = (df.loan_amnt/df.person_income - df.loan_percent_income).abs()
print("\nloan_percent_income ~= loan_amnt/income for", f"{(chk<0.01).mean():.1%} of rows")
df["income_band"] = pd.qcut(df.person_income, 5)
print("\nrate by income quintile\n", df.groupby("income_band", observed=True)[TARGET].mean().round(3).to_string())
df["lpi_band"] = pd.cut(df.loan_percent_income,[-.01,.1,.2,.3,.4,1])
print("\nrate by loan_percent_income band\n", df.groupby("lpi_band", observed=True)[TARGET].agg(["mean","count"]).round(3).to_string())
corr = df[NUM+[TARGET]].corr()
print("\nCORR WITH TARGET\n", corr[TARGET].drop(TARGET).sort_values().round(3).to_string())

# plots
sns.set_theme(style="whitegrid")
ax = df[TARGET].value_counts().sort_index().plot.bar(color=["#4c78a8","#e45756"]); ax.set_title("Target distribution"); plt.tight_layout(); plt.savefig(OUT/"target.png",dpi=120); plt.close()
fig,axs = plt.subplots(2,4,figsize=(16,7))
for a,c in zip(axs.ravel(),NUM): sns.histplot(df[c],bins=40,ax=a); a.set_title(c)
axs.ravel()[-1].axis("off"); plt.tight_layout(); plt.savefig(OUT/"numeric_dist.png",dpi=120); plt.close()
fig,axs = plt.subplots(1,4,figsize=(18,4))
for a,c in zip(axs,CAT): df.groupby(c)[TARGET].mean().sort_values().plot.bar(ax=a,color="#4c78a8"); a.set_title(f"Target rate by {c}",fontsize=9); a.set_ylabel("rate")
plt.tight_layout(); plt.savefig(OUT/"categorical_target_rate.png",dpi=120); plt.close()
plt.figure(figsize=(8,6)); sns.heatmap(corr,annot=True,fmt=".2f",cmap="coolwarm",center=0); plt.title("Correlation"); plt.tight_layout(); plt.savefig(OUT/"correlation.png",dpi=120); plt.close()
fig,axs = plt.subplots(1,4,figsize=(18,4))
for a,c in zip(axs,["person_income","loan_int_rate","loan_percent_income","loan_amnt"]): sns.boxplot(data=df,x=TARGET,y=c,ax=a); a.set_title(c)
axs[0].set_yscale("log"); plt.tight_layout(); plt.savefig(OUT/"boxplots_by_target.png",dpi=120); plt.close()
print("\nSaved figures:", sorted(p.name for p in OUT.iterdir()))
