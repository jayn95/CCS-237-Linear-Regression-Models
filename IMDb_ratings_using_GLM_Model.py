import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from statsmodels.genmod.families import Gaussian, Gamma
from statsmodels.genmod.families.links import identity, log

# --- Load dataset ---
df = pd.read_csv("imdb_dataset_cleaned.csv")

# Example: Scatter Plot between Runtime (X) and IMDB_Rating (Y)
x_col = 'Released_Year'
y_col = 'IMDB_Rating'

# ---------- Matplotlib Version ----------
plt.figure(figsize=(7, 5))
plt.scatter(df[x_col], df[y_col], color='blue', s=50)  # s = point size
plt.title(f'Scatter Plot of {x_col} vs {y_col}')
plt.xticks(rotation=90)
# plt.locator_params(axis='x', nbins=15)
plt.xlabel(f'{x_col} values')
plt.ylabel(f'{y_col} values')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# ---------- Seaborn (ggplot2-style) Version ----------
sns.set_theme(style='whitegrid')
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x=x_col, y=y_col, color='blue', s=70)
plt.title(f'Scatter Plot of {x_col} vs {y_col}')
plt.xticks(rotation=90)
# plt.locator_params(axis='x', nbins=15)
plt.xlabel(f'{x_col} values')
plt.ylabel(f'{y_col} values')
plt.show()

# --- Define dependent and independent variables ---
# dependent = 'IMDB_Rating'
# independents = ['Released_Year', 'Runtime', 'Meta_score', 'No_of_Votes', 'Gross']

# # --- Create subplots ---
# sns.set_theme(style='whitegrid')
# fig, axes = plt.subplots(2, 3, figsize=(15, 8))  # 2 rows × 3 columns
# axes = axes.flatten()

# # --- Plot each independent vs dependent ---
# for i, col in enumerate(independents):
#     sns.scatterplot(data=df, x=col, y=dependent, ax=axes[i], color='blue', s=40, alpha=0.7)
#     axes[i].set_title(f'{col} vs {dependent}')
#     axes[i].set_xlabel(col)
#     axes[i].set_ylabel(dependent)

# # --- Hide any unused subplot (if number of plots < grid cells) ---
# for j in range(len(independents), len(axes)):
#     fig.delaxes(axes[j])

# plt.suptitle("Scatter Plots of Independent Variables vs IMDB_Rating", fontsize=14, y=1.02)
# plt.tight_layout()
# plt.show()


# --- Handle missing or non-numeric data ---
# Drop rows with missing dependent variable or predictors
df = df.dropna(subset=['IMDB_Rating', 'Released_Year', 'Runtime', 'Genre', 
                       'Meta_score', 'No_of_Votes', 'Gross'])

# --- Encode Genre if categorical ---
if df['Genre'].dtype == 'object':
    le = LabelEncoder()
    df['Genre'] = le.fit_transform(df['Genre'].astype(str))

# --- Convert all numeric columns explicitly ---
numeric_cols = ['Released_Year', 'Runtime', 'Genre', 'Meta_score', 'No_of_Votes', 'Gross', 'IMDB_Rating']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Drop rows that still have NaN after conversion
df = df.dropna(subset=numeric_cols)

# --- Define predictors and response ---
X = df[['Released_Year', 'Runtime', 'Genre', 'Meta_score', 'No_of_Votes', 'Gross']]
y = df['IMDB_Rating']

# --- Add constant ---
X = sm.add_constant(X)

# --- Fit GLM models ---
glm_identity = sm.GLM(y, X, family=Gaussian(identity())).fit()
glm_log = sm.GLM(y, X, family=Gaussian(log())).fit()
glm_gamma = sm.GLM(y, X, family=Gamma(log())).fit()

# --- Summaries ---
print("=== GLM: Gaussian (Identity Link) ===")
print(glm_identity.summary())
print("\n=== GLM: Gaussian (Log Link) ===")
print(glm_log.summary())
print("\n=== GLM: Gamma (Log Link) ===")
print(glm_gamma.summary())

# --- Compute comparison metrics ---
def pseudo_r2(model):
    return 1 - model.deviance / model.null_deviance

def compute_metrics(model, name):
    preds = model.fittedvalues
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    return {
        "Model": name,
        "AIC": model.aic,
        "BIC": model.bic,
        "Pseudo_R2": pseudo_r2(model),
        "RMSE": rmse,
        "MAE": mae
    }

results = [
    compute_metrics(glm_identity, "Gaussian (identity)"),
    compute_metrics(glm_log, "Gaussian (log)"),
    compute_metrics(glm_gamma, "Gamma (log)")
]

summary_df = pd.DataFrame(results)
print("\n=== Model Comparison Summary ===")
print(summary_df)

# --- Plot Observed vs Predicted ---
models = {
    "Gaussian (identity)": glm_identity,
    "Gaussian (log)": glm_log,
    "Gamma (log)": glm_gamma
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (name, model) in zip(axes, models.items()):
    preds = model.fittedvalues
    ax.scatter(preds, y, alpha=0.6)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")

plt.tight_layout()
plt.show()
