import pandas as pd
import numpy as np
from pygam import LinearGAM, GammaGAM, GAM, s
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from pygam.links import LogLink

# --- Load dataset ---
df = pd.read_csv("imdb_dataset_cleaned.csv")

# --- Clean and prepare data ---
df = df.dropna(subset=['IMDB_Rating', 'Released_Year', 'Runtime', 'Genre',
                       'Meta_score', 'No_of_Votes', 'Gross'])

# Encode Genre if categorical
if df['Genre'].dtype == 'object':
    le = LabelEncoder()
    df['Genre'] = le.fit_transform(df['Genre'].astype(str))

# Convert to numeric (safe cast)
numeric_cols = ['Released_Year', 'Runtime', 'Genre', 'Meta_score', 'No_of_Votes', 'Gross', 'IMDB_Rating']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=numeric_cols)

# --- Define predictors and response ---
X = df[['Released_Year', 'Runtime', 'Genre', 'Meta_score', 'No_of_Votes', 'Gross']].values
y = df['IMDB_Rating'].values

# --- 1. GAM (Gaussian with identity link) ---
gam_identity = LinearGAM(
    s(0) + s(1) + s(2) + s(3) + s(4) + s(5)
).fit(X, y)

# --- 2. GAM (Gaussian with log link) ---
gam_log = GAM(
    s(0) + s(1) + s(2) + s(3) + s(4) + s(5),
).fit(X, y)

# --- 3. GAM (Gamma with log link) ---
gam_gamma = GammaGAM(
    s(0) + s(1) + s(2) + s(3) + s(4) + s(5),
).fit(X, y)

# --- Evaluate models ---
def compute_metrics(model, X, y, name):
    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    aic = model.statistics_['AIC']
    pseudo_r2 = model.statistics_['pseudo_r2']
    return {
        "Model": name,
        "AIC": aic,
        "Pseudo_R2": pseudo_r2,
        "RMSE": rmse,
        "MAE": mae
    }

results = [
    compute_metrics(gam_identity, X, y, "GAM (Gaussian, identity)"),
    compute_metrics(gam_log, X, y, "GAM (Gaussian, log)"),
    compute_metrics(gam_gamma, X, y, "GAM (Gamma, log)")
]

summary_df = pd.DataFrame(results)
print("\n=== Model Comparison Summary ===")
print(summary_df)

# --- Plot observed vs predicted for each model ---
models = {
    "GAM (Gaussian, identity)": gam_identity,
    "GAM (Gaussian, log)": gam_log,
    "GAM (Gamma, log)": gam_gamma
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (name, model) in zip(axes, models.items()):
    preds = model.predict(X)
    ax.scatter(preds, y, alpha=0.6)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")

plt.tight_layout()
plt.show()

# --- Plot smooth functions for the best model (example: Gamma) ---
best_model = gam_gamma  # Choose based on summary_df
fig, axs = plt.subplots(3, 2, figsize=(10, 10))
axs = axs.ravel()

for i, ax in enumerate(axs):
    XX = best_model.generate_X_grid(term=i)
    ax.plot(XX[:, i], best_model.partial_dependence(term=i, X=XX))
    ax.plot(XX[:, i], best_model.partial_dependence(term=i, X=XX, width=0.95)[1], c='r', ls='--')
    ax.set_title(f"Smoothing: {df.columns[1:][i]}")

plt.tight_layout()
plt.show()
