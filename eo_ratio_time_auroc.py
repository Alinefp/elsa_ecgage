#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error, r2_score, mean_absolute_error
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#%%
# Load your data
data = pd.read_stata('/Users/af955/Library/CloudStorage/OneDrive-YaleUniversity/ELSA_Databases/ecg_age.dta')

# Define models with their respective variables
models = {
    "Model 1": ["idadea", "sex"],
    "Model 2": ["delta_ecg", "idadea", "sex"],
    "Model 3": ["who_cvdx_br"],
    "Model 4": ["delta_ecg", "who_cvdx_br"]
}

#%%
# Define time-to-event variables
time_to_event = 's_deratero_mes'  
event_occurred = 's_deratero'  

# Filter to include only data within the 5-year window and drop NaNs
data = data[(data[time_to_event] <= 5) & (data[event_occurred].notna())].dropna()

# Standardize predictor variables
scaler = StandardScaler()
for model_name, predictors in models.items():
    data[predictors] = scaler.fit_transform(data[predictors])

#%%
# Initialize results dictionary
results = {}
cindex_scores = {}
auroc_scores = {}

# For each model, fit a Cox Proportional Hazards model
for model_name, predictors in models.items():
    # Drop rows with NaNs in predictor variables
    model_data = data[predictors + [time_to_event, event_occurred]].dropna()

    # Check for NaNs again
    if model_data.isnull().values.any():
        raise ValueError(f"{model_name} contains NaN values after filtering.")

    # Split data into train and test sets
    train_data, test_data = train_test_split(model_data, test_size=0.2, random_state=42)
    
    # Fit the Cox model on training data
    cph = CoxPHFitter()
    try:
        cph.fit(train_data, duration_col=time_to_event, event_col=event_occurred)
    except ConvergenceError as e:
        print(f"{model_name} failed to converge: {e}")
        continue

    # Predict survival probabilities for test data at the 5-year time point
    survival_probs = cph.predict_survival_function(test_data, times=[5]).T
    predicted_probs = 1 - survival_probs[5]  # Probability of event by 5 years
    
    # Calculate observed outcomes in test data
    observed_events = test_data[event_occurred]

    # Calculate concordance index (c-index) as a performance metric
    cindex = concordance_index(test_data[time_to_event], -predicted_probs, observed_events)
    cindex_scores[model_name] = cindex

    # Calculate AUROC for discrimination at 5 years
    auroc = roc_auc_score(observed_events, predicted_probs)
    auroc_scores[model_name] = auroc

    # Store predicted vs observed for calibration plot
    results[model_name] = {
        "Predicted Probability": predicted_probs,
        "Observed": observed_events.values
    }

    print(f"{model_name} trained successfully. C-index: {cindex:.2f}, AUROC: {auroc:.2f}")

#%%
# Calibration Plot: Predicted vs Observed Ratios
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Predicted vs Observed Ratios for Each Model (5 Years)')

for i, (model_name, result) in enumerate(results.items()):
    ax = axes[i // 2, i % 2]
    # Binning predicted probabilities
    bins = pd.qcut(result["Predicted Probability"], q=10)
    obs_rate = result["Observed"].groupby(bins).mean()
    pred_rate = result["Predicted Probability"].groupby(bins).mean()

    # Plot calibration curve
    ax.plot(pred_rate, obs_rate, 'o-')
    ax.plot([0, 1], [0, 1], 'r--')  # Reference line for perfect calibration
    ax.set_title(f"{model_name}")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Event Rate")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#%%
# AUROC Curves for each model
plt.figure(figsize=(10, 8))
for model_name, result in results.items():
    fpr, tpr, _ = roc_curve(result["Observed"], result["Predicted Probability"])
    plt.plot(fpr, tpr, label=f'{model_name} (AUROC = {auroc_scores[model_name]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # Reference line for no discrimination
plt.title('AUROC Curve for Each Model')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

#%%
# Display results summary
summary_df = pd.DataFrame({
    "C-Index": cindex_scores,
    "AUROC": auroc_scores
}).T
print("Model Performance Summary:")
print(summary_df)

#%%
# Check event distribution in the test set
print(f"Event occurrences in test set for {model_name}:")
print(test_data[event_occurred].value_counts())
# %%
