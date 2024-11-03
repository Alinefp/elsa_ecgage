#%%
!pip install lifelines

#%%
from lifelines import CoxPHFitter, KaplanMeierFitter
import pandas as pd

#%%
# Load your data (replace 'your_data.csv' with your actual data file)
data = pd.read_stata('/Users/af955/Library/CloudStorage/OneDrive-YaleUniversity/ELSA_Databases/ecg_age.dta')

# Define models with their respective variables
models = {
    "Model 1": ["idadea", "sex"],
    "Model 2": ["delta_ecg", "idadea", "sex"],
    "Model 3": ["who_cvdx_br"],
    "Model 4": ["delta_ecg", "who_cvdx_br"]
}

#%%
for model_name, predictors in models.items():
    cph = CoxPHFitter()
    cph.fit(data[['s_deratero_mes', 's_deratero'] + predictors], duration_col='s_deratero_mes', event_col='s_deratero')
    
    # Print the summary for each model
    print(f"{model_name} Summary:")
    print(cph.summary)
    
    # Plot the survival function for each predictor set
    cph.plot()
    plt.title(f'Survival Function for {model_name}')
    plt.show()
    
# %%
# Define predictors for a Cox model (example here for 'Model 2')
predictors = ['delta_ecg', 'idadea', 'sex']

# Step 1: Fit Cox Proportional Hazards model
cph = CoxPHFitter()
cph.fit(data[['s_deratero_mes', 's_deratero'] + predictors], duration_col='s_deratero_mes', event_col='s_deratero')

# Step 2: Predict survival probabilities at specific time points
time_points = [12, 24, 36]  # Define specific time points (in months, for example) for comparison
predicted_survival = pd.DataFrame({
    t: cph.predict_survival_function(data[predictors], times=[t]).values.flatten()
    for t in time_points
})

# Step 3: Calculate observed survival probabilities using Kaplan-Meier estimator
kmf = KaplanMeierFitter()
kmf.fit(data['s_deratero_mes'], event_observed=data['s_deratero'])

observed_survival = {t: kmf.survival_function_at_times(t).values[0] for t in time_points}

# Step 4: Compute Expected/Observed Ratios
eo_ratios = {}
for t in time_points:
    expected_survival_mean = predicted_survival[t].mean()  # Mean predicted survival probability at time t
    observed_survival_prob = observed_survival[t]          # Observed survival probability at time t
    eo_ratios[t] = expected_survival_mean / observed_survival_prob if observed_survival_prob != 0 else np.nan

# Display E/O ratios for each time point
print("Expected/Observed Ratios at Specified Time Points:")
for t, eo_ratio in eo_ratios.items():
    print(f"Time {t} months: E/O ratio = {eo_ratio:.3f}")
    
# %%
# Step 2: Calculate the predicted number of events at each time point
predicted_events = {}
for t in time_points:
    # Calculate predicted survival probabilities for each individual at time t
    survival_probs = cph.predict_survival_function(data[predictors], times=[t])
    # Expected number of events is (1 - survival probability) for each individual
    expected_events = 1 - survival_probs.values.flatten()
    # Sum the expected events across individuals for each time point
    predicted_events[t] = expected_events.sum()

# Step 3: Calculate observed number of events using Kaplan-Meier estimator
kmf = KaplanMeierFitter()
kmf.fit(data['s_deratero_mes'], event_observed=data['s_deratero'])
observed_events = {t: data[(data['s_deratero_mes'] <= t) & (data['s_deratero'] == 1)].shape[0] for t in time_points}

# Step 4: Compute Expected/Observed Ratios
eo_ratios = {}
for t in time_points:
    eo_ratios[t] = predicted_events[t] / observed_events[t] if observed_events[t] != 0 else np.nan

# Display E/O ratios for each time point
print("Expected/Observed Ratios at Specified Time Points:")
for t, eo_ratio in eo_ratios.items():
    print(f"Time {t} months: E/O ratio = {eo_ratio:.3f}")