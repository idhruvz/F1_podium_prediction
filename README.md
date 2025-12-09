# Formula 1 Podium Prediction & Weather Impact Analysis

This project is an end-to-end **machine learning system** that predicts the probability that a Formula 1 driver will finish on the **podium (Top 3)** using historical race data, driver form, and constructor performance.

It also includes an **exploratory weather impact analysis** and a **Streamlit dashboard** for interactive podium prediction.

## 1. Project Goals

- Predict whether a driver will finish on the podium before a race
- Engineer realistic driver and constructor performance features
- Compare baseline (Logistic Regression) and advanced (Random Forest) models
- Explore the impact of wet vs dry race conditions
- Deploy the final model as a simple web app using Streamlit

---

## 2. Data & Preprocessing

The project uses historical Formula 1 race data with:

- Race information (year, round, circuit, raceId)
- Driver and constructor IDs
- Grid position, finishing position, points
- DNFs and race outcomes

Key preprocessing steps:

- Converted `position` to numeric (DNFs handled via `NaN` → large value)
- Filtered invalid grid positions
- Created binary target:

  ```text
  podium = 1 if finishing position ≤ 3 else 0
  
## 3. Exploratory Data Analysis (EDA)

The EDA explores:

- Distributions of:
  - Grid position  
  - Final race position  
  - Driver points  
- Podium class imbalance  
- Relationship between grid and final result  
- Podium probability as a function of starting grid position  
- Constructor dominance across seasons  

### Main Insights

- Starting closer to the front strongly increases podium probability.
- Podium finishes are highly imbalanced compared to non-podium finishes.
- Team strength varies by era due to constructor dominance cycles.

---

## 4. Feature Engineering

Time-aware rolling features were created to approximate driver and team **form** and **momentum**.

### Driver Features

- **driver_avg_finish_last3**  
  Average finishing position over the last 3 races  

- **driver_points_last5**  
  Total points scored over the last 5 races  

- **driver_dnfs_last5**  
  Number of DNFs in the last 5 races  

- **position_delta**  
  Grid position minus finishing position  

### Constructor Features

- **constructor_points_last5**  
  Team points scored in the last 5 races  

- **constructor_wins_last10**  
  Team wins in the last 10 races  

- **constructor_dnfs_last5**  
  Team DNFs in the last 5 races  

All rolling features are computed in **sorted time order** to strictly avoid **data leakage**.

