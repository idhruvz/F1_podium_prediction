# Formula 1 Podium Prediction & Weather Impact Analysis
🔗 **Live Demo:** https://f1podiumprediction-tg4nt9oj6rwx2gaiza3rgp.streamlit.app/


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

## 5. Machine Learning

### 5.1 Train/Test Split

- A **time-based train/test split** was used.
- Older seasons were used for training.
- Newer seasons were used for testing.
- This setup simulates **real-world future race prediction** and prevents data leakage.

### 5.2 Models Used

Two models were trained and compared:

- **Logistic Regression**
  - Used as a baseline linear classifier  
  - Helped verify basic feature usefulness

- **Random Forest (Final Model)**
  - Captures non-linear relationships  
  - Handles interactions between driver form, grid and team performance  
  - Delivered significantly better performance than Logistic Regression  

 **Final Random Forest Performance:**

- ROC–AUC ≈ **0.95**
- Strong recall for the **podium class**
- Stable performance on unseen test seasons

---

## 6. Weather Impact Analysis (Exploratory)

Since the dataset did not include official weather labels:

- A **proxy-based list of historically wet races** was manually created.
- Races were labelled as:
  - `Dry = 0`
  - `Wet = 1`

The following comparisons were performed:

- **DNF rate: Wet vs Dry**
- **Average position change (Grid − Finish): Wet vs Dry**

 **Important Limitation:**

- Wet races were limited in number
- Weather labels were approximate
- Results were treated as **exploratory only**
- Weather was **not used as a predictive feature** in the model


## 7. Streamlit Dashboard

A full interactive **Streamlit dashboard** was built to showcase the model and analysis.

### 7.1 Podium Prediction Tab

- User inputs:
  - Grid position
  - Driver recent performance
  - Constructor recent performance
- Outputs:
  - Podium probability (%)
  - Text-based interpretation:
    - High likelihood
    - Moderate likelihood
    - Low likelihood

### 7.2 Model & Data Insights Tab

- Podium probability vs grid position plot
- Random Forest feature importance
- Model interpretation explaining:
  - Role of grid
  - Driver form impact
  - Constructor performance influence
  - Reliability penalties

### 7.3 Weather Impact Analysis Tab

- Dry vs Wet race summary table
- DNF rate comparison plot
- Position gain comparison plot
- Explanation of sample-size limitations


## 8. Technology Stack

- **Programming Language:** Python  
- **Data Handling:** Pandas, NumPy  
- **Machine Learning:** Scikit-learn  
- **Visualization:** Matplotlib  
- **Dashboard:** Streamlit  
- **Model Saving:** Joblib  


## 9. How to Run the Project Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/idhruvz/F1_podium_prediction.git
   cd F1_podium_prediction


