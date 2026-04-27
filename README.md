🌫️ Detailed Workflow & Methodology — AQI Prediction System
🧭 Problem Statement

The objective of this project is to predict the Air Quality Index (AQI) for the next hour using historical environmental and pollution data. AQI is influenced by multiple interacting factors such as particulate matter, gases, weather conditions, and temporal patterns. Therefore, the problem is formulated as a time-series regression task.

📥 1. Data Collection & Understanding

The dataset is obtained from Kaggle and contains hourly observations (2010–2026) for a city.

🔑 Core Features
🌍 Pollution Indicators
PM2.5 (fine particles)
PM10 (coarse particles)
NO₂, SO₂, CO, O₃ (gaseous pollutants)
🌦️ Meteorological Factors
Temperature
Humidity
Wind speed
🌤️ Environmental Factors
Solar radiation
Rainfall
Cloud coverage
🎯 Target Variable
AQI (Air Quality Index)
🧠 Model Flow Description

This project follows a structured and systematic approach to predict the Air Quality Index (AQI) using historical environmental data. The model flow begins with raw data ingestion and progresses through multiple stages including preprocessing, feature engineering, model training, and final prediction.

📥 Data Input

The process starts with loading a time-series dataset containing hourly observations of air pollutants and meteorological parameters. Each record represents environmental conditions at a specific timestamp. The dataset includes pollutant concentrations such as PM2.5, PM10, NO₂, SO₂, CO, and O₃, along with weather-related features like temperature, humidity, wind speed, rainfall, solar radiation, and cloud coverage.

🧹 Data Preparation

Before feeding the data into the models, several preprocessing steps are performed to ensure data quality and consistency. Missing values are handled using forward-fill and backward-fill methods to maintain continuity in the time series. The datetime column is converted into a proper timestamp format and the dataset is sorted chronologically. This step is crucial because time-series models rely heavily on the correct temporal order of data.

⏱️ Temporal Feature Extraction

To capture time-based patterns, additional features are extracted from the datetime column. These include hour, day, month, year, day of the week, and day of the year. Since time is cyclical in nature (for example, hour 23 is close to hour 0), sine and cosine transformations are applied to encode these cyclical patterns. This helps the model understand periodic trends such as daily and seasonal variations in air quality.

🧠 Feature Engineering

A significant part of the model flow involves creating meaningful features that capture the underlying environmental dynamics. Several domain-specific indices are constructed to enhance predictive power. For instance, total pollution is calculated by summing all pollutant concentrations, while the fine-to-coarse ratio (PM2.5/PM10) provides insight into particle composition.

Additional features such as dispersion index (influenced by wind and cloud coverage), washout effect (impact of rainfall and wind), humidity-temperature index, ozone formation index, and inversion risk are engineered to reflect real-world atmospheric behavior. These features allow the model to better understand how different environmental factors interact and influence AQI.



🔁 Time-Series Feature Construction

To incorporate historical dependency, time-series features are generated. Lag features are created by including past values of AQI and other variables at different time intervals (e.g., 1 hour, 3 hours, 24 hours). Rolling statistics such as moving averages and standard deviations are also computed to capture trends and variability over time.

Additionally, differencing and exponential moving averages are applied to highlight short-term changes and smooth temporal patterns. These features enable the model to learn how past conditions influence future AQI levels.

🎯 Target Definition

The prediction goal is defined by shifting the AQI column forward by one time step. This means the model learns to predict the AQI for the next hour based on current and past observations. This transformation converts the problem into a supervised learning task with temporal dependencies.

🤖 Machine Learning Model Flow

Two traditional machine learning models are used: XGBoost and Random Forest.

The data is split into training and testing sets while preserving chronological order. The models are trained on the training data, where they learn relationships between input features and AQI values.

XGBoost works by building multiple decision trees sequentially, where each new tree focuses on correcting errors made by previous ones. This allows it to capture complex nonlinear relationships in the data.

Random Forest, on the other hand, builds multiple independent decision trees using random subsets of data and features. The final prediction is obtained by averaging the outputs of all trees, making it robust and less prone to overfitting.

🔁 Deep Learning Model Flow (LSTM)

In addition to machine learning models, a Long Short-Term Memory (LSTM) network is used to capture temporal dependencies more effectively.

The data is first normalized using MinMax scaling to ensure stable training. Then, sequences of 24 consecutive time steps are created, where each sequence represents the past 24 hours of data. The model uses this sequence to predict the AQI for the next hour.

The LSTM network processes input sequentially, maintaining an internal memory that helps it retain important past information while discarding irrelevant data. The architecture consists of an LSTM layer followed by dense layers that produce the final AQI prediction.

📊 Model Evaluation

The performance of the models is evaluated using standard regression metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score. These metrics help assess how accurately the models predict AQI and how well they generalize to unseen data.

🔮 Prediction System

A prediction function is implemented to forecast AQI for any given datetime. The system retrieves the most recent 24 hours of data prior to the input time, applies the same preprocessing and scaling steps, and feeds the sequence into the trained LSTM model.

The predicted value is then transformed back to its original scale to provide the final AQI prediction. This approach closely mimics real-world forecasting scenarios, where recent observations are used to estimate near-future conditions.
🔄 Cyclical Feature Encoding
📌 Why Cyclical Encoding is Needed

Many time-based features such as hour, day, and month are cyclical in nature, meaning they repeat after a fixed interval:

Hour: 0 → 23 → 0
Month: Jan → Dec → Jan
Day of week: Mon → Sun → Mon

However, if these features are used directly as numerical values, the model may interpret them incorrectly.

❗ Problem with Raw Values

For example:

Hour 23 and Hour 0 are actually very close in time
But numerically: 23 and 0 appear far apart

👉 This creates a false discontinuity, which negatively affects model performance.

🔁 Solution: Cyclical Transformation

To preserve the natural circular behavior of time, we transform these features using sine and cosine functions.

📐 Mathematical Representation

For any cyclical feature:

sin_value=sin(2π⋅x/T)
cos_value=cos(2π⋅x/T)
Where:
x = current value (e.g., hour)
T = total cycle length (e.g., 24 hours)
⚙️ Implementation in the Project
🕒 Hour Encoding
hour_sin = np.sin(2 * np.pi * df['hour'] / 24)
hour_cos = np.cos(2 * np.pi * df['hour'] / 24)
📅 Month Encoding
month_sin = np.sin(2 * np.pi * df['month'] / 12)
month_cos = np.cos(2 * np.pi * df['month'] / 12)
📆 Year (Seasonal Cycle)
year_sin = np.sin(2 * np.pi * df['dayofyear'] / 365)
year_cos = np.cos(2 * np.pi * df['dayofyear'] / 365)
🧠 Intuition Behind the Transformation

Instead of representing time as a straight line, cyclical encoding maps it onto a circle.

🎯 Key Benefits:
Preserves periodic relationships
Eliminates artificial gaps (e.g., 23 → 0)
Helps models learn seasonal and daily patterns
Improves prediction accuracy
🔍 Example Insight

After transformation:

Hour 23 and Hour 0 will have very similar sine and cosine values
The model correctly understands that they are close in time
📊 Visual Interpretation

If plotted:

Sine and cosine form a circular pattern
Each time value corresponds to a point on the unit circle

👉 This allows the model to capture:

Daily pollution cycles
Seasonal variations
Repeating environmental patterns

📌 Summary

The overall model flow transforms raw environmental data into structured and meaningful features, incorporates temporal dependencies through time-series techniques, and applies both machine learning and deep learning models to generate accurate AQI predictions. By combining domain knowledge with advanced modeling techniques, the system effectively captures the complex dynamics of air pollution and provides reliable forecasts.
