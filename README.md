 Deep Learning-Based Modeling and Forecasting of Greenhouse Gas Emissions

 What This Project Is About

We built deep learning models to forecast industrial greenhouse gas emissions over time. The idea came from a real gap: Ethiopia is industrializing fast, but there are no reliable tools or facility-level datasets to predict how much GHG those industries will emit. We trained our models on U.S. EPA data (2011–2023) because it's detailed and well-structured, then designed the system so it can work with local Ethiopian data too.

Out of the three models we tried — LSTM, GRU, and a hybrid of both — the GRU came out on top with an R² of about 0.94. We also built a simple prototype where anyone can upload a CSV file with emission records and get a forecast for the next year.

 The Problem We Were Trying to Solve

Most countries, including Ethiopia, don't have the kind of facility-level emission data that makes accurate forecasting possible. The data that does exist is often country-wide, which is too broad to be useful for monitoring specific industries or factories. Meanwhile, Ethiopia is shifting from an agriculture-based economy to an industry-driven one, and without a way to predict emissions, it's hard for policymakers to plan ahead or meet the environmental commitments the country has signed onto internationally.

We used the U.S. GHGRP dataset as a stand-in — it has all the variables we needed, like unit type, heat capacity, and year-by-year emissions — and trained a model that can eventually be applied to Ethiopian facilities once local data becomes available.

 What We Were Trying to Do

The main goal was to build a working model that predicts GHG emissions from historical data. More specifically, we wanted to:

- Compare LSTM, GRU, and hybrid models on the same data and see which performs best
- Build something that handles the messiness of real industrial data (outliers, missing values, multiple gases with different warming effects)
- Create a prototype that a non-technical user could actually run with their own data

 The Data

We used the U.S. EPA Greenhouse Gas Reporting Program dataset covering 2011 to 2023.

The key variables fall into a few groups. Facility and location info includes things like the facility ID, name, city, and state. Industry classification comes from NAICS codes and sector/subpart labels. Each record also has a reporting year, unit-level details like unit type and heat input capacity, and the actual emission figures for CO2, biogenic CO2, methane, and nitrous oxide.

We also created a few extra variables ourselves. The most important one is Total CO2e, which converts all three gases into a single CO2-equivalent figure using standard global warming potential multipliers (25 for methane, 298 for nitrous oxide). We also created a Series ID to uniquely track each physical unit across facilities and years, and a flag to mark years where a unit reported zero emissions.

 How We Built It

 Getting the Data Ready

The raw data came in .xlsb format, so we used pyxlsb to load it. After that, we checked for duplicates (none found) and missing values (quite a few). For missing heat capacity values, we filled them using the median for that unit type, falling back to the global median when unit type was also missing. Missing categorical fields got labeled as "Unknown," and missing biogenic CO2 was set to zero since it's commonly unreported rather than truly absent.

 Handling Skewness and Scaling

The emission values and heat capacity were extremely right-skewed with a lot of outliers. We applied a Yeo-Johnson transformation to bring the distributions closer to normal — it works on both positive and negative values, unlike Box-Cox. After that, we scaled everything to a 0–1 range using MinMaxScaler. We specifically chose MinMaxScaler over StandardScaler because StandardScaler can produce negative values, which would deactivate ReLU neurons and cause information loss during training.

Categorical columns were encoded using embedding ID mapping, which doesn't inflate the number of dimensions the way one-hot encoding would.

 Building Sequences

Time series models need the data structured as sequences. We separated the features into two groups: dynamic features that change year to year (CO2e, heat capacity, zero-emission flag) and static features that stay the same (the categorical columns).

We then used a sliding window approach with a lookback of 6 years — meaning the model sees 6 years of history to predict the next year. We chose 6 based on autocorrelation analysis; it gave the best signal without generating too-short sequences. The final data split was 70% for training, 10% for validation, and 20% for testing, all split by time.

 The Models

All three models use the same basic structure: one branch processes the dynamic features using recurrent layers, another branch processes the static features using a dense layer, and the outputs of both branches are combined before the final prediction.

The LSTM model uses two stacked LSTM layers (64 then 32 units) for the dynamic branch and a 16-unit dense layer for the static branch. These are concatenated and passed through another dense layer before the output.

The GRU model is structured the same way but uses GRU layers instead. It also includes a small dropout layer (10%) before the output to reduce overfitting.

The hybrid model runs both an LSTM sub-branch and a GRU sub-branch in parallel on the dynamic features, then merges them together along with the static branch. It has more parameters overall but ended up performing worse than the simpler models.

All three were trained using the Adam optimizer with a learning rate of 0.0001 and Huber loss.

 Results

| Model  | MAE (Scaled) | RMSE (Scaled) | R2     | Inference Time (s) | Parameters |
|--------|-------------|---------------|--------|-------------------|------------|
| GRU    | 0.0265      | 0.0506        | 0.9369 | 10.70             | 23,537     |
| LSTM   | 0.0280      | 0.0524        | 0.9322 | 9.71              | 30,705     |
| Hybrid | 0.0525      | 0.0670        | 0.8892 | 10.48             | 55,697     |

The GRU performed best across all metrics and also had the fewest parameters, which makes it more efficient. The difference between GRU and LSTM looked small, so we ran a Diebold-Mariano test to check whether it was statistically meaningful — it was (p < 0.05). The hybrid model was a clear underperformer; it struggled especially with peak emission values, consistently predicting lower than the actual values.

Looking at the trends in the data, U.S. emissions have been declining over the years, and our 5-year forecast from 2023 onward projects a further reduction of roughly 1.76 billion metric tons CO2e.

 The Prototype

We wanted the project to be actually usable, not just a research exercise. The prototype lets someone upload a CSV file with facility-level emission data and get a next-year forecast back. Under the hood it loads the saved model, scaler, and transformation parameters, runs the same preprocessing steps on the new data, and then outputs real-world CO2e values in metric tons after reversing the transformations.

One practical detail: our model needs 6 years of history per unit to make a prediction. If a facility only has 3 years of data, the prototype fills in the missing earlier years by repeating the earliest available record (backfill padding).

 Tools Used

- Python for everything
- TensorFlow and Keras for building and training the models
- pyxlsb for reading the raw Excel data
- scikit-learn for scaling and transformation
- NumPy and Pandas for data handling

 What We'd Do Next

There are a few clear directions to take this further. The most important one is getting actual Ethiopian facility-level emission data — from national inventories or direct industry partnerships — so the model can be retrained and validated locally. Adding economic variables like GDP would likely improve accuracy since industrial output and emissions tend to move together. We'd also like to turn the prototype into a proper web app so it's accessible to facilities and environmental agencies without needing any coding knowledge.


## References

Full references are in the project report. The main sources are the U.S. EPA GHGRP dataset, published work on LSTM and GRU architectures for time series forecasting, and documentation on the Yeo-Johnson transformation and sliding window methods.
