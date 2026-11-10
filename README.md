Predicting the price of a used car from brand, year, mileage, engine, transmission, etc.

The final model Random Forest on a log-transformed target has a typical mean absolute error (MAE) of about $10–15k, roughly half of what a naive "always predict the median" baseline gives.

Data

    1. Source: Used Car Price Prediction Dataset (Kaggle), license CC BY 4.0, 4,009 listings, 12 columns
    
    2. Issues found in the raw data:
        price and milage stored as text ($21,000, 51,000 mi.)

        missing values in fuel_type, accident, clean_title; disguised missing values (–, not supported)

        engine and transmission are free text with several characteristics packed into one string

        model has ~1,650 unique values (not used), colors have ~270 / ~130 variants

        price are skewed to the right (from ~$2k to ~$2M)


Approach

    1. Split 80 for train set and 20 for test set, stratified by price quintiles (random_state=35)

    2. Cleaning (custom functions in src/preprocessing.py):
        milage, price converted to numbers
        fuel_type: electric and hydrogen cars recovered from the engine text, – marked as Unknown
        accident: missing to Unknown; clean_title: binary flag
        engine horsepower, displacement and cylinders extracted from engine text
        transmission type and gear count extracted from transmission

    3. Preprocessing pipeline: median imputation, scaling for numeric features, one-hot encoding for categorical (rare categories merged)


Model's Results

    On train set MAE 9910.57 RMSE 28622.31

    On test set MAE 15002.79 RMSE 120106.62


Limitations

    1. Very expensive cars are extremely rare, so the model predicts them poorly

    2. The test set is small (~800 rows), so it's noisy

    3. RMSE on the test set is therefore not representative of typical performance