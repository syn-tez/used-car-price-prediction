import re
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def milage_clean(df):
    df = df.copy()

    df["milage"] = df["milage"].str.replace(r"[,\smi.]", "", regex=True).astype(int)  # removing non-numeric characters and converting to int

    return df


def fuel_type_imputation(df):
    df = df.copy()

    is_hydrogen = df["engine"].str.contains("Hydrogen", case=False, na=False)
    df.loc[is_hydrogen, "fuel_type"] = "Hydrogen"

    df.loc[df["fuel_type"].isna(), "fuel_type"] = "Electric"

    df["fuel_type"] = df["fuel_type"].replace("–", "Unknown")

    return df


def accident_imputation(df):
    df = df.copy()

    df["accident"] = (df["accident"].replace({"None reported": "None",
                                            "At least 1 accident or damage reported": "Reported"})
                                    .fillna("Unknown"))

    return df


def clean_title_encode(df):
    df = df.copy()

    df["clean_title"] = (df["clean_title"] == "Yes").astype(int)

    return df


def engine_features_extract(df):
    df = df.copy()

    engine = df["engine"]

    df["horsepower"] = engine.str.extract(r"(\d+(?:\.\d+)?)\s*HP", expand=False).astype(float)
    df["displacement_l"] = engine.str.extract(r"(\d+(?:\.\d+)?)\s*(?:Liter|L)\b", expand=False).astype(float)

    cyl = engine.str.extract(r"(\d+)\s*Cylinder", expand=False)
    cyl = cyl.fillna(engine.str.extract(r"\bV(\d+)\b", expand=False))
    df["cylinders"] = cyl.astype(float)

    is_electric = engine.str.contains("Electric", case=False, na=False)
    df.loc[is_electric, ["displacement_l", "cylinders"]] = 0

    return df


def transmission_features_extract(df):
    df = df.copy()

    t = df["transmission"]

    df["gear_count"] = (t.str.extract(r"(\d+)[\s-]*(?:speed|spd)", flags=re.IGNORECASE, expand=False).astype(float))  # number of gears

    df.loc[t.str.contains("Single-Speed", case=False, na=False), "gear_count"] = 1

    conditions = [
        t.str.contains(r"CVT|Variable", case=False, na=False),
        t.str.contains("Single-Speed", case=False, na=False),
        t.str.contains(r"A/?T\s*/\s*M/?T", case=False, na=False),
        t.str.contains(r"M/T|\bMT\b|Manual", case=False, na=False),
        t.str.contains(r"A/T|\bAT\b|Auto|Dual Shift", case=False, na=False),
    ]

    choices = ["CVT", "Single-Speed", "Automatic", "Manual", "Automatic"]
    df["transmission_type"] = np.select(conditions, choices, default="Unknown")

    return df


COLOR_KEYWORDS = {
    "black": "Black", "ebony": "Black", "nero": "Black",
    "white": "White", "silver": "Silver", "gray": "Gray", "grey": "Gray",
    "blue": "Blue", "red": "Red", "green": "Green", "gold": "Gold",
    "brown": "Brown", "beige": "Beige", "orange": "Orange", "yellow": "Yellow",
}
COLOR_PATTERN = r"\b(" + "|".join(COLOR_KEYWORDS) + r")\b"


def simplify_color(series):
    found = series.str.extract(COLOR_PATTERN, flags=re.IGNORECASE, expand=False).str.lower()
    result = found.map(COLOR_KEYWORDS).fillna("Other")
    is_unknown = series.isna() | series.str.strip().isin(["–", "-"])
    
    return result.mask(is_unknown, "Unknown")


def group_colors(df):
    df = df.copy()
    
    df["ext_col_group"] = simplify_color(df["ext_col"])
    df["int_col_group"] = simplify_color(df["int_col"])
    
    return df


cleaning = Pipeline([
    ("milage",      FunctionTransformer(milage_clean)),
    ("fuel_type",   FunctionTransformer(fuel_type_imputation)),
    ("accident",    FunctionTransformer(accident_imputation)),
    ("clean_title", FunctionTransformer(clean_title_encode)),
    ("engine",      FunctionTransformer(engine_features_extract)),
    ("transmission", FunctionTransformer(transmission_features_extract)),
    ("colors",      FunctionTransformer(group_colors)),
])


# numerical features
num_cols = ["model_year", "milage", "horsepower", "displacement_l", "cylinders"]

# categorical features
cat_cols = ["brand", "fuel_type", "accident", "transmission_type", "ext_col_group", "int_col_group"]

# binary feature
bin_cols = ["clean_title"]


num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])


cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False)),
])


column_transformer = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols),
        ("bin", "passthrough", bin_cols),
    ],
    remainder="drop",
)

column_transformer.set_output(transform="pandas")


final_pipeline = Pipeline([
    ("cleaning", cleaning),
    ("preprocessing", column_transformer),
])