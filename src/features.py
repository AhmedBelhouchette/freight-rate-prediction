import pandas as pd

DATA = "data"
FEATURES = ["distance", "equip", "weight", "pickup_lat", "pickup_lon",
            "delivery_lat", "delivery_lon", "quote_signal", "dow"]
EQUIPMENT = ["Dry Van", "Flatbed", "Reefer"]


def load(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()
    return df


def clean(df):
    df = df.copy()
    df["weight"] = df["weight"].abs()
    df["weight"] = df["weight"].fillna(df.groupby("equipment")["weight"].transform("median"))
    df["market_index"] = df["market_index"].fillna(df.groupby("date")["market_index"].transform("mean"))
    return df


def add_features(df):
    df = df.copy()
    df["dow"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["equip"] = pd.Categorical(df["equipment"], categories=EQUIPMENT)
    return df


def remove_bad_prices(df):
    df = df.copy()
    df["rpm"] = df["posted_rate"] / df["distance"]
    lane_median = df.groupby(["pickup", "delivery", "equipment"])["rpm"].transform("median")
    ratio = df["rpm"] / lane_median
    return df[(ratio >= 0.5) & (ratio <= 2)]
