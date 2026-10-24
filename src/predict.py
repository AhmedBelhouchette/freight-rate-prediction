import lightgbm as lgb
import pandas as pd

from features import DATA, FEATURES, load, clean, add_features


def predict(model, df):
    return model.predict(df[FEATURES]) * df["distance"]


def main():
    model = lgb.Booster(model_file="models/model.txt")
    train = load(f"{DATA}/train-test.csv")
    val = add_features(clean(load(f"{DATA}/validation.csv")))

    # 1. validation predictions
    template = load(f"{DATA}/validation-predictions-template.csv")
    preds = dict(zip(val["load_id"], predict(model, val)))
    template["predicted_rate"] = template["load_id"].map(preds)
    template.to_csv("validation_predictions.csv", index=False)
    print("saved validation_predictions.csv")

    # 2. december chart: coords from train, quote_signal = december median from validation
    #    (december daily averages only vary as much as sampling noise, so one value for the month)
    dec = load(f"{DATA}/december-chart-inputs.csv")
    cols = list(dec.columns)
    coords = pd.concat([
        train[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1),
        train[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1),
    ]).groupby("city").first()
    decf = dec.copy()
    for side in ["pickup", "delivery"]:
        decf[f"{side}_lat"] = decf[side].map(coords["lat"])
        decf[f"{side}_lon"] = decf[side].map(coords["lon"])
    decf["quote_signal"] = val.loc[val["date"] >= "2025-12-01", "quote_signal"].median()
    decf = add_features(decf)

    dec["predicted_rate"] = predict(model, decf)
    dec[cols].to_csv(f"{DATA}/december-chart-inputs-filled.csv", index=False)
    print("saved data/december-chart-inputs-filled.csv")


if __name__ == "__main__":
    main()
