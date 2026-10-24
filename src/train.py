import os
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

from features import DATA, FEATURES, load, clean, add_features, remove_bad_prices

PARAMS = dict(n_estimators=800, learning_rate=0.05, num_leaves=63, verbose=-1)


def baseline_mape(train, test):
    lane_rpm = train.groupby(["pickup", "delivery", "equipment"])["rpm"].median()
    equip_rpm = train.groupby("equipment")["rpm"].median()
    keys = list(zip(test["pickup"], test["delivery"], test["equipment"]))
    rpm = pd.Series(keys, index=test.index).map(lane_rpm)
    rpm = rpm.fillna(test["equipment"].map(equip_rpm))
    return mean_absolute_percentage_error(test["posted_rate"], rpm * test["distance"])


def main():
    data = add_features(remove_bad_prices(clean(load(f"{DATA}/train-test.csv"))))

    # time split: train jan-aug, test sep-oct
    train = data[data["date"] < "2025-09-01"]
    test = data[data["date"] >= "2025-09-01"]

    model = lgb.LGBMRegressor(**PARAMS).fit(train[FEATURES], train["rpm"])
    pred = model.predict(test[FEATURES]) * test["distance"]
    print(f"baseline MAPE: {baseline_mape(train, test) * 100:.2f}%")
    print(f"model    MAPE: {mean_absolute_percentage_error(test['posted_rate'], pred) * 100:.2f}%")

    # final model on all jan-oct
    final = lgb.LGBMRegressor(**PARAMS).fit(data[FEATURES], data["rpm"])
    os.makedirs("models", exist_ok=True)
    final.booster_.save_model("models/model.txt")
    print("saved models/model.txt")


if __name__ == "__main__":
    main()
