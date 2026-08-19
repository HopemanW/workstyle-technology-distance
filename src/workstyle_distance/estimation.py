from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def age_distance_benchmark(df: pd.DataFrame) -> pd.Series:
    y=df["future_growth"].to_numpy(float)
    X=np.column_stack([np.ones(len(df)), df["technology_exposure"], df["log_age"],
                       df["workstyle_distance"], df["log_age"]*df["workstyle_distance"],
                       np.log1p(df["size"]), df["productivity"]])
    b=np.linalg.lstsq(X,y,rcond=None)[0]
    return pd.Series(b,index=["intercept","technology_exposure","log_age","workstyle_distance",
                              "age_x_distance","log_size","productivity"])

def fit_growth_forest(df: pd.DataFrame):
    X=df[["log_age","workstyle_distance","technology_exposure","size","productivity"]]
    y=df["future_growth"]
    model=RandomForestRegressor(n_estimators=400,min_samples_leaf=15,random_state=1,n_jobs=-1)
    model.fit(X,y)
    return model
