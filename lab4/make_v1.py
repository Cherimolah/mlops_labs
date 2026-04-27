import os
import pandas as pd
from catboost.datasets import titanic

os.makedirs("data", exist_ok=True)

train, _ = titanic()
df = train[["Pclass", "Sex", "Age"]].copy()
df.to_csv("data/titanic.csv", index=False)
