import pandas as pd

df = pd.read_csv("data/titanic.csv")

mean_age = df["Age"].mean()
df["Age"] = df["Age"].fillna(mean_age)

df.to_csv("data/titanic.csv", index=False)
print(f"Средний возраст={mean_age}")
#df.to_csv("data/filled.csv", index=False)