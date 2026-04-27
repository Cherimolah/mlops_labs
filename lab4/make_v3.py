import pandas as pd

df = pd.read_csv("data/titanic.csv")
#df = pd.read_csv("data/filled.csv")

df = pd.get_dummies(df, columns=["Sex"], prefix="Sex").astype(
    {"Sex_female": int, "Sex_male": int}
)

df.to_csv("data/titanic.csv", index=False)
print(f"{list(df.columns)}")
#df.to_csv("data/final.csv", index=False)