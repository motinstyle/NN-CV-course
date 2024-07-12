import pandas as pd

df = pd.read_csv("submission1.csv")
df["id"] = df['id'].str.replace('./test\\unknown\\', '')
df.set_index('id', inplace=True)
print(df.head(n=6))
df.to_csv("submission.csv")

