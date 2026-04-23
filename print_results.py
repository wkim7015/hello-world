import pandas as pd

df = pd.read_csv('portfolio_values.csv', index_col=0, parse_dates=True).squeeze()

print(f"Total Bias-Free Return: {(df.iloc[-1] / df.iloc[0]) - 1:.2%}")
print("Check logs for out-of-sample strategy results.")
