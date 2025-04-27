from openbb import obb

#print(obb)

hist = obb.equity.price.historical('AAPL', '2023-01-01', '2023-10-01')
df = hist.to_df()
print(df)
