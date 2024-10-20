nba["College"].fillna("No College", inplace = True)

df['A'].fillna(df['A'].median(), inplace=True)

nba["College"].fillna( method ='ffill', limit = 1, inplace = True)

df.dropna(inplace=True) #dropping the row
