from sklearn.model_selection import train_test_split
import pandas as pd

df = 'Diffusion_Data.csv'
df = pd.read_csv(df)

train = df.loc[df['group'] != 'non-transition-metal']
test_cd = df.loc[df['group'] == 'non-transition-metal']

train, test = train_test_split(train, test_size=0.2)
test = pd.concat([test, test_cd])

train.to_csv('Diffusion_Data_train_subset.csv', index=False)
test.to_csv('Diffusion_Data_test_subset.csv', index=False)

print(train)
print(test)

df = 'Supercon_data_features_selected.xlsx'
df = pd.read_excel(df)

train = df.loc[df['group'] != 'Fe-based']
test_cd = df.loc[df['group'] == 'Fe-based']

train, test = train_test_split(train, test_size=0.2)
test = pd.concat([test, test_cd])

train.to_csv('Supercon_data_features_selected_train_subset.csv', index=False)
test.to_csv('Supercon_data_features_selected_test_subset.csv', index=False)

print(train)
print(test)
