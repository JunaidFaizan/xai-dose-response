import pandas as pd
R=r"results\dose_response\\"
p=pd.read_csv(R+"preference.csv"); print(p.columns.tolist()); print(p.comparison.unique()[:10] if 'comparison' in p else p.head())
