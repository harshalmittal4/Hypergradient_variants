import pandas as pd

df = pd.DataFrame(columns = ['sgd_HdAdam', 'sgd_Hd', 'sgd_Hdmomentum', 'sgd', 'sgdn_Hd',
 'sgdn_Hdmomentum', 'sgdn', 'adam_HdAdam', 'adam', 'adam_Hd'])

 df['alpha0'] = pd.Series([0.01, 0.005, 0.001, 0.0005, 0.0001])

 def insert_df(optim, iterations, alpha):
 	df.loc[df['alpha0']==alpha, optim] = iterations
 	