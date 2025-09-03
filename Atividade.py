import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv("amazon.csv", encoding="latin1")

display(df.head()) # pegar os 5 primeiros

print("Informações básicas do dataset:")
df.info()  #Informaçao da tabela

print("Estatísticas descritivas:")
display(df.describe())

df_am = df[df['state'] == 'Amazonas']
X = df_am[['year']]
y = df_am[['number']]
print('Dados preparados para predição (Amazonas)')
model = LinearRegression()
model.fit(X, y)
print('Modelo treinado com sucesso (Amazonas)')
ano = pd.DataFrame({'year': [2020]})
pred_am = model.predict(ano)
print(f'Número previsto de incêndios no Amazonas em {ano["year"][0]}: {pred_am[0][0]:.2f}')
plt.scatter(X, y)
plt.plot(X, model.predict(X), color='red')
plt.xlabel('Ano')
plt.ylabel('Número de incêndios')
plt.title('Amazonas - year vs number')
plt.show()

df_mt = df[df['state'] == 'Mato Grosso'][['date','number']]
df_ro = df[df['state'] == 'Rondonia'][['date','number']]
df_merge = pd.merge(df_mt, df_ro, on='date')
X2 = df_merge[['number_y']]
y2 = df_merge[['number_x']]
print('Dados preparados para predição (MT x RO)')
model2 = LinearRegression()
model2.fit(X2, y2)
print('Modelo treinado com sucesso (MT x RO)')
ro_val = pd.DataFrame({'number_y': [500]})
pred_mt = model2.predict(ro_val)
print(f'Se Rondônia = {ro_val["number_y"][0]} incêndios, Mato Grosso = {pred_mt[0][0]:.2f}')
plt.scatter(X2, y2)
plt.plot(X2, model2.predict(X2), color='green')
plt.xlabel('Incêndios em Rondônia')
plt.ylabel('Incêndios em Mato Grosso')
plt.title('MT x RO - number')
plt.show()
