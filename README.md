# Grupo 4 — Regressão Linear Simples com `amazon.csv` (Explicação do Código)

Este documento explica, linha a linha, o código usado para responder aos **Temas 4.1 e 4.2** da atividade (Incêndios Florestais no Brasil), mantendo o mesmo estilo “básico” visto em aula.

---

## Visão geral

Você vai:
1. **Carregar** o arquivo `amazon.csv`.
2. **Filtrar** linhas do estado específico (Amazonas; depois Mato Grosso e Rondônia).
3. **Selecionar colunas** para `X` (variável explicativa) e `y` (alvo).
4. **Treinar** um modelo de Regressão Linear Simples com `LinearRegression`.
5. **Prever** valores usando o modelo treinado.
6. **Visualizar** com gráficos de dispersão e a reta de tendência.

---

## Bibliotecas usadas

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
```

- **pandas (`pd`)**: leitura de CSV, filtragem de linhas, seleção de colunas e `DataFrame`.
- **sklearn.linear_model.LinearRegression**: modelo de **regressão linear** (ajusta uma reta aos dados).
- **matplotlib.pyplot (`plt`)**: criação dos **gráficos** (dispersão + linha do modelo).

---

## Leitura do dataset

```python
df = pd.read_csv("amazon.csv", encoding="latin1")
```

- **`pd.read_csv(...)`**: lê o arquivo CSV e cria um `DataFrame` chamado `df`.
- **`encoding="latin1"`**: informa a codificação do arquivo; evita erros de leitura em alguns datasets brasileiros.

---

## Tema 4.1 — Amazonas: `year` → `number`

### 1) Filtrar o estado Amazonas
```python
df_am = df[df['state'] == 'Amazonas']
```
- Cria um novo `DataFrame` apenas com as linhas em que a coluna **`state`** é **`Amazonas`**.

### 2) Preparar `X` (entrada) e `y` (alvo)
```python
X = df_am[['year']]
y = df_am[['number']]
```
- **`X`**: variáveis explicativas (aqui, apenas a coluna `year`).  
  > Mantemos como `DataFrame` de 2 dimensões, por isso usamos **`[['year']]`** (duplas chaves).
- **`y`**: variável alvo (número de incêndios, `number`). Também fica 2D: **`[['number']]`**.

### 3) Mensagem de controle (console)
```python
print('Dados preparados para predição (Amazonas)')
```
- Escreve no console para você acompanhar o fluxo.

### 4) Instanciar e treinar o modelo
```python
model = LinearRegression()
model.fit(X, y)
```
- **`LinearRegression()`**: cria o modelo de regressão linear **vazio**.
- **`model.fit(X, y)`**: **ajusta** o modelo aos dados. Internamente encontra a melhor reta:  
  \\( \\, \hat{y} = a \cdot X + b \\, \\).

### 5) Mensagem de sucesso
```python
print('Modelo treinado com sucesso (Amazonas)')
```

### 6) Montar o valor para prever
```python
ano = pd.DataFrame({'year': [2020]})
```
- Cria um `DataFrame` com a(s) entrada(s) para predição. Aqui, queremos a previsão **para 2020**.

### 7) Fazer a predição
```python
pred_am = model.predict(ano)
```
- **`model.predict(...)`**: aplica o modelo treinado às entradas de `ano`.  
- Retorna um array (ou `ndarray`) com a(s) previsão(ões).

### 8) Mostrar o resultado
```python
print(f'Número previsto de incêndios no Amazonas em {ano["year"][0]}: {pred_am[0][0]:.2f}')
```
- **`f-string`**: formata o texto e insere os valores.  
- **`{ano["year"][0]}`** pega o **primeiro** valor da coluna `year`.
- **`{pred_am[0][0]:.2f}`** formata a previsão com **duas casas decimais**.

### 9) Gráfico de dispersão + reta do modelo
```python
plt.scatter(X, y)
plt.plot(X, model.predict(X), color='red')
plt.xlabel('Ano')
plt.ylabel('Número de incêndios')
plt.title('Amazonas - year vs number')
plt.show()
```
- **`plt.scatter(X, y)`**: cada ponto representa um par (`year`, `number`).
- **`plt.plot(X, model.predict(X), ...)`**: desenha a **reta** ajustada pelo modelo.  
- **`xlabel`/`ylabel`/`title`**: rótulos e título do gráfico.
- **`plt.show()`**: exibe o gráfico.

> **Interpretação**: Se a linha fica “ascendente”, tende a sugerir aumento com o tempo; se é “descendente”, queda. A dispersão dos pontos indica quão bem a reta “pega” a relação.

---

## Tema 4.2 — Mato Grosso (y) vs Rondônia (X)

Neste tema, queremos **alinhar por data** os números de incêndios de **Mato Grosso** e **Rondônia**, colocar lado a lado, e então ajustar um modelo para **prever MT a partir de RO**.

### 1) Selecionar e juntar por data
```python
df_mt = df[df['state'] == 'Mato Grosso'][['date','number']]
df_ro = df[df['state'] == 'Rondonia'][['date','number']]
df_merge = pd.merge(df_mt, df_ro, on='date')
```
- `df_mt`: linhas apenas do **Mato Grosso**, com as colunas `date` e `number`.
- `df_ro`: linhas apenas de **Rondônia**, com `date` e `number`.
- **`pd.merge(..., on='date')`**: une os dois dataframes **pela coluna `date`**, criando um dataframe com **duas colunas `number`**:  
  - **`number_x`**: “number” proveniente de **df_mt** (Mato Grosso).  
  - **`number_y`**: “number” proveniente de **df_ro** (Rondônia).

### 2) Preparar `X` (Rondônia) e `y` (Mato Grosso)
```python
X2 = df_merge[['number_y']]
y2 = df_merge[['number_x']]
```
- **`X2`**: coluna `number_y` (Rondônia) — variável explicativa.
- **`y2`**: coluna `number_x` (Mato Grosso) — alvo.

### 3) Mensagem de controle
```python
print('Dados preparados para predição (MT x RO)')
```

### 4) Treinar o modelo
```python
model2 = LinearRegression()
model2.fit(X2, y2)
```
- Mesmo procedimento do Tema 4.1, agora com `X2` e `y2`.

### 5) Mensagem de sucesso
```python
print('Modelo treinado com sucesso (MT x RO)')
```

### 6) Predizer para um valor específico de RO
```python
ro_val = pd.DataFrame({'number_y': [500]})
pred_mt = model2.predict(ro_val)
print(f'Se Rondônia = {ro_val["number_y"][0]} incêndios, Mato Grosso = {pred_mt[0][0]:.2f}')
```
- Cria um `DataFrame` com o valor **`number_y = 500`** (incêndios em Rondônia).
- **`model2.predict(ro_val)`**: calcula a previsão de **MT** (y2) para esse valor de **RO**.

### 7) Gráfico de dispersão + reta do modelo
```python
plt.scatter(X2, y2)
plt.plot(X2, model2.predict(X2), color='green')
plt.xlabel('Incêndios em Rondônia')
plt.ylabel('Incêndios em Mato Grosso')
plt.title('MT x RO - number')
plt.show()
```
- **`plt.scatter(X2, y2)`**: pontos (RO vs MT) alinhados por data.
- **`plt.plot(X2, model2.predict(X2), ...)`**: reta prevista pelo modelo.
- Rótulos e título para identificar os eixos e o gráfico.

---

## Dicas de interpretação

- **Correlação visual**: se os pontos acompanham a linha de tendência, a relação linear é mais forte. Se estão muito espalhados, a relação é fraca.
- **Predição**: a previsão é **um valor esperado segundo a reta**; não garante o valor real para um mês específico.
- **Sazonalidade / não linearidade**: se os dados tiverem padrão sazonal ou curva, um modelo linear simples pode não capturar bem.

---

## Possíveis erros e soluções rápidas

- **Arquivo não encontrado**: garanta que `amazon.csv` foi **enviado** ao Colab e está na **mesma pasta** onde o notebook roda.
- **Tipos de dados**: se o CSV vier com dados em formatos estranhos, o `LinearRegression` pode reclamar. No código básico, partimos do pressuposto de que as colunas já estão legíveis (como usado em aula).
- **Colunas ausentes**: verifique se `state`, `year`, `number` e `date` existem no seu CSV.

---

## Resumo do fluxo para cada tema

1. **Filtrar** as linhas relevantes do estado.
2. **Separar** `X` (explicativa) e `y` (alvo).
3. **Treinar** `LinearRegression` com `fit`.
4. **Prever** com `predict`.
5. **Plotar** os dados e a reta.

Pronto! Esse é o “esqueleto” do que está acontecendo em cada linha do seu código.
