# trabalhando com arrays no numpy

altura=[1.60,1.85,1.70,1.58]
peso=[51.5,111.1,80.3,49.9]

import numpy as np # importando o numpy

np_altura=np.array(altura)
np_peso=np.array(peso)

# imprimindo o tipo de np_altura

print(type(np_altura))

# calculando IMC 

imc=np_peso/np_altura**2

# se tentamos trabalhar com peso/altura**2 recebemos uma mensagem de erro

print(imc)

# encontrando subgrupos

print(imc>23) #mostra false or true

print(imc[imc>23]) # mostra quais são true

# convertendo Kg para lbs

np_peso_lbs=2.2*np_peso

print("O valor dos pesos %s em libras é %s." % (np_peso,np_peso_lbs))

# criando DataFrame no pandas usando dicionários

info_paises={
    "pais":["Brasil","Argentina","Bolívia","Chile"],
    "capital":["Brasília","Buenos Aires","La Paz","Santiago"],
    "area":[8.51,2.78,1.10,0.76],
    "populacao":[211.76,43.59,10.97,18.05]
}

import pandas as pd

s_amer=pd.DataFrame(info_paises)

print(s_amer)

# mudando os índices de 0 a 3 para sigla dos países

s_amer.index=["BR","AR","BO","CL"]

print(s_amer)

# outra forma de criar um DataFrame é impotando um arquivo .csv

acoes=pd.read_csv("acoes2.csv",index_col=0) 
#index_col=0 significa que os dados da primeira coluna serão os índices

print(acoes)

print("Imprimindo acoes['valor']")
print(acoes["valor"]) # imprime coluna empresa como Series

print("Imprimindo acoes[['valor']]")
print(acoes[["valor"]]) # imprime coluna valor como DataFrame

# podemos imprimir mais de uma coluna na forma de DataFrame

print(s_amer[["pais","populacao"]])

# imprimindo linhas do DataFrame

print(acoes[0:1]) # imprime a linha 1

print(s_amer[1:3]) # imprime as linhas 2 e 3

# usando .loc e .iloc

print(s_amer.loc["BR","populacao"],"\n",s_amer.loc[["BR","CL"]])

# .loc[a,b] imprime elemento b do indice a, e .loc[[c,d]] imprime linhas dos indices c e d

print(s_amer.iloc[0,3],"\n",s_amer.iloc[[0,3]])

# .iloc mesmo que .loc mas com indices numéricos

# Vamos importar o conjunto de dados Iris, que contém informações sobre flores, e que já vem com a biblioteca seaborn
#import seaborn
#iris=seaborn.load_dataset("iris",cache=True,data_home='/local_para_armazenar_os_dados')

# importando .csv no pandas
csv_file="iris.csv"

df=pd.read_csv(csv_file)

print(df.head(3))

# fazendo um novo DataFrame a partir do anterior

df_sepal=df[["sepal_length","sepal_width"]]

print(df_sepal.head(3))

# criando um "sub"DataFrame usando .loc ou .iloc

df_sample=df.iloc[10:20:2,0:2]

print(df_sample)

# visualizando as espécies da tabela iris

print(df["species"].unique())

# visualizando valores maiores ou menores que um valor dado

print(df_sample["sepal_length"]>5,"\n",df_sample[df_sample["sepal_length"]>5])

# criando um novo DataFrame apenas da espécie setosa

df_setosa=df[df["species"]=="setosa"]

print(df_setosa.head(3))

df_setosa.to_csv("iris_setosa.csv")

# importando dados da internet

url="https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

df=pd.read_csv(url, header=None)

print(df.head(3))
print(df.tail(3))

# vamos colocar nomes nas colunas

headers=[]

nomes="symboling,normalized-losses,make,fuel-type,aspiration,num-of-doors,body-style,drive-wheels,engine-location,wheel-base,length,width,height,curb-weight,engine-type,num-of-cylinders,engine-size,fuel-system,bore,stroke,compression-ratio,horsepower,peak-rpm,city-mpg,highway-mpg,price"

headers=nomes.split(",")

print(headers)

df.columns=headers

print(df.head(3))

# criando um arquivo no computador com os dados com cabeçalho

df.to_csv("car_price.csv", index=False)

# Vamos visualizar se as atribuições de tipo de dado estão corretas

print(df.dtypes)

# Vamos repassar o símbolo "?" por NaN 
# para que o método dropna() possa ser usado depois para retirar valores faltando

df1=df.replace("?",np.NaN)

# identificando se um valor é um valor faltando

missing_data=df1.isnull() # função 'oposta' .notnull()

print(missing_data.head(5))

# contando os valores faltando em cada coluna

for coluna in missing_data.columns.values.tolist():
    print(coluna)
    print(missing_data[coluna].value_counts())
    print("")

# agora vamos jogar fora as linhas que não possuem preço

df=df1.dropna(subset=["price"], axis=0) 

# axis=1 deleta coluna com valores nan
# opção inplace=true modifica o dataframe

# vamos resetar os índices depois de ter deletado algumas linhas

df.reset_index(drop=True,inplace=True)

# usando resumo estatístico para uma coluna específica

print(df[["make","price"]].describe(include="all"))

# Mudando o dtype da coluna normalized-losses de object para float

df=df.astype({"normalized-losses":"float","price":"float","bore":"float", "stroke":"float","peak-rpm":"float"})

print(df["normalized-losses"].dtypes)

# Calculando o valor médio dos valores da coluna normalized-losses

media=df["normalized-losses"].mean()

print("Média de normalized-losses:", media)

# repassando os valores NaN pelo valor médio

df["normalized-losses"]=df["normalized-losses"].replace(np.NaN,media)

print(df["normalized-losses"])

# outro modo de calcular média e repassar valores NaN, agora para "bore"

avg_bore=df['bore'].astype('float').mean(axis=0)
print("Average of bore:", avg_bore)

df["bore"].replace(np.nan, avg_bore, inplace=True)

avg_stroke = df["stroke"].astype("float").mean(axis = 0)
print("Average of stroke:", avg_stroke)

df["stroke"].replace(np.nan, avg_stroke, inplace = True)

avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
print("Average horsepower:", avg_horsepower)

df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)

avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
print("Average peak rpm:", avg_peakrpm)

df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)

# vamos contar qual "num-of-doors" é mais frequente

print(df["num-of-doors"].value_counts())

print(df["num-of-doors"].value_counts().idxmax())

# vamos repassar o "num-of-doors" com NaN por "four" 

df["num-of-doors"].replace(np.nan,"four",inplace=True)

# fazendo mudança de unidades de mpg para L/100km na coluna "city-mpg"

df["city-mpg"]=235/df["city-mpg"]

df.rename(columns={"city-mpg":"city-L/100km"},inplace=True)

df["highway-L/100km"]=235/df["highway-mpg"] #adiciona nova coluna

print(df)

# normalizando a caracteristica "length" usando método scaling

df["length"]=df["length"]/df["length"].max()

print(df["length"].head())

# normalizando a caracteristica "width" usando método min-max

df["width"]=(df["width"]-df["width"].min())/(df["width"].max()-df["width"].min())

print(df["width"].head())

# normalizando a caracteristica "height" usando método z-score

df["height"]=(df["height"]-df["height"].mean())/df["height"].std()

print(df["height"].head())

# transformando em variáveis categoricas
# usando linspace para achar 4 valores igualmente espaçados em um dado intervalo

bins=np.linspace(min(df["price"]),max(df["price"]),4)

# criando lista com nomes de cada categoria

group_names=["Barato","Mediano","Caro"]

# acrescentando coluna dos preços categorizados

df["preço-binned"]=pd.cut(df["price"],bins,labels=group_names,include_lowest=True)

# vamos fazer o mesmo para "horsepower"

df["horsepower"]=df["horsepower"].astype(int,copy=True)

bins_horse=np.linspace(min(df["horsepower"]),max(df["horsepower"]),4)

cavalos=["baixo","médio","alto"]

df["horsepower-binned"]=pd.cut(df["horsepower"],bins_horse,labels=cavalos,include_lowest=True)

print(df[["price","preço-binned",'horsepower','horsepower-binned']])

print(df["horsepower-binned"].value_counts(),"\n",df["preço-binned"].value_counts())

# transformando variável categórica em variavel numérica

variavel_dummy=pd.get_dummies(df["fuel-type"])

# renomeando colunas 

variavel_dummy.rename(columns={'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'}, inplace=True)

# juntando df com variavel_dummy

df=pd.concat([df,variavel_dummy],axis=1)

# deletando coluna original "fuel-type"

df.drop("fuel-type",axis=1,inplace=True)

print(df.head())

# fazendo o mesmo para "aspiration"

dummy_variable_2 = pd.get_dummies(df["aspiration"])
dummy_variable_2.rename(columns={'std':'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace=True)
df = pd.concat([df, dummy_variable_2], axis=1)
df.drop("aspiration", axis = 1, inplace=True)

# arrumando colunas que seram salvas

df=df[['symboling', 'normalized-losses', 'make', 'num-of-doors', 'body-style',
       'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width',
       'height', 'curb-weight', 'engine-type', 'num-of-cylinders',
       'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio',
       'horsepower', 'peak-rpm', 'city-L/100km', 'highway-L/100km', 'preço-binned', 'horsepower-binned',
       'fuel-type-diesel', 'fuel-type-gas', 'aspiration-std',
       'aspiration-turbo', 'price']]

# salvando

df.to_csv('car_clean.csv', index=False)

print(df)