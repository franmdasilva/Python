import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats # Statistical functions
from scipy.stats import f_oneway # Perform one-way ANOVA.

df=pd.read_csv("car_clean.csv",index_col=False)

print(df.head(10))

print(df.info())

# vamos visualizar um resumo estatístico de alguns dados numéricos

print(df[["symboling","normalized-losses","wheel-base","price"]].describe())

# resumo estatístico de todos os dados, inclusive dados de string

print(df.describe(),"\n",df.describe(include=['object']))

# resumo das variáveis categóricas

preço_binned_counts=df["preço-binned"].value_counts()

preço_binned_counts.name="Counts"
preço_binned_counts.index.name="binned_price"

print("\n",preço_binned_counts)

# resumo das variáveis categóricas, saída como dataframe

drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts.index.name = 'drive-wheels'

print(drive_wheels_counts)

# vamos calcular a correlação entre os dados do tipo int e float

print(df.corr())

# vamos fazer um heatmap da correlação entre as variáveis numéricas

plt.clf() 

plt.figure(figsize=(13,13))
sns.heatmap(df.corr(), cmap="RdBu")
plt.savefig("heat_correlation.png")

print(df[['bore','stroke' ,'compression-ratio','horsepower']].corr())

# plotando boxplots comparando variáveis categóricas e numéricas

plt.clf() 

sns.boxplot(x="drive-wheels", y="price", data=df)

# vizualizando
# plt.show()

# salvando

plt.savefig("box_wheels-price.png")

# novamente agora para "body-style"

plt.clf() # cleaning memory
sns.boxplot(x="body-style", y="price", data=df)

plt.savefig("box_style-price.png")

# plotando scatterplot para comparação de duas variáveis numéricas 

plt.clf() 
sns.scatterplot(data=df, x="engine-size", y="price")

plt.title("Scatterplot of Engine Size vs Price")
plt.savefig("scatter_engine-price.png")

print("\n Correlação entre tamanho do motor e preço do carro: \n",df[["engine-size", "price"]].corr())

# plotando scatterplot para "city-L/100km" mais a linha de regressão ajustada para os dados.

plt.clf() 
sns.regplot(x="city-L/100km", y="price", data=df)
plt.ylim(0,)

plt.savefig("scatter_city-price.png")

# usando groupby para análise de variáveis categóricas

df_teste=df[["drive-wheels","body-style","price"]]

df_grp=df_teste.groupby(["drive-wheels","body-style"], as_index=False).mean()

# melhorando a vizualização da tabela

df_pivot=df_grp.pivot(index="drive-wheels", columns="body-style")

print(df_grp,"\n",df_pivot)

#criando um heatmap da tabela acima com matplotlib

plt.clf() 
plt.pcolor(df_pivot, cmap="RdBu")
plt.colorbar()

# o mesmo agora com seaborn

plt.clf()

h=sns.heatmap(df_pivot, cmap="RdBu")
plt.savefig("heat_wheels-style.png")

# vamos fazer outro heatmap agora para "num-of-doors" e "horsepower-binned"

plt.clf()
df_teste=df[["num-of-doors","horsepower-binned","price"]]

df_grp=df_teste.groupby(["num-of-doors","horsepower-binned"], as_index=False).mean()

df_pivot=df_grp.pivot(index="num-of-doors", columns="horsepower-binned")
df_pivot=df_pivot.fillna(0) #preenche valores faltando com 0

print("\n",df_grp,"\n",df_pivot)

h=sns.heatmap(df_pivot, cmap="RdBu")
plt.savefig("heat_doors-horse.png")

# vamos plotar o preço médio de cada marca de carro

media=df[["make","price"]].groupby(["make"],as_index=False).mean().sort_values(by=["price"])

print(media)

plt.clf()
plt.figure(figsize=(7,9))
p=sns.barplot(x="make", y="price", data=media)
p.set_xticklabels(p.get_xticklabels(),rotation=80)
plt.savefig("bar_make-price.png")

# usando o método ANOVA "Analysis of Variance"

df_anova=df[["make","price"]]
anova_agrup=df_anova.groupby(["make"])

anova_caso1=f_oneway(anova_agrup.get_group("honda")["price"],anova_agrup.get_group("subaru")["price"])


anova_caso2=f_oneway(anova_agrup.get_group("honda")["price"],anova_agrup.get_group("jaguar")["price"])

print(anova_caso1,"\n",anova_caso2)

# vamos calcular ANOVA de novo agora para drive-wheels e price

df_teste=df[["drive-wheels","price"]]
grup_roda=df_teste[["drive-wheels","price"]].groupby(["drive-wheels"])

print(grup_roda.head(2))

# visualizando um grupo específico

quatro=grup_roda.get_group("4wd")["price"]

print(quatro)

# ANOVA
f_val, p_val=stats.f_oneway(grup_roda.get_group('fwd')['price'],grup_roda.get_group('rwd')['price'],grup_roda.get_group('4wd')['price'])  
 
print( "ANOVA results: F=", f_val, ", P =", p_val)

# vamos calcular a correlação Pearson e o p-value para "wheel-base" e "price"

pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])

print("O valor da correlação Pearson entre wheel-base e price é: ",pearson_coef)
print("E o valor do p-value é: ",p_value)

# novamente agora para 'horsepower' e 'price'

pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("O valor da correlação Pearson entre horsepower e price é: ", pearson_coef, " com um P-value de P = ", p_value)  
