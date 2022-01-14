import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler #normalisação usando método z-score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV

df=pd.read_csv("car_clean.csv",index_col=False)

print(df.head(10))

# criando um objeto de regressão linear

lm=LinearRegression()

print(lm)

# vamos treinar o modelo usando "highway-L/100km" como preditor e "price" como resposta

x=df[["highway-L/100km"]]
y=df["price"]

# fitando o modelo

lm.fit(x,y)

# criando uma previsão

y_pred=lm.predict(x)

print(y_pred[0:4])

# encontrando os coeficientes da reta

a=lm.intercept_
b=lm.coef_

print("a=",a,"b =",b[0])

print("Para highway-L/100km de ",df.iloc[10]["highway-L/100km"],
"\n preço no df: ",df.iloc[10]["price"],
"\n preço da previsão: ",a+b[0]*df.iloc[10]["highway-L/100km"])

# vamos fazer outra regressão linear desta vez para "engine-size" e "price"

lm1=LinearRegression()
lm1.fit(df[["engine-size"]],df["price"])

a1=lm1.intercept_
b1=lm1.coef_

print("a1=",a1,"b1 =",b1[0])

print("Para engine-size de ",df.iloc[11]["engine-size"],
"\n preço no df: ",df.iloc[11]["price"],
"\n preço da previsão: ",a1+b1[0]*df.iloc[11]["engine-size"])

# agora vamos fazer regressão multilinear

z=df[['horsepower', 'curb-weight', 'engine-size', 'highway-L/100km']]

lm.fit(z,df["price"])

a=lm.intercept_

b1, b2, b3, b4=lm.coef_

print(a,b1,b2,b3,b4)


print("Para horsepower de ",df.iloc[1]["horsepower"],
"\n curb-weight de ",df.iloc[1]["curb-weight"],
"\n engine-size de ",df.iloc[1]["engine-size"],
"\n highway-L/100km de ",df.iloc[1]["highway-L/100km"],
"\n preço no df: ",df.iloc[1]["price"],
"\n preço da previsão: ",
a+b1*df.iloc[1]["horsepower"]+b2*df.iloc[1]["curb-weight"]+b3*df.iloc[1]["engine-size"]+b4*df.iloc[1]["highway-L/100km"])

# vamos fazer outra regressão multilinear agora usando "city-L/100km" e "normalized-losses" para prever "price"

lm2=LinearRegression()
lm2.fit(df[["city-L/100km","normalized-losses"]],df["price"])

a=lm2.intercept_

b1, b2=lm2.coef_

print("Para city-L/100km de ",df.iloc[1]["city-L/100km"],
"\n normalized-losses de ",df.iloc[1]["normalized-losses"],
"\n preço no df: ",df.iloc[1]["price"],
"\n preço da previsão: ",
a+b1*df.iloc[1]["city-L/100km"]+b2*df.iloc[1]["normalized-losses"])

# visualizando se a regressão é boa previsora

# primeiro para "highway-L/100km" e "price"

plt.clf() 

width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-L/100km", y="price", data=df)
plt.ylim(0,)

plt.savefig("regplot_highway-price.png")
plt.close()

# agora para "peak-rpm" e "price"

plt.clf() 

plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)

plt.savefig("regplot_rpm-price.png")
plt.close()

# podemos comparar os gráficos com as correlações 

print(df[["peak-rpm","highway-L/100km","price"]].corr())

# fazendo plotes residuais para saber se a regressão é boa

plt.clf() 

width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(x=df["highway-L/100km"],y=df["price"])

plt.savefig("resid_highway-price.png")
plt.close()

# vamos fazer o plote residual de engine-size e price
plt.clf() 

plt.figure(figsize=(width, height))
sns.residplot(x=df["engine-size"],y=df["price"])

plt.savefig("resid_engine-price.png")
plt.close()
# para a regressão multilinear vamos fazer um plot da distribuição dos preços previstos vs os preços da tabela

# primeiro vamos criar a previsão

y_pred=lm.predict(z)

# vamos usar kdeplot para plotar as duas distribuições

plt.clf() 
fig, ax = plt.subplots()

sns.kdeplot(df["price"], color="black", label="Valores reais", ax=ax)
ax.legend()
sns.kdeplot(y_pred, color="red", label="Valores previstos", ax=ax)
ax.legend()

# colocando labels e título

plt.title("Preços reais vs fitados")
plt.xlabel("Preço (em dólares)")
plt.ylabel("Proporção de carros")

plt.savefig("dist.png")
plt.close()

# vamos fazer regressão polinomial
# primeiro vamos definir as variáveis x e y

x=df["highway-L/100km"]
y=df["price"]

# vamos fitar o polinômio cúbico usando polyfit 
# e depois criar a função polinomial usando poly1d

fit=np.polyfit(x,y,3)
f=np.poly1d(fit)

print(f)

# vamos plotar a interpolação vs os dados reais

def plot_polyxreal(y_fitado,y_real,x_real,x_nome):
    x_fit=np.linspace(df[x_nome].min(),df[x_nome].max(),100)
    
    plt.clf() 
    plt.plot(x_real, y_real,".")
    plt.plot(x_fit, y_fitado(x_fit),"-")
    plt.title("Fit polinomial vs valores reais")
    plt.xlabel(x_nome)
    plt.ylabel("price")
    
plot_polyxreal(f,y,x,"highway-L/100km")

plt.savefig("polyfit_highway-price.png")
plt.close()

# vamos vazer outro fit polinomial agora para engine-size por price

x=df["engine-size"]
y=df["price"]
fit=np.polyfit(x,y,4)
f=np.poly1d(fit)

print(f)

plot_polyxreal(f,y,x,"engine-size")

plt.savefig("polyfit_engine-price.png")
plt.close()

# vamos fazer regressão multipolinomial
# começamos criando o objeto PolynomialFeatures

mp=PolynomialFeatures(degree=2)

print(mp)

# vamos fazer o fit

z_fit=mp.fit_transform(z) # fita os dados e retorna novos valores de acordo com o fit

print(z.shape,z_fit.shape)

# vamos criar um pipeline
# lista de etapas do pipeline 

Input=[("scaler", StandardScaler()),("polynomial", PolynomialFeatures(include_bias=False)), ("linear", LinearRegression())]

# criando o pipeline

pipe=Pipeline(Input)

print(pipe)

# vamos utilizar o pipe para fazer uma normalização, uma transformação polinomial e um fit

p1=pipe.fit(z,y)

print(p1)

# vamos utilizar o pipe para fazer uma normalização, uma transformação polinomial e fazer uma previsão

p1_pred=pipe.predict(z)

print(p1_pred[0:4])

# vamos calcular o quão preciso nossas regressões são
# vamos fazer um fit linear

x=df[["engine-size"]]
lm.fit(x,y)

# vamos calcular o R-square

r=lm.score(x,y)

print("Podemos dizer que ",r*100,"% da variação de preços é explicada por esta regressão linear.")

# agora vamos calcular o Mean Squared Error (MSE)
# primeiro temos que fazer uma previsão

y_prev=lm.predict(x)

# agora vamos calcular o mse

mse=mean_squared_error(df["price"],y_prev)

print("O mse entre os preços previstos e os reais é: ",mse)

# vamos calcular R^2 e MSE para regressão multilinear

lm.fit(z, df['price'])
r_mtl=lm.score(z,df["price"])

y_mtl=lm.predict(z)
mse_mtl=mean_squared_error(df["price"],y_mtl)

print("Podemos dizer que ",r_mtl*100,"% da variação de preços é explicada por esta regressão multilinear.")
print("O mse entre os preços previstos e os reais é: ",mse_mtl)

# agora para o caso polinomial

r_poly=r2_score(y,f(x))

mse_poly=mean_squared_error(df["price"],f(x))

print("Podemos dizer que ",r_poly*100,"% da variação de preços é explicada por esta regressão polinomial.")
print("O mse entre os preços previstos e os reais é: ",mse_poly)

# pegando apenas os dados numéricos de um dataframe

dfn=df._get_numeric_data()

print(dfn.head(),dfn.info())

def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    plt.clf() 
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    fig, ax = plt.subplots()
    
    sns.kdeplot(RedFunction, color="red", label=RedName, ax=ax)
    ax.legend()
    sns.kdeplot(BlueFunction, color="blue", label=BlueName, ax=ax)
    ax.legend()
    plt.title(Title)
    plt.xlabel("Preço (em dólares)")
    plt.ylabel("Proporção de carros")

def PollyPlot(x_train, y_train, x_test, y_test, x_prevision, y_prevision):
    plt.clf() 
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
 
    plt.plot(x_train, y_train, 'ro', label='Dados de treino')
    plt.plot(x_test, y_test, 'go', label='Dados de teste')
    plt.plot(x_prevision, y_prevision, label='Função previsão')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()

# vamos separar nossos dados em x e y

y_data=dfn["price"]

x_data=dfn.drop("price",axis=1)

# vamos separar nossos dados em dados de treinamento e teste usando a função train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, test_size=0.1,random_state=4)

# test_size=0.10 significa que 10% dos dados será usada para teste

print("amostra de teste :", x_test.shape[0])
print("amostra para treinamento:",x_train.shape[0])

# vamos usar regressão linear para "horsepower" por "price"

lm.fit(x_train[["horsepower"]],y_train)

# Vamos calcular R^2 nos dados de treino e de teste e comparar

r2_train=lm.score(x_train[["horsepower"]],y_train)

r2_test=lm.score(x_test[["horsepower"]],y_test)

print("r2_train =", r2_train,"r2_test =", r2_test)

# agora vamos usar o modelo Cross-validation Score

cross=cross_val_score(lm, x_data[["horsepower"]], y_data, cv=4)
# cv=4 número de folds
# a saída de cross_val_score é o R^2 para cada fold

print(cross)

print("A média de R^2 para os folds é: ",cross.mean(),"e o desvio padrão é:",cross.std())

# agora vamos fazer uma previsão usando 4 folds

y_pred=cross_val_predict(lm,x_data[["horsepower"]],y_data,cv=4)

print(y_pred[0:5])

# vamos fazer um fit, uma previsão multilinear e o plote de distribuição para os dados _train(90%)

lm.fit(x_train[["horsepower", "curb-weight", "engine-size", "highway-L/100km"]], y_train)

ym_pred_train=lm.predict(x_train[["horsepower", "curb-weight", "engine-size", "highway-L/100km"]])

print(ym_pred_train[0:5])

Title= "Plote de distribuição: Dados de treino previstos vs dados reais"
DistributionPlot(y_train, ym_pred_train, "Valores reais (treino)", "Valores previstos (treino)", Title)

plt.savefig("dist2.png")
plt.close()

# novamente, agora para _test(10%)

ym_pred_test=lm.predict(x_test[["horsepower", "curb-weight", "engine-size", "highway-L/100km"]])

print(ym_pred_test[0:5])

Title= "Plote de distribuição: Dados de teste previstos vs dados reais"
DistributionPlot(y_test, ym_pred_test, "Valores reais (teste)", "Valores previstos (teste)", Title)

plt.savefig("dist3.png")
plt.close()

# vamos criar um objeto de regressão polinomial de grau 5

p5=PolynomialFeatures(degree=5)

# vamos separar nossos dados em dados de treinamento e teste novamente agora com teste=40%

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, test_size=0.42,random_state=0)

# vamos obter os valores transformados do fit para train e test 

x_train_p5=p5.fit_transform(x_train[["horsepower"]])
x_test_p5=p5.fit_transform(x_test[["horsepower"]])

print("x_train_p5",x_train_p5[0:2])

# vamos treinar um modelo de regressão linear com os dados transformados
# y= intercept_ + coef_[1] * x_1 + coef_[2] * x_2 +...+ coef_[5] * x_5
# onde x_1=x**1, x_2=x**2, ..., x_5= x**5

lm.fit(x_train_p5,y_train)

a=lm.intercept_
b=lm.coef_

print("a=",a,"b =",b)

# em seguida vamos fazer uma previsão usando os dados de teste transformados

y_prev_p5=lm.predict(x_test_p5)

# e vamos comparar os valores previstos com os reais

print("Valores previstos:", y_prev_p5[0:4])
print("Valores reais:", y_test[0:4].values)

# agora que o modelo já foi treinado vamos fazer uma previsão utilizando todos os dados
# primeiro vamos "arrumar" os dados em ordem crescente

xmax=max(x_data[["horsepower"]].values)

xmin=min(x_data[["horsepower"]].values)

x=np.arange(xmin, xmax, 0.1)

print("xmin =", xmin,"xmax =", xmax,"x =",x)

# vamos ajustar a forma de x

x_prevision=x.reshape(-1, 1)

print("x_prevision =", x_prevision)

# vamos transformar os dados ordenados e fazer a previsão

y_prev_t=lm.predict(p5.fit_transform(x_prevision))

# agora vamos plotar tudo

PollyPlot(x_train[["horsepower"]], y_train, x_test[["horsepower"]], y_test, x_prevision, y_prev_t)

plt.savefig("polly_train_test.png")
plt.close()

# vamos calcular R^2 para os dados de treino e para os de teste

r2_train=lm.score(x_train_p5, y_train)

r2_test=lm.score(x_test_p5, y_test)

print("Podemos dizer que ",r2_train*100,"% dos preços de treino pode ser explicados por este modelo.")

print("O valor negativo de r2_test=",r2_test,"indica overfitting.")

# Vamos ver como o R^2 muda nos dados de teste para polinômios de ordem diferente

r2_list=[]

ordem=[1,2,3,4]

for n in ordem:
    poly=PolynomialFeatures(degree=n)

    x_train_poly=poly.fit_transform(x_train[["horsepower"]])
    x_test_poly=poly.fit_transform(x_test[["horsepower"]])

    lm.fit(x_train_poly,y_train)
    
    a=lm.intercept_
    b=lm.coef_
    
    print("a=",a,"b =",b)

    r2_list.append(lm.score(x_train_poly, y_train))

plt.clf() 
plt.plot(ordem,r2_list)
plt.xlabel("ordem")
plt.ylabel("R^2")
plt.title("R^2 usando dados de teste")
plt.savefig("polly_R2.png")
plt.close()

# vamos definir uma função para facilitar o trabalho feito acima

def f(order, test_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
    poly = PolynomialFeatures(degree=order)
    x_train_poly = poly.fit_transform(x_train[['horsepower']])
    x_test_poly = poly.fit_transform(x_test[['horsepower']])
    lm = LinearRegression()
    lm.fit(x_train_poly,y_train)
    y_prev_t=lm.predict(poly.fit_transform(x_prevision))
    PollyPlot(x_train[["horsepower"]], y_train, x_test[["horsepower"]], y_test, x_prevision, y_prev_t)

# interact(f, order=(0, 6, 1), test_data=(0.05, 0.95, 0.05)) -> Jupyter notebook

f(3,0.25)
plt.savefig("polly_train_test2.png")
plt.close()

# Vamos fazer uma regressão multipolinomial
# primeiro vamos criar uma regressão polinomial de grau 2

p2=PolynomialFeatures(degree=2)

# vamos obter os valores transformados do fit para train e test 

x_train_p2=p2.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-L/100km']])

x_test_p2=p2.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-L/100km']])

# vamos treinar o modelo

lm.fit(x_train_p2,y_train)

b=lm.coef_

print(b.shape)

# vamos fazer uma previsão usando dados de teste

y_prev_p2=lm.predict(x_test_p2)

# vamos plotar o resultado

Title= "Plote de distribuição: Dados de previstos (multipolinomial) vs dados reais"
DistributionPlot(y_train, ym_pred_train, "Valores reais (treino)", "Valores previstos (treino)", Title)

plt.savefig("dist3.png")
plt.close()

# vamos fazer regressão ridge, (regressão com um parâmetro de amortecimento)
# vamos começar pegando os dados transformados por p2

x_train_p2=p2.fit_transform(x_train[['horsepower','curb-weight','engine-size','highway-L/100km','normalized-losses','symboling']])
x_test_p2=p2.fit_transform(x_test[['horsepower','curb-weight','engine-size','highway-L/100km','normalized-losses','symboling']])

# vamos criar um objeto de regressão ridge, com \alpha=0.1

rd=Ridge(alpha=0.1,normalize=True)

# vamos fitar o modelo, e fazer uma predição

rd.fit(x_train_p2,y_train)

ypre=rd.predict(x_test_p2)

print('predição:', ypre[0:4])
print('conjunto de teste:', y_test[0:4].values)

# vamos fazer um loop para descobrir qual valor de \alpha minimiza o erro

R2_test = []
R2_train = []
Alpha=np.array(range(0,500))

for alpha in Alpha:
    rd_loop=Ridge(alpha=alpha) 
    rd_loop.fit(x_train_p2, y_train)
    R2_test.append(rd_loop.score(x_test_p2, y_test))
    R2_train.append(rd_loop.score(x_train_p2, y_train))

plt.clf() 
plt.plot(Alpha,R2_test, label='dados de validação')
plt.plot(Alpha,R2_train, 'r', label='dados de treino')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()
plt.savefig("ridge.png")
plt.close()

# vamos usar GridSearchCV para facilitar o trabalho de achar o melhor \alpha
# vamos começar criando um dicionário para os valores de \alpha que queremos testar

parametros1=[{'alpha': [0.001,0.1,1,10,100,1000,10000,100000,100000]}]

# vamos criar um objeto de regressão ridge sem fixar \alpha
# e um objeto de grid search (cv=número de folds)

rdg=Ridge()

grs=GridSearchCV(rdg,parametros1,cv=4)

# vamos fitar o modelo

grs.fit(x_data[['horsepower','curb-weight','engine-size','highway-L/100km','normalized-losses','symboling']], y_data)

# vamos criar uma variável para armazenar o valor dos melhores parâmetros

best_alpha=grs.best_estimator_

print(best_alpha)

# vamos testar o modelo nos dados de teste

r2b=best_alpha.score(x_test[['horsepower','curb-weight','engine-size','highway-L/100km','normalized-losses','symboling']], y_test)

print(r2b)

# vamos fazer um novo modelo, agora testando também se é melhor utilizar normalize = true ou false

parametros2= [{'alpha': [0.001,0.1,1, 10, 100, 1000,10000,100000,100000],'normalize':[True,False]} ]
grs2=GridSearchCV(Ridge(), parametros2,cv=4)
grs2.fit(x_data[['horsepower','curb-weight','engine-size','highway-L/100km','normalized-losses','symboling']],y_data)
best_p=grs2.best_estimator_

print(best_p)