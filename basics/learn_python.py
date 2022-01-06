#imprimindo:

print("Usando \" dentro de \" \".")
print('Usando \' dentro de \' \'.')

a=["s",'d',1,3]
print(a)

# recuos são usados para formar blocos de comandos
# o recuo padrão são 4 espaços (opcional)
# menos de 4 espaços também funcionam

x=1
if x==1:
    print("x vale 1.")

# definindo um número inteiro:

a=1
print(a)

# definindo um número com casas decimais:

b=1.
c=1.0
d=float(1)
e=bool(1)
print(b,c,d,e)

# definindo strings (palavras):

p1='palavra1'
print(p1)

p2="palavra2"
print(p2)

# operações simples

um=1
dois=2
tres=um+dois

print(tres)

x=1+2-4*9/3. 
y=5%3 # resto da divisão
z=2**3 # potência
w=11/3 # divisão 
k=11//3 # divisão inteira

print(x,y,z,w,k)

# operações com strings
oi="Oi"
mundo="mundo"
oimundo=oi+" "+mundo+"."
muitosmundos=mundo*5

print(oimundo)
print(muitosmundos)

# operação com listas

impar=[1,3,5]
par=[2,4,6]
inteiros=impar+par

print(inteiros)

print([1,2,3]*3)

# desempacotando iteráveis

l1=[1,2,3]
l2=[2,3,4]
l3=(*l1,*l2)
print(l1,l2,l3)

pessoa=("Fran",33,True)
nome,idade,registrado=pessoa

print(nome)

# podemos definir mais de um tipo de variável na mesma linha:

a,b,c=1,2.,"teste"
print(a,b,c)

# testando e imprimindo variáveis:

if isinstance(a,int):
    print("a = %d" % a) 

if isinstance(b,float):
    print("b = %.2f" % b) # .2 significa 2 dígitos depois da virgula

if c=="teste":
    print("c = %s" % c)

# construindo e imprimindo listas:

lista=[]

lista.append(0.)
lista.append("æßðđŋħł")
lista.append(9)

# imprimindo o primeiro elemento da linha:

print(lista[0])

# imprimindo todos os elementos da lista:

for i in lista:
    print(i)

nomes=["Maria","Ana","Fernanda"]
sobrenomes=["Silva","Barbosa","Souza"]

print("O sobrenome de %s é %s." % (nomes[1],sobrenomes[1]))

# criando lista de objetos vazios:

x=object()
y=object()

listax=[x]*3
listay=[y]*4
listaxy=listax+listay

listaA=[1,2,3,1,1,1]
listaB=[0,1,2,3,4]
listaAB=listaA+listaB

# posição de elementos em uma lista

print("O primeiro elemento de listaB é %d e o terceiro elemento da listaA é %d." % (listaB[0],listaA[2]))

# quantidade de elementos de uma lista

print("A listax tem %d elementos e a listaAB tem %d elementos." % (len(listax),len(listaAB)))

# também podemos usar len() para imprimir comprimento de uma string

nome='Franciele Manoel da Silva'
print("A palavra %s tem %d letras." % (nome,len(nome)))

# quantidade de um tipo de elemento na lista

print("A listaxy tem %d elementos x e a listaAB tem %d vezes o número 1." % (listaxy.count(x),listaAB.count(1)))

# também podemos usar count() para contar letras em uma string

print("O nome %s tem %d letras 'e'." % (nome,nome.count("e")))

# imprimir um intervalo determinado de caracteres:

print(nome[0:4])
print(nome[0:10:2]) #de 2 em 2

# inverter string

print(nome[::-1])

# tudo maiúsculo ou tudo minusculo e separar palavras

print(nome.upper(),nome.lower(),nome.split(" "))

# condições de falso e verdadeiro

x=1

print(x==1,x==2,x<3,x!=1.)

# usando and e or

ocupacao="estudante"
idade=25

if ocupacao=="estudante" and idade==25:
    print("Você é um %s e tem %d anos" % (ocupacao,idade))
    
if ocupacao=="estudante" or idade==25:
    print("Ou você é um %s ou tem %d anos" % (ocupacao,idade))
    
# usando in

aprovados=["João","Ricardo"]
nome="João"

if nome in aprovados:
    print("Você foi aprovado.")
else:
    print("Você foi reprovado.")
    
# usando is

a=[1,2,3]
b=[1,2,3]

if (a==b):
    print("A lista a é igual a lista b.")
elif (a is b): # else if
    print("A lista a 'is' b.")
else:
    print("As listas a e b são diferentes.")

print(a==b,a is b)

# usando not

print(not False, not True == False)

# usando for 

lista=[2,"aaaa",True,[1,2,3]]

for item in lista:
    print(item)

for letra in "Teste":
    print(letra)

for i in range(3):
    print("range 0 a 2:",i)

for i in range(0,10,3):
    print("de 0 a 10 de 3 em 3:",i)

# usando while, break e continue

i=1

while i<=3:
    print("i=",i)
    i=i+1

i=0 

while True:
    i+=1 # o mesmo que i=i+1
    if i>4:
        break
    else:
        print(i,"menor que 5")

for j in range(1,10):
    if j%2==1:
        continue
    print(j,"é par")
else:
    print("acabou o range")

# definindo e utilizando funções 

def frase(): #função sem argumentos
    print("Função!")

frase()

def soma(a,b): # somando dois números
    return a+b

print("soma(1,2)=",soma(1,2))

def saudacao(nome,desejo):
    if nome == "":
        print("Você não digitou um nome.")
    else:
        print("Olá, meu nome é %s e te desejo %s" % (nome,desejo))

nome=input("Digite seu nome. \n") # digitar nome no terminal
saudacao(nome,"Feliz Natal!")

def comidas():
    return "arroz","feijão","bife","batata","salada"

def cardápio(nome):
    return "No cardápio de hoje temos: %s" % nome

def nome_comidas():
    comida=comidas()
    for nome in comida:
        print(cardápio(nome))

nome_comidas()

a=["maçã","banana","laranja"]

def frutas(nome): 
    print("As frutas são: %s" % nome)

for i in range(3):
    frutas(a[i])
    print(cardápio(a[i]))

# usando função com keyword arguments
# keyword arguments => o nome importa, não a ordem

def f(*,a,b):
    print(a,b)

f(b=3,a="a")

# desempacotando listas e dicionários nos argumentos da função

args={"a":1,"b":3}

f(**args)

def g(a,b,c):
    print(a,b,c)

larg=[4,5,"l"]

g(*larg)

# usando decorators "função composta"

def print_arg(funcao):
    def envelope(numero):
        print("O argumento para",funcao.__name__,"é ",numero)
        return funcao(numero)
    
    return envelope

@print_arg
def soma_um(x):
    return x+1

print(soma_um(1))

# função anônima ou lambda

vezes_tres=lambda x: x*3

print(vezes_tres(11))

# usando função lambda junto com map

num=[1,3,5,7]

mais_dois=map(lambda x: x+2,num)

print(list(mais_dois))

# usando list comprehension

w=[x*3.14 for x in range(1,10) if x%2==0]

print(w)

def soma_dois(x):
    return x+2

m=[soma_dois(x) for x in range(1,3)]

print(m)

# usando list comprehension para criar matrizes

mtx=[[j for j in range(3)] for i in range(3)]

print(mtx)

# achatando a matrix

fl=[valor for sublista in mtx for valor in sublista]

print(fl)

# classes e objetos

class mclasse:
    variavel="teste"

    def funcao(self):
        print("Mensagem dentro da classe.")

# definindo o objeto mobjeto a partir da classe mclasse

mobjeto=mclasse()

print(mobjeto.variavel)

mobjeto.variavel="outro"

print(mobjeto.variavel)

mobjeto.funcao()

class roupas:
    tipo=""
    cor=""
    tamanho=""
    preço=10.00

    def descrição(self):
        texto="Este %s %s de tamanho %s vale $%.2f." % (self.tipo,self.cor,self.tamanho,self.preço)
        return texto

objeto=roupas()

print(objeto.descrição())

objeto.tipo="vestido"
objeto.cor="preto"
objeto.tamanho="M"
objeto.preço=89.90

print(objeto.descrição())

#trabalhando com inheritance

class veiculo:
    def __init__(self,ligado=False,velocidade=0):
        self.ligado=ligado
        self.velocidade=velocidade    

    def ligar(self):
        self.ligado=True
        print("Veiculo ligado, vamos dirigir!")

    def aumentar_velocidade(self,delta):
        if self.ligado:
            self.velocidade=self.velocidade+delta
            print("Vroooooom!")
        else:
            print("Você precisa ligar o veiculo primeiro.")
    
    def parar(self):
        self.velocidade=0
        print("Estacionando.")

# usando inheritance para criar a classe carro

class Carro(veiculo):
    mala=False

    def abri_trunk(self):
        self.mala=True
        print("O porta-malas esta aberto.")

    def fech_trunk(self):
        self.mala=False

carro=Carro()
carro.aumentar_velocidade(10)
print(carro.mala)
carro.abri_trunk()
print(carro.mala)
carro.aumentar_velocidade(40)
carro.parar()

# usando inheritance e substituindo a inicialização (__init__)
# usamos super() para poder acessar o __init__ da classe mãe

class Moto(veiculo):
    def __init__(self,ligado=False,velocidade=0,pedal=False):
        self.pedal=pedal
        print("A pedaleira não está para baixo.")
        super().__init__()

moto=Moto()

moto.pedal
moto.ligar()
moto.aumentar_velocidade(30)
print(moto.velocidade)

#substituindo outros métodos da classe mãe

class Bicicleta(veiculo):
    def ligar(self):
        self.ligado=True
        print("Tire o pezinho do chão.")

    def aumentar_velocidade(self,delta):
        if self.ligado:
            self.velocidade=self.velocidade+delta
            print("Pedale mais rápido!")
        else:
            print("Comece a pedalar.")
    
    def parar(self):
        self.velocidade=0
        print("Parando de pedalar.")

bicicleta=Bicicleta()
bicicleta.ligar()
bicicleta.aumentar_velocidade(20)
print(bicicleta.velocidade)
bicicleta.parar()

# criando dois objetos a partir da mesma classe

carro1=Carro()
carro2=Carro()

# cada objeto tera uma identidade diferente

print(id(carro1),id(carro2))

carro1.ligar()
carro1.aumentar_velocidade(40)

print(carro1.velocidade,carro2.velocidade)

# usando docstring

class docstring:
    """ Esta classe não faz nada. """

    def nada(self):
        """ Esta função também não faz nada. """
    pass

print(docstring.__doc__)

# trabalhando com dicionários

num_tel={}
num_tel["Franciele"]=988775544
num_tel["Luis"]=999887766
num_tel["Albertina"]=988112233

print(num_tel)

precos={
    "tomate":3.99,
    "laranja":2.98,
    "maçã":4.59
}

print(precos)

roupas=dict([("camiseta",20.00),
     ("calça",40.00),
     ("casaco",50.00)])

print(roupas)

# usando dict.fromkeys(keys,value)

nomes=("Ana","Pedro","Maria","José")
cel=dict.fromkeys(nomes,None)

print(cel)

# trabalhando com iterador e iteráveis

meu_iterável=range(1,3)
meu_iterador=meu_iterável.__iter__()

print(meu_iterador.__next__())
print(meu_iterador.__next__())

# iterando sobre iteráveis como strings,listas e dicionários

palavra= "palavra"

for letra in palavra:
    print(letra)


for nome,valor in precos.items():
    print("O quilo de %s custa %.2f." % (nome,valor))

# usando list comprehension para iterar sobre dicionário

dados={"nome":"Fran","idade":99,"país":"Brasil"}

print([f"{i}:{j}" for i,j in dados.items()])

# trabalhando com arquivos

file=open("cidades.txt","r") # r = reading

print(file.name) # nome do arquivo

print(file.mode) # modo (leitura, escrita, append)

file.close()

with open("cidades.txt","r") as file: # fecha automaticamente
    file_content=file.read()

print(file_content)

with open("cidades.txt","r") as file: 
    list_content=file.readlines() # cria uma lista de cada linha

print(list_content)

novas_cidades=["São Paulo\n","Rio de Janeiro\n","Belo Horizonte\n"]

with open("cidades2.txt","w") as file: # w = write
    for line in novas_cidades:
        file.write(line)

with open("cidades2.txt","r") as file: 
    list_content2=file.readlines() # cria uma lista de cada linha

print(list_content2)

# lendo um arquivo linha por linha e transformando em iterável

lista=[]

with open('cidades.txt') as cities:
    for line in cities:
        lista.append(line)

print(lista)

# criando uma classe iterável

class Pares:
    anterior=0

    def __iter__(self):
        return self
    
    def __next__(self):
        self.anterior+=2

        if self.anterior>8:
            raise StopIteration

        return self.anterior

pares=Pares()

for i in pares:
    print(i)

# retirando e colocando itens no dicionário

del num_tel["Luis"]

num_tel["Teste"]=999887766

print(num_tel)

precos.pop("maçã")

precos["banana"]=2.49

print(precos)

# visualizando keys do dicionário

items=roupas.keys()

print(items)

a=list(roupas)
b=sorted(roupas)

print(a,b)

# verificando se uma key consta no dicionário

try:
    roupas["meias"]
except KeyError:
    print("Essa key não existe.")

print("meias" not in roupas)

# usando get em dicionários

print(roupas.get("meias"))

# módulos e pacotes

import time # importando o módulo time

ache_itens=[]

for item in dir(time):
    if "name" in item:
        ache_itens.append(item)

print(sorted(ache_itens)) # sorted para ordem alfabética    

