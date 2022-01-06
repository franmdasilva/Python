print ('Hello, world!') 

print ('Olá, mundo!')

name = input ('What is your first name? \n') 

print ('Hi, %s.' % name)

idade = input ('Quantos anos você tem? \n')

print ('Você tem %s anos.' % idade)

friends = ['john', 'pat', 'gary', 'michael']
for i, name in enumerate(friends):
    print ("iteration {iteration} is {name}".format(iteration=i, name=name))

comidas = ['arroz', 'feijao', 'cafe', 'leite']
for i, nome in enumerate(comidas):
    print ("Numero {iteracao} e {nome}.".format(iteracao=i, nome=nome))

parents, babies = (1, 1)
while babies < 100:
    print ('This generation has {0} babies'.format(babies))
    parents, babies = (babies, parents + babies)

pais, bebes = (1, 0)
while bebes < 100:
    print ('Esta geracao tem {0} pais e {1} bebes.'.format(pais, bebes))
    pais, bebes = (pais*4, pais*4/2)

def greet(name):
    print ('Hello', name)

greet('Jack')
greet('Jill')
greet('Bob')

def cor(nome):
    print ('As cores são:', nome)

cor('verde')
cor('amarelo')
cor('azul')

import re
for test_string in ['555-1212', 'ILL-EGAL']:
    if re.match(r'^\d{3}-\d{4}$', test_string):
        print (test_string, 'is a valid US local phone number')
    else:
        print (test_string, 'rejected')

import re
for cpf_check in ['009.967.439-44', '1234']:
    if re.match(r'^\d{3}.\d{3}.\d{3}-\d{2}$', cpf_check):
        print (cpf_check, 'Número de CPF válido.')
    else:
        print (cpf_check, 'Número inválido.')

prices = {'apple': 0.40, 'banana': 0.50}
my_purchase = {
    'apple': 1,
    'banana': 6}
grocery_bill = sum(prices[fruit] * my_purchase[fruit]
                   for fruit in my_purchase)
print ('I owe the grocer $%.2f' % grocery_bill)

precos = {'arroz': 4.76, 'feijão': 7.48, 'óleo': 8.41}
compras = {
    'arroz': 2,
    'feijão': 1,
    'óleo': 1}
valor_a_pagar = sum(precos[produtos] * compras[produtos]
                    for produtos in compras)
print ('O valor total a pagar é $%.2f reais.' % valor_a_pagar)

# This program adds up integers that have been passed as arguments in the command line
import sys
try:
    total = sum(int(arg) for arg in sys.argv[1:])
    print ('sum =', total)
except ValueError:
    print ('Please supply integer arguments')

import sys 
try:
    soma = sum(int(arg) for arg in sys.argv[1:])
    print ('Soma =', soma)
except ValueError:
    print ('Digite somente números inteiros.')

# indent your Python code to put into an email
import glob
# glob supports Unix style pathname extensions
python_files = glob.glob('teste.py')
for file_name in sorted(python_files):
    print ('    ------' + file_name)

    with open(file_name) as f:
        for line in f:
            print ('    ' + line.rstrip())

    print()

import glob
arquivosdat = glob.glob('teste.dat')
for dat_nome in sorted(arquivosdat):
    print ('    ~~~~' + dat_nome)

    with open(dat_nome) as f:
        for line in f:
            print('    ' + line.rstrip())

from time import localtime

activities = {8: 'Sleeping',
              9: 'Commuting',
              17: 'Working',
              18: 'Commuting',
              20: 'Eating',
              22: 'Resting' }

time_now = localtime()
hour = time_now.tm_hour

for activity_time in sorted(activities.keys()):
    if hour < activity_time:
        print (activities[activity_time])
        break
else:
    print ('Unknown, AFK or sleeping!')

cronograma = {9: 'Acordar',
              12: 'Trabalhar',
              14: 'Almoçar',
              18: 'Trabalhar',
              20: 'Acabou o trabalho',
              23: 'Janta',
              8: 'Dormir' }

hora_agora = localtime()
hora = hora_agora.tm_hour
minuto = hora_agora.tm_min

print ('Agora são {0}:{1}h.'.format(hora,minuto))

for horario in sorted(cronograma.keys()):
    if hora < horario:
       print (cronograma[horario])
       break
else:
    print ('Dormindo ou outros.')

REFRAIN = ''' 
%d bottles of beer on the wall,
%d bottles of beer,
take one down, pass it around,
%d bottles of beer on the wall!
'''
bottles_of_beer = 9
while bottles_of_beer > 1:
    print (REFRAIN % (bottles_of_beer, bottles_of_beer,
        bottles_of_beer - 1))
    bottles_of_beer -= 1

divisao= '''
Se você dividir %d bananas para duas pessoas, você terá %d para cada um.
'''
bananas = 8
while bananas >= 2:
    print (divisao % (bananas, bananas/2))
    bananas = bananas/2

class BankAccount(object):
    def __init__(self, initial_balance=0):
        self.balance = initial_balance
    def deposit(self, amount):
        self.balance += amount
    def withdraw(self, amount):
        self.balance -= amount
    def overdrawn(self):
        return self.balance < 0
my_account = BankAccount(15)
my_account.withdraw(50)
print (my_account.balance, my_account.overdrawn())

class conta_bancaria(object):
    def __init__(self, saldo_inicial=0):
        self.saldo = saldo_inicial
    def salario(self, valor):
        self.saldo += valor
    def gastos(self, valor):
        self.saldo -= valor
    def divida(self):
        return self.saldo < 0
minha_conta = conta_bancaria(25)
minha_conta.salario(1000)
minha_conta.gastos(965)
print ('O seu saldo atual é de {0} reais. É {1} que você tem uma dívida.'.format(minha_conta.saldo,minha_conta.divida()))

from itertools import groupby
lines = '''
This is the
first paragraph.

This is the second.
'''.splitlines()
# Use itertools.groupby and bool to return groups of
# consecutive lines that either have content or don't.
for has_chars, frags in groupby(lines, bool):
    if has_chars:
        print (' '.join(frags))
# PRINTS:
# This is the first paragraph.
# This is the second.

from itertools import groupby

linhas = '''
Este texto
está
desarrumado.

Vamos
arrumar isso.
'''.splitlines()

for tem_carac, frags in groupby(linhas, bool):
    if tem_carac:
        print (' '.join(frags))

import csv

# need to define cmp function in Python 3 
def cmp(a, b):
    return (a > b) - (a < b) #Se a>b cmp(a,b)=1, e se a<b cmp(a,b)=-1.

# write stocks data as comma-separated values
with open('stocks.csv', 'w', newline='') as stocksFileW:
    writer = csv.writer(stocksFileW)
    writer.writerows([
        ['GOOG', 'Google, Inc.', 505.24, 0.47, 0.09],
        ['YHOO', 'Yahoo! Inc.', 27.38, 0.33, 1.22],
        ['CNET', 'CNET Networks, Inc.', 8.62, -0.13, -1.4901]
    ])

# read stocks data, print status messages
with open('stocks.csv', 'r') as stocksFile:
    stocks = csv.reader(stocksFile)

    status_labels = {-1: 'down', 0: 'unchanged', 1: 'up'}
    for ticker, name, price, change, pct in stocks:
        status = status_labels[cmp(float(change), 0.0)] 
        print ('%s is %s (%.2f)' % (name, status, float(pct)))

# Criando o arquivo acoes.csv:
with open('acoes.csv','w', newline='') as acoes_arqW:
    writer = csv.writer(acoes_arqW)
    writer.writerows([
        ['PETR4.SA','PETROBRAS PN',24.02,-1.56],
        ['ELET3.SA','ELETROBRAS ON',34.50,2.07],
        ['UGPA3.SA','ULTRAPAR ON',19.82,-6.86]
])

# Lendo o arquivo acoes.csv:
with open('acoes.csv','r') as acoes_arq:
    acoes = csv.reader(acoes_arq)

    situacao_nome = {-1: 'caindo', 0: 'estável', 1: 'subindo'}
    for sigla, nome, preço, variação in acoes:
        situacao = situacao_nome[cmp(float(variação),0.0)]
        print ('%s está %s %.2f porcento, com preço atual de %.2f .' % (nome, situacao, float(variação),float(preço)))

import itertools

def iter_primes():
     # an iterator of all numbers between 2 and +infinity
     numbers = itertools.count(2)

     # generate primes forever
     while True:
         # get the first number from the iterator (always a prime)
         prime = next(numbers)
         yield prime

         # this code iteratively builds up a chain of
         # filters...slightly tricky, but ponder it a bit
         numbers = filter(prime.__rmod__, numbers)

for p in iter_primes():
    if p > 20:
        break
    print (p)

dinner_recipe = '''<html><body><table>
<tr><th>amt</th><th>unit</th><th>item</th></tr>
<tr><td>24</td><td>slices</td><td>baguette</td></tr>
<tr><td>2+</td><td>tbsp</td><td>olive oil</td></tr>
<tr><td>1</td><td>cup</td><td>tomatoes</td></tr>
<tr><td>1</td><td>jar</td><td>pesto</td></tr>
</table></body></html>'''

# From http://effbot.org/zone/element-index.htm
import xml.etree.ElementTree as etree
tree = etree.fromstring(dinner_recipe)

# For invalid HTML use http://effbot.org/zone/element-soup.htm
# import ElementSoup, StringIO
# tree = ElementSoup.parse(StringIO.StringIO(dinner_recipe))

pantry = set(['olive oil', 'pesto'])
for ingredient in tree.iter('tr'):
    amt, unit, item = ingredient
    if item.tag == "td" and item.text not in pantry:
        print ("%s: %s %s" % (item.text, amt.text, unit.text))