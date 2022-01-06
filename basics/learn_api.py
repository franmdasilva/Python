import requests #Requests allows you to send HTTP/1.1 requests extremely easily.

import os #This module provides a portable way of using operating system dependent functionality.
from PIL import Image #PIL is the Python Imaging Library which provides the python interpreter with image editing capabilities
from IPython.display import IFrame # funciona apenas no Jupyter

url='http://ufsc.br/'
r=requests.get(url)

print(r.status_code)

print("\n",r.request.headers)

print("\n","request body:", r.request.body)

header=r.headers

print("\n",r.headers)

print("\n",header['date'])

print("\n",header['Content-Type'])

print("\n", r.encoding)

print("\n",r.text[0:100])

# Use single quotation marks for defining string
url='https://preview.redd.it/gg4u32nlrsq61.jpg?width=960&height=871&crop=smart&auto=webp&s=6085ef263766865512ea5077d89a786cb26e59ec'

r=requests.get(url)

print("\n",r.headers)

print("\n",r.headers['Content-Type'])

path=os.path.join(os.getcwd(),'image.jpg')

print("\n",path)

with open(path,'wb') as f:
    f.write(r.content)

print(Image.open(path))