#!/usr/bin/env python
# coding: utf-8

# Dando continuidade ao pré-processamento dos dados, vamos agora verificar os dados faltantes,
# construir gráficos, codificação e singularidades. Não podemos esquecer do objetivo, pois suas escolhas devem ser sempre no sentido de uma boa resposta final.Cada escolha vai refletir na acurácia do seu modelo, de forma positiva ou não.

# In[1]:


#Importando bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#carregando o dataset Titanic
data = pd.read_csv('train.csv')


# In[3]:


#Verificando as 10 primeiras linhas do dataframe
data.head(10)


# A partir do dataset Titanic, vamos construir um modelo que consiga prever de forma satisfatória,
# aqueles individuos que sobreviveram ou não ao acidente do navio.
# Podemos verificar que há algumas informações que não serão relevantes para a resolução
# do nosso problema.
# Por exemplo, a variável Name  não parece ter importância para a previsão de sobrevivência ou
# não do indivíduo. Assim como o número do ticket e o local de embarque. Assim, iremos retirar esses dados do dataframe. Vamos retirar também o ID que não será necessário.
# 

# In[4]:


data.drop(columns = ['Name','Ticket','Embarked','PassengerId'], axis = 1, inplace= True)


# In[5]:


#Verificando os valores missing
data.isnull().sum()


# Podemos verificar que a variável Age possui 177 valores faltantes (19,9%) e Cabin 687 (77,1%)
# Devido ao alto número de missing em Cabin, vamos retirá-la. Essa é uma decisão do cientista
# de dados, a escolha tem que ser feita levando em conta o conhecimento, estudo do caso e 
# experiência do mesmo. As vezes é mais aconselhável retirar o dado, em outras convém substituí-lo pela média, entre outras alternativas.

# In[6]:


data.drop('Cabin', axis = 1, inplace = True)


# Já no caso da variável Age, vamos construir um histograma para verificar qual o comportamento
# da variável. Para construir vamos utilizar a função hist do matplotlib. Verificamos que 
# as idades (Age) concentram-se em sua maioria entre 20 e 40 anos. Por isso, vamos substituir
# os dados missing pela média e verificar se houve uma mudança significativa nos dados.

# In[7]:


data.Age.hist()


# In[8]:


#Verificando a média das idades
data.Age.mean()


# In[9]:


# Preenche os dados missing de Age com a media
data.Age.fillna(data.Age.mean(), inplace = True)
data.Age.isnull().sum()


# In[10]:


#Verificando a nova média
#Percebemos que a média pouco se modificou!
data.Age.mean()


# In[11]:


#Verificando se os dados missing foram realmente zerados
data.isnull().sum()


# Podemos observar que temos a variável Sex separada entre male e female. Vamos usar uma 
# ferramenta chamada Label Encoding para codificar essa variável, já que devemos ter variaveis
# numéricas para a construção do modelo.Primeiro, modificamos a variável para categoria e, então aplicamos o encoding.

# In[14]:


data['Sex'] = data['Sex'].astype('category')
data['Sex'] = data['Sex'].cat.codes


# In[19]:


data.head()


# In[ ]:




