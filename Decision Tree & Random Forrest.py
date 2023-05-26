#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[130]:


data = pd.read_csv('SAHEART.csv')


# In[131]:


data.info()


# In[132]:


data.describe().T


# In[133]:


data.isna().sum()


# In[134]:


data.head()


# In[ ]:





# In[135]:


from sklearn.preprocessing import LabelEncoder


# In[136]:


le = LabelEncoder()


# In[137]:


data['famhist_n'] = le.fit_transform(data['famhist'])
data['chd_n'] = le.fit_transform(data['chd'])


# In[138]:


data


# In[139]:


data = data.drop(['famhist'], axis = 'columns')
data = data.drop(['chd'], axis = 'columns')


# In[140]:


data


# In[ ]:





# In[141]:


x = data[['sbp','tobacco','ldl','adiposity','typea','obesity',
         'alcohol','age','famhist_n']]
y = data['chd_n']


# In[142]:


x


# In[ ]:





# In[143]:


from sklearn.model_selection import train_test_split


# In[144]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[145]:


from sklearn.linear_model import LogisticRegression


# In[146]:


lr_model = LogisticRegression()


# In[147]:


lr_model.fit(X_train,y_train)


# In[148]:


y_predlr = lr_model.predict(X_test)


# In[149]:


y_predlr


# In[150]:


from sklearn.metrics import accuracy_score,mean_squared_error


# In[151]:


mm = mean_squared_error(y_predlr,y_test)


# In[152]:


mm


# In[153]:


ac = accuracy_score(y_predlr,y_test)*100
ac


# In[ ]:





# In[264]:


corr = data.corr()
corr


# In[279]:


dd = data[['age','tobacco','ldl','adiposity','chd_n']]
xd = dd.head(15)


# In[278]:


import seaborn as sns
sns.pairplot(data = dd, hue="chd_n")


# In[280]:


sns.barplot(x='age',y='chd_n',data = xd)


# In[268]:


corr['chd_n'].sort_values(ascending = False)


# In[156]:


data.groupby('chd_n').mean()


# In[ ]:





# In[199]:


x1 = data[['age','tobacco','famhist_n','alcohol']]
y1 = data[['chd_n']]
x1


# In[200]:


from sklearn.model_selection import train_test_split


# In[211]:


X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state=42)


# In[ ]:





# In[202]:


from sklearn.linear_model import LogisticRegression


# In[203]:


lr_model1 = LogisticRegression()


# In[204]:


lr_model1.fit(X_train,y_train)


# In[205]:


y_predlr1 = lr_model1.predict(X_test)


# In[206]:


y_predlr1


# In[207]:


from sklearn.metrics import classification_report, confusion_matrix


# In[208]:


print(classification_report(y_test,y_predlr1))


# In[209]:


print(confusion_matrix(y_test,y_predlr1))


# In[210]:


lr_model1.predict([[52,12,1,97]])


# In[ ]:





# In[213]:


from sklearn import tree


# In[245]:


tree_model = tree.DecisionTreeClassifier(criterion = 'gini', max_depth = 4)


# In[246]:


tree_model.fit(x1,y1)


# In[247]:


tree_model.predict([[52,12,1,97]])


# In[248]:


from sklearn.model_selection import train_test_split


# In[249]:


X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state=42)


# In[250]:


tree_model.fit(X_train,y_train)


# In[251]:


tree_predicted = tree_model.predict(X_test)
tree_predicted


# In[252]:


print(classification_report(y_test,y_predlr1))


# In[253]:


print(classification_report(y_test,tree_predicted))


# In[254]:


import matplotlib.pyplot as plt


# In[256]:


plt.figure(figsize=(15,10))
tree.plot_tree(tree_model,filled=True)
plt.show()


# In[257]:


data


# In[ ]:





# In[258]:


corr['chd_n'].sort_values(ascending = False)


# In[321]:


rfcinput = data[['famhist_n','ldl',
                'adiposity','sbp']]
rfctarget = data['chd_n']


# In[322]:


x_train,x_test,y_train,y_test = train_test_split(rfcinput,rfctarget,test_size=0.5)


# In[323]:


from sklearn.ensemble import RandomForestClassifier


# In[324]:


rfc = RandomForestClassifier(n_estimators = 20,random_state = 42)


# In[325]:


rfc.fit(x_train,y_train)


# In[326]:


rfc_pred = rfc.predict(x_test)
rfc_pred


# In[327]:


rfc.predict([[52,12,1,97]])


# In[328]:


print(classification_report(y_test,rfc_pred))


# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt #seaborn is based on graph 
sns.set(color_codes=True) #Adds nice background to graphs
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


customer = pd.read_csv('customerspends.csv')
customer


# In[5]:


plt.scatter(customer['Apparel'],customer['Beauty and Healthcare'])


# In[6]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_customer = scaler.fit_transform(customer[['Apparel',
                                         'Beauty and Healthcare']])


# In[7]:


from sklearn.cluster import KMeans


# In[8]:


k = 6
clusters = KMeans(k, random_state = 42)
clusters.fit(scaled_customer)
customer["clusterid"] = clusters.labels_


# In[9]:


customer[customer.clusterid==0]


# In[10]:


customer[customer.clusterid==1]


# In[11]:


customer[customer.clusterid==2]


# In[12]:


customer[customer.clusterid==3]


# In[13]:


customer[customer.clusterid==4]


# In[14]:


customer[customer.clusterid==5]


# In[16]:


from scipy.cluster.hierarchy import linkage, dendrogram


# In[17]:


complete_clustering = linkage(scaled_customer, method="complete", metric="euclidean")
average_clustering = linkage(scaled_customer, method="average", metric="euclidean")
single_clustering = linkage(scaled_customer, method="single", metric="euclidean")


# In[18]:


dendrogram(complete_clustering)
plt.show()


# In[19]:


dendrogram(average_clustering)
plt.show()


# In[20]:


dendrogram(single_clustering)
plt.show()


# In[ ]:




