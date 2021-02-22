#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets,svm
svc = svm.SVC(gamma=0.001, C=100.)
digits = datasets.load_digits()


# In[25]:


#desciption
print(digits.DESCR)


# In[31]:


# 2D array representation of image
digits.images[4]


# In[32]:


# Plotting img on graph
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(digits.images[4], cmap=plt.cm.gray_r, interpolation='nearest')


# In[34]:


#img representaion on graph
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.subplot(321)
plt.imshow(digits.images[1791], cmap=plt.cm.gray_r,
interpolation='nearest')
plt.subplot(322)
plt.imshow(digits.images[1792], cmap=plt.cm.gray_r,
interpolation='nearest')
plt.subplot(323)
plt.imshow(digits.images[1793], cmap=plt.cm.gray_r,
interpolation='nearest')
plt.subplot(324)
plt.imshow(digits.images[1794], cmap=plt.cm.gray_r,
interpolation='nearest')
plt.subplot(325)
plt.imshow(digits.images[1795], cmap=plt.cm.gray_r,
interpolation='nearest')
plt.subplot(326)
plt.imshow(digits.images[1796], cmap=plt.cm.gray_r,
interpolation='nearest')


# In[35]:


svc.fit(digits.data[1:1790], digits.target[1:1790])
print(svc.predict(digits.data[1791:]))
print(digits.target[1791:])


# In[39]:


svc.fit(digits.data[400:1000], digits.target[400:1000])
s1=svc.predict(digits.data[1786:1796])
s2=digits.target[1786:1796]
print(s1==s2)


# In[40]:


svc.fit(digits.data[:40], digits.target[:40])
s1=svc.predict(digits.data[1701:1711])
s2=digits.target[1701:1711]
print(s1==s2)


# In[41]:


svc.fit(digits.data[40:1140], digits.target[40:1140])
s1=svc.predict(digits.data[1600:1610])
s2=digits.target[1600:1610]
print(s1==s2)

