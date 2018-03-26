
# coding: utf-8

# In[365]:


import json
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import editdistance as ed


# In[366]:


def clean_data(path,fs,lc):
    
    #load the data
    foursquare_data = pd.read_json(open(path+"/"+fs+'.json'))
    locu_data = pd.read_json(open(path+"/"+lc+'.json'))

    #drop the columns that have most of the columns same
    foursquare = foursquare_data.drop(['country', 'region', 'locality'], axis=1)
    locu = locu_data.drop(['country', 'region', 'locality'], axis=1)

    def cleanPhone(x):
        if x is None or x is ',':
            return ''
        else:
            return x.replace('(', '').replace(')', '').replace('-', '').replace(' ', '')

    def cleanWebsite(x):
        if x is None or x is ',':
            return ''
        else:
            return x.replace('https://', '').replace('http://', '').replace('www.', '')

    def cleanName(x):
        if x is None or x is ',':
            return ''
        else:
            x = x.replace('\'', '').replace('#', '').replace('&', '').replace('-', ' ').replace('/', ' ')
            x = x.lower()
            return x

    def cleanAddress(x):
        if x is None or x is ',':
            return ''
        else:
            x = x.replace('\'', '').replace('#', '').replace('&', '').replace('.', '').replace('@', '').            replace(',', '').replace('-', '').replace('/', '').replace(' ', '')
            x = x.lower()
            x = x.replace('street', 'st').replace('avenue', 'ave').replace('boulevard', 'blvd').            replace('place', 'pl').replace('square', 'sq').replace('plaza', 'plz')
            return x

    foursquare['phone'] = foursquare['phone'].map(cleanPhone)
    foursquare['website'] = foursquare['website'].map(cleanWebsite)
    foursquare['name'] = foursquare['name'].map(cleanName)
    foursquare['street_address'] = foursquare['street_address'].map(cleanAddress)

    locu['phone'] = locu['phone'].map(cleanPhone)
    locu['website'] = locu['website'].map(cleanWebsite)
    locu['name'] = locu['name'].map(cleanName)
    locu['street_address'] = locu['street_address'].map(cleanAddress)

    f=open(path+"/fs_clean.json","w")
    f.write(foursquare.to_json(orient='records'))
    f=open(path+"/lc_clean.json","w")
    f.write(locu.to_json(orient='records'))


# In[367]:


clean_data("ibdownload_matches_train","foursquare_train","locu_train")


# In[368]:


matches = open('ibdownload_matches_train/matches_train.csv').readlines()[1:]
locu_match = {}
for l in matches:
    l = l.strip().split(',')
    locu_match[l[0]] = l[1]


# In[369]:


foursquare_data = json.load(open('ibdownload_matches_train/fs_clean.json'))
locu_data = json.load(open('ibdownload_matches_train/lc_clean.json'))
foursquare_data = np.array(foursquare_data)
locu_data = np.array(locu_data)

fs_id_ind = {}
lo_id_ind = {}
for l in locu_data:
    lo_id_ind[l['id']] = l
fs_phone_ind = {}
fs_name_ind = {}
for l in foursquare_data:
    fs_id_ind[l['id']] = l
    if l['phone'] is not None and l['phone']!='':
        fs_phone_ind[l['phone']] = l
    if l['name'].replace(" ","") not in fs_name_ind:
        fs_name_ind[l['name'].replace(" ","")] = set()
    fs_name_ind[l['name'].replace(" ","")].add(l['id'])

foursquare_sortedx = sorted(foursquare_data,key=lambda x:x['latitude'])
foursquare_sortedy = sorted(foursquare_data,key=lambda x:x['longitude'])

def binary_search(array, target, key):
    lower = 0
    upper = len(array)
    while lower < upper:
        x = lower + int((upper - lower)/2)
        val = array[x][key]
        if target == val:
            return x
        elif target > val:
            if lower == x:
                return lower
            lower = x
        elif target < val:
            upper = x
    return lower

c = 0
tot = 0
ann = {}
for index,l in enumerate(locu_data):
    if l['latitude'] is not None and l['longitude'] is not None:
        param_lat = 0.001
        param_lon = 0.001
        lat_l = binary_search(foursquare_sortedx,l['latitude']-param_lat,'latitude')
        lat_r = binary_search(foursquare_sortedx,l['latitude']+param_lat,'latitude')
        lon_l = binary_search(foursquare_sortedy,l['longitude']-param_lon,'longitude')
        lon_r = binary_search(foursquare_sortedy,l['longitude']+param_lon,'longitude')
        setA = set()
        for i in range(lat_l,min(lat_r+2,len(foursquare_data))):
            setA.add(foursquare_sortedx[i]['id'])
        setB = set()
        for i in range(lon_l,min(lon_r+2,len(foursquare_data))):
            setB.add(foursquare_sortedy[i]['id'])
        setC = setA.intersection(setB)
        if l['name'].replace(" ","") in fs_name_ind:
            setC.update(fs_name_ind[l['name'].replace(" ","")])
        if l['phone'] in fs_phone_ind:
            setC.add(fs_phone_ind[l['phone']]['id'])
        if l['id'] in locu_match:
            if locu_match[l['id']] in setC:
                c+=1
            else:
                print(lo_id_ind[l['id']],fs_id_ind[locu_match[l['id']]])
                print(lat_l,lat_r,lon_l,lon_r)
                print(lo_id_ind[l['id']]['latitude']-fs_id_ind[locu_match[l['id']]]['latitude'])
                print(lo_id_ind[l['id']]['longitude']-fs_id_ind[locu_match[l['id']]]['longitude'])
                print("---------------------------------------------")
        ann[index] = setC
        tot+=len(setC)
print(c,tot)


# In[370]:


x_train = []
y_train = []
x_test = []
y_test = []
X = []
y = []
c = 0
for l in ann:
    c+=1
    for fid in ann[l]:
        l1 = locu_data[l]
        l2 = fs_id_ind[fid]
        if l1['id'] in locu_match and locu_match[l1['id']]==fid:
            label=1
        else:
            label=0
        
        f_code = 0
        if l1['postal_code']=='' or l2['postal_code']=='' or l1['postal_code']==l2['postal_code']:
            f_code=1
        
        f_phone = 0
        if l1['phone']=='' or l2['phone']=='' or l1['phone']==l2['phone']:
            f_phone=1
            
        f_name = 0
        if l1['name']=='' or l2['name']=='':
            f_name = 0
        else:
            f_name = ed.eval(l1['name'],l2['name'])
        
        f_add = 0
        if l1['street_address']=='' or l2['street_address']=='':
            f_add = 0
        else:
            f_add = ed.eval(l1['street_address'],l2['street_address'])
        
        f_web = 0
        if l1['website']=='' or l2['website']=='':
            f_web = 0
        else:
            f_web = ed.eval(l1['website'],l2['website'])
        
        
        if c<=450:
            x_train.append([f_code,f_phone,f_name,f_add])
            y_train.append(label)
        else:
            x_test.append([f_code,f_phone,f_name,f_add])
            y_test.append(label)
        X.append([f_code,f_phone,f_name,f_add])
        y.append(label)
sum(y_test)


# In[371]:


from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import f1_score,precision_score, recall_score


# In[372]:


len(y_test)


# In[373]:


x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# In[374]:


y_pred = RFC().fit(x_train,y_train).predict(x_test)


# In[375]:


f1_score(y_pred,y_test),precision_score(y_pred,y_test),recall_score(y_pred,y_test)


# In[376]:


rfc = RFC().fit(X,y)
y_pred = rfc.predict(X)


# In[377]:


f1_score(y_pred,y),precision_score(y_pred,y),recall_score(y_pred,y)


# In[386]:


clean_data("ibdownload_foursquare_test","foursquare_test","locu_test")


# In[387]:


foursquare_data = json.load(open('ibdownload_foursquare_test/fs_clean.json'))
locu_data = json.load(open('ibdownload_foursquare_test/lc_clean.json'))
foursquare_data = np.array(foursquare_data)
locu_data = np.array(locu_data)

fs_id_ind = {}
lo_id_ind = {}
for l in locu_data:
    lo_id_ind[l['id']] = l
fs_phone_ind = {}
fs_name_ind = {}
for l in foursquare_data:
    fs_id_ind[l['id']] = l
    if l['phone'] is not None and l['phone']!='':
        fs_phone_ind[l['phone']] = l
    if l['name'].replace(" ","") not in fs_name_ind:
        fs_name_ind[l['name'].replace(" ","")] = set()
    fs_name_ind[l['name'].replace(" ","")].add(l['id'])

foursquare_sortedx = sorted(foursquare_data,key=lambda x:x['latitude'])
foursquare_sortedy = sorted(foursquare_data,key=lambda x:x['longitude'])

def binary_search(array, target, key):
    lower = 0
    upper = len(array)
    while lower < upper:
        x = lower + int((upper - lower)/2)
        val = array[x][key]
        if target == val:
            return x
        elif target > val:
            if lower == x:
                return lower
            lower = x
        elif target < val:
            upper = x
    return lower

c = 0
tot = 0
ann = {}
for index,l in enumerate(locu_data):
    if l['latitude'] is not None and l['longitude'] is not None:
        param_lat = 0.001
        param_lon = 0.001
        lat_l = binary_search(foursquare_sortedx,l['latitude']-param_lat,'latitude')
        lat_r = binary_search(foursquare_sortedx,l['latitude']+param_lat,'latitude')
        lon_l = binary_search(foursquare_sortedy,l['longitude']-param_lon,'longitude')
        lon_r = binary_search(foursquare_sortedy,l['longitude']+param_lon,'longitude')
        setA = set()
        for i in range(lat_l,min(lat_r+2,len(foursquare_data))):
            setA.add(foursquare_sortedx[i]['id'])
        setB = set()
        for i in range(lon_l,min(lon_r+2,len(foursquare_data))):
            setB.add(foursquare_sortedy[i]['id'])
        setC = setA.intersection(setB)
        if l['name'].replace(" ","") in fs_name_ind:
            setC.update(fs_name_ind[l['name'].replace(" ","")])
        if l['phone'] in fs_phone_ind:
            setC.add(fs_phone_ind[l['phone']]['id'])
        ann[index] = setC
        tot+=len(setC)
print(c,tot)


# In[392]:


X = []
c = 0
pairs = []
for l in ann:
    c+=1
    for fid in ann[l]:
        l1 = locu_data[l]
        l2 = fs_id_ind[fid]

        pairs.append((l1['id'],l2['id']))
        
        f_code = 0
        if l1['postal_code']=='' or l2['postal_code']=='' or l1['postal_code']==l2['postal_code']:
            f_code=1
        
        f_phone = 0
        if l1['phone']=='' or l2['phone']=='' or l1['phone']==l2['phone']:
            f_phone=1
            
        f_name = 0
        if l1['name']=='' or l2['name']=='':
            f_name = 0
        else:
            f_name = ed.eval(l1['name'],l2['name'])
        
        f_add = 0
        if l1['street_address']=='' or l2['street_address']=='':
            f_add = 0
        else:
            f_add = ed.eval(l1['street_address'],l2['street_address'])
        
        f_web = 0
        if l1['website']=='' or l2['website']=='':
            f_web = 0
        else:
            f_web = ed.eval(l1['website'],l2['website'])

        X.append([f_code,f_phone,f_name,f_add])
len(X)


# In[393]:


y_pred = rfc.predict(X)


# In[394]:


sum(y_pred)


# In[395]:


len(locu_data)


# In[397]:


f = open('matches_test.csv','w')
f.write("locu_id,foursquare_id\n")
for l in range(len(y_pred)):
    if y_pred[l]==1:
        l = pairs[l]
        f.write(str(l[0])+','+str(l[1])+'\n')
f.close()

