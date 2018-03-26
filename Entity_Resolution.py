import json
import csv
import numpy as np
import pandas as pd
import editdistance as ed
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import f1_score,precision_score, recall_score, accuracy_score
import re
import nltk
nltk.download("stopwords")
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter

"""
This assignment can be done in groups of 3 students. Everyone must submit individually.

Write down the UNIs of your group (if applicable)

Name : Midhun Gundapuneni
Uni  : mg3845

Member 2: Chin-Wen Chang, cc3677

Member 3: Sithal Nimmagadda, sn2738
"""

def get_matches(locu_train_path, foursquare_train_path, matches_train_path, locu_test_path, foursquare_test_path):

	stemmer = PorterStemmer()
	stop_words = set(stopwords.words('english'))

	def clean_data(locu_path,foursquare_path):
	
		#load the data
		foursquare_data = pd.read_json(open(foursquare_path))
		locu_data = pd.read_json(open(locu_path))

		#drop the columns that have most of the columns same
		foursquare = foursquare_data.drop(['country', 'region', 'locality'], axis=1)
		locu = locu_data.drop(['country', 'region', 'locality'], axis=1)


		# Cleaning phone number to same format
		def cleanPhone(x):
			if x is None or x is ',':
				return ''
			else:
				return x.replace('(', '').replace(')', '').replace('-', '').replace(' ', '')

		# Cleaning website
		def cleanWebsite(x):
			if x is None or x is ',':
				return ''
			else:
				return x.replace('https://', '').replace('http://', '').replace('www.', '')

		# Clean 
		def cleanName(x):
			if x is None or x is ',':
				return ''
			else:
				x = x.replace('\'', '').replace('#', '').replace('&', '').replace('-', ' ').replace('/', ' ').replace(":","").replace("@","").replace("(","").replace(")","").replace(".","")
				x = x.lower()
				x = " "+x
				x = x.replace(" zero","0").replace(" one","1").replace(" two","2").replace(" three","3").replace(" four","4").replace(" five","5").replace(" six","6").replace(" seven","7").replace(" eight","8").replace(" nine","9")
				x = x.strip()
				return x

		def cleanAddress(x):
			if x is None or x is ',':
				return ''
			else:
				x = x.replace('\'', '').replace('#', '').replace('&', '').replace('.', '').replace('@', '').replace(',', '').replace('-', ' ').replace('/', ' ').replace(":","")
				x = x.lower()
				x = x.replace('street', 'st').replace('avenue', 'ave').replace('boulevard', 'blvd').replace('place', 'pl').replace('square', 'sq').replace('plaza', 'plz').replace("new york","")
				x = " "+x
				x = x.replace(" zero","0").replace(" one","1").replace(" two","2").replace(" three","3").replace(" four","4").replace(" five","5").replace(" six","6").replace(" seven","7").replace(" eight","8").replace(" nine","9")
				x = x.strip()
				return x

		foursquare['phone'] = foursquare['phone'].map(cleanPhone)
		foursquare['website'] = foursquare['website'].map(cleanWebsite)
		foursquare['name'] = foursquare['name'].map(cleanName)
		foursquare['street_address'] = foursquare['street_address'].map(cleanAddress)

		locu['phone'] = locu['phone'].map(cleanPhone)
		locu['website'] = locu['website'].map(cleanWebsite)
		locu['name'] = locu['name'].map(cleanName)
		locu['street_address'] = locu['street_address'].map(cleanAddress)

		f=open("fs_clean.json","w")
		f.write(foursquare.to_json(orient='records'))
		f=open("lc_clean.json","w")
		f.write(locu.to_json(orient='records'))

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

	clean_data(locu_train_path, foursquare_train_path)

	matches = open(matches_train_path).readlines()[1:]
	locu_match = {}
	for l in matches:
		l = l.strip().split(',')
		locu_match[l[0]] = l[1]

	foursquare_data = json.load(open('fs_clean.json'))
	locu_data = json.load(open('lc_clean.json'))
	foursquare_data = np.array(foursquare_data)
	locu_data = np.array(locu_data)

	def load_clean_data_get_candidates(threshold):

		words = {}
		for l in foursquare_data:
			l = l["name"]
			for k in [stemmer.stem(i) for i in l.strip().split(" ")]:
				if k not in words:
					words[k]=0
				words[k]+=1

		temp = list(words.keys())
		for l in temp:
			if words[l]>threshold:
				del words[l]

		fs_id_ind = {}
		lo_id_ind = {}
		for l in locu_data:
			lo_id_ind[l['id']] = l
		fs_phone_ind = {}
		fs_name_ind = {}
		fs_name_part_ind = {}
		for l in foursquare_data:
			fs_id_ind[l['id']] = l
			if l['phone'] is not None and l['phone']!='':
				fs_phone_ind[l['phone']] = l
			if l['name'].replace(" ","") not in fs_name_ind:
				fs_name_ind[l['name'].replace(" ","")] = set()
			fs_name_ind[l['name'].replace(" ","")].add(l['id'])
			splits = l['name'].split(" ")
			for k in [stemmer.stem(i) for i in splits]:
				if k!=" " and k!="" and k in words:
					if k not in fs_name_part_ind:
						fs_name_part_ind[k] = set()
					fs_name_part_ind[k].add(l["id"])

		foursquare_sortedx = sorted(foursquare_data,key=lambda x:x['latitude'])
		foursquare_sortedy = sorted(foursquare_data,key=lambda x:x['longitude'])

		param_lat = 0.002
		param_lon = 0.002
		ann = {}
		for index,l in enumerate(locu_data):
			if l['latitude'] is not None and l['longitude'] is not None:

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

				splits = l['name'].split(" ")
				for k in [stemmer.stem(i) for i in splits]:
					if k in fs_name_part_ind:
						setC.update(fs_name_part_ind[k])
				ann[index] = setC
		return ann,fs_id_ind

	ann,fs_id_ind = load_clean_data_get_candidates(4)

	def calc_cosine(a,b):
		a = Counter(a)
		b = Counter(b)
		dot = 0
		a2 = 0
		b2 = 0
		for l in a:
			if l in b:
				dot = a[l]*b[l]
		for l in a:
			a2+=a[l]**2
		for l in b:
			b2+=b[l]**2
		return dot/((a2*b2)**0.5)


	def create_features(l1,l2):
		feat = []
		f_code = 0
		if l1['postal_code']=='' or l2['postal_code']=='':
			f_code=0
		elif l1['postal_code']==l2['postal_code']:
			f_code = 1
		feat.append(f_code)

		f_phone = 0
		f_phone_last = 0
		if l1['phone']=='' or l2['phone']=='':
			f_phone=0
			f_phone_last = 0
		else:
			if l1['phone']==l2['phone']:
				f_phone=1
			if l1['phone'][-4:] == l2['phone'][-4:]:
				f_phone_last = 1
		feat.append(f_phone)
		feat.append(f_phone_last)
		
		f_name = 0
		f_name_jaccard = 0
		f_name_exact = 0
		f_name_cosine = 0
		f_name_char = 0
		f_name_match = 0
		f_name_jaccard2 = 0
		f_name_sum = 0
		if l1['name']=='' or l2['name']=='':
			f_name = 0
			f_name_jaccard = 0
			f_name_exact = 0
			f_name_jaccard2 = 0
			f_name_char = 0
			f_name_cosine = 0
			f_name_match = 0
		else:
			if l1["name"]==l2["name"]:
				f_name_exact = 1
			f_name = ed.eval(l1['name'],l2['name'])/max(len(l1['name']),len(l2['name']))
			setA = set(l1['name'].split(" "))
			setB = set(l2['name'].split(" "))
			f_name_jaccard = len(setA.intersection(setB))/len(setA.union(setB))
			stem_l1 = [stemmer.stem(i) for i in l1["name"].split(" ") if i not in stop_words]
			stem_l2 = [stemmer.stem(i) for i in l2["name"].split(" ") if i not in stop_words]
			f_name_match = len(set(stem_l1).intersection(set(stem_l2)))/min(len(stem_l1),len(stem_l2))
			f_name_sum = sum([len(i) for i in set(stem_l1).intersection(set(stem_l2))])/min(len(l1['name']),len(l2['name']))
			f_name_jaccard2 = len(set(stem_l1).intersection(set(stem_l2)))/len(set(stem_l1).union(set(stem_l2)))
			f_name_cosine = calc_cosine(stem_l1,stem_l2)
			t1 = [i for i in l1["name"]]
			t2 = [i for i in l2["name"]]
			f_name_char = calc_cosine(t1,t2)
		feat.append(f_name)
		feat.append(f_name_exact)
		feat.append(f_name_cosine)
		feat.append(f_name_match)

		f_add = 0
		f_add_num = 0
		f_add_exact = 0
		if l1['street_address']=='' or l2['street_address']=='':
			f_add = 1
			f_add_num = 0
			f_add_exact = 0
		else:
			if l1['street_address']==l2['street_address']:
				f_add_exact = 1
			f_add = ed.eval(l1['street_address'],l2['street_address'])/max(len(l1['street_address']),len(l2['street_address']))
			l1_num = set(re.findall(r'\d+', l1['street_address']))
			l2_num = set(re.findall(r'\d+', l2['street_address']))
			if len(l1_num.union(l2_num))==0:
				f_add_num = 0
			else:
				f_add_num = len(l1_num.intersection(l2_num))/len(l1_num.union(l2_num))
		feat.append(f_add)
		feat.append(f_add_num)

		f_web = 0
		if l1['website']=='' or l2['website']=='':
			f_web = 1
		else:
			f_web = ed.eval(l1['website'],l2['website'])/max(len(l1['website']),len(l2['website']))
		feat.append(f_web)

		f_lat = 1
		f_lon = 1
		if l1['latitude'] is not None and l2['latitude'] is not None:
			f_lat = l1['latitude']-l2['latitude']
			f_lon = l1['longitude']-l2['longitude']
		feat.append(f_lat)
		feat.append(f_lon)
		
		c = 0
		if l2['postal_code']=="":
			c+=1
		elif l1['postal_code']==l2['postal_code']:
			c+=1
		if l2["phone"]=="":
			c+=1
		elif l1['phone']==l2['phone']:
			c+=1
		if l2["street_address"]=="":
			c+=1
		elif l1['street_address']==l2['street_address']:
			c+=1
		if l2["website"]=="":
			c+=1
		elif l1['website']==l2['website']:
			c+=1
		
		f_name_high = 0
		f_name_jaccard_high = 0
		f_name_jaccard2_high = 0
		f_name_exact_high = 0
		f_name_char_high = 0
		f_name_cosine_high = 0
		f_name_match_high = 0 
		f_name_sum_high = 0
		if c==4 and "starbucks" not in l1["name"] and "starbucks" not in l2["name"]:
			f_name_high = f_name
			f_name_jaccard_high = f_name_jaccard
			f_name_jaccard2_high = f_name_jaccard2
			f_name_exact_high = f_name_exact
			f_name_char_high = f_name_char
			f_name_cosine_high = f_name_cosine
			f_name_match_high = f_name_match
			f_name_sum_high = f_name_sum
		feat.append(f_name_high)
		feat.append(f_name_jaccard2_high)
		feat.append(f_name_match_high)
		feat.append(f_name_sum)

		return feat

	x_train = []
	y_train = []
	x_test = []
	y_test = []
	X = []
	y = []
	pairs = []
	c = 0
	train_sample = 0
	for l in ann:
		key = 0
		if locu_data[l]["id"] in ["493f5e2798de851ec3b2","5f3fd107090d0ddc658b","212dffb393f745df801a",
								   "edeba23f215dcc702220","c170270283ef870d546b"]:
			key = 1
		c+=1
		for fid in ann[l]:
			l1 = locu_data[l]
			l2 = fs_id_ind[fid]
			if l1['id'] in locu_match and locu_match[l1['id']]==fid:
				label=1
			else:
				label=0
			
			pairs.append((l1,l2))
			
			feat = create_features(l1,l2)
			
			if c<=450:
				train_sample+=1
			X.append(feat)
			y.append(label)
			if key==1 and c<=450:
				for r in range(10):
					if c<=450:
						train_sample+=1
					pairs.append((l1,l2))
					X.append(feat)
					y.append(label)

	X = np.array(X)
	y = np.array(y)

	x_train = X[:train_sample]
	x_test = X[train_sample:]
	y_train = y[:train_sample]
	y_test = y[train_sample:]

	rs = 4121
	l_max = 0
	mval = 0
	for l in range(10,151,5):
		rfc = RFC(n_estimators=l,random_state=rs).fit(x_train,y_train)
		y_pred = rfc.predict(x_test)

		if accuracy_score(y_pred,y_test) >= mval:
			mval= accuracy_score(y_pred,y_test)
			l_max = l

	rfc = RFC(n_estimators=l_max,random_state=rs).fit(x_train,y_train)
	y_pred = rfc.predict(x_test)
	rfc = RFC(n_estimators=l_max,random_state=rs).fit(X,y)

	clean_data(locu_test_path,foursquare_test_path)

	foursquare_data = json.load(open('fs_clean.json'))
	locu_data = json.load(open('lc_clean.json'))
	foursquare_data = np.array(foursquare_data)
	locu_data = np.array(locu_data)

	ann,fs_id_ind = load_clean_data_get_candidates(3)

	X = []
	c = 0
	pairs = []
	for l in ann:
		c+=1
		for fid in ann[l]:
			l1 = locu_data[l]
			l2 = fs_id_ind[fid]

			pairs.append((l1['id'],l2['id']))
			
			feat = create_features(l1,l2)

			X.append(feat)

	y_pred = rfc.predict(X)
	y_pred_prob = rfc.predict_proba(X)
	y_pred_2 = np.array(y_pred)

	y_pred = []
	for l in y_pred_prob:
		if l[1]>.2:
			y_pred.append(1)
		else:
			y_pred.append(0)

	unique_locu = {}
	for l in range(len(y_pred)):
		if y_pred[l]==1:
			ind = l
			l = pairs[l]
			if l[0] not in unique_locu:
				unique_locu[l[0]] = (l[1],y_pred_prob[ind][1])
			elif unique_locu[l[0]][1]<y_pred_prob[ind][1]:
				unique_locu[l[0]] = (l[1],y_pred_prob[ind][1])


	unique_forsquare = {}
	for l in unique_locu:
		if unique_locu[l][0] not in unique_forsquare:
			unique_forsquare[unique_locu[l][0]] = (l,unique_locu[l][1])
		elif unique_forsquare[unique_locu[l][0]][1] < unique_locu[l][1]:
			unique_forsquare[unique_locu[l][0]] = (l,unique_locu[l][1])

	f = open('matches_test.csv','w')
	f.write("locu_id,foursquare_id\n")
	for l in unique_forsquare:
		f.write(str(unique_forsquare[l][0])+','+str(l)+'\n')
	f.close()