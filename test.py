import Entity_Resolution as ER

for l in range(100):
	ER.get_matches("train/locu_train.json","train/foursquare_train.json","train/matches_train.csv","test/locu_test.json","test/foursquare_test.json")
	f = open("matches_test.csv").readlines()
	print(len(f))