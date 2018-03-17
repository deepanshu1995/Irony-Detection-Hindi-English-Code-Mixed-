import csv
count = 0
id_to_tweet_map = {}
tweet_to_id_map = {}
id_to_class_map = {}
tweet_to_class_map = {}
ids = []
texts = []
with open('dataset.tsv') as dataset:
	for line in csv.reader(dataset, delimiter="\t"):
		
		class_name = []
		text = line[0]
		tweet_id = line[1]
		label = line[2]
		id_to_tweet_map[tweet_id] = text
		tweet_to_id_map[text] = tweet_id
		id_to_class_map[tweet_id] = label
		tweet_to_class_map[text] = label
		



#print id_to_tweet_map['876766143972691968']
#print tweet_to_id_map['Sir ji #Sarcasm laga diya kijiye..Warna kuch logo ki satak jati hai!']
#print id_to_class_map['876766143972691968']
#print tweet_to_class_map['Sir ji #Sarcasm laga diya kijiye..Warna kuch logo ki satak jati hai!']
#print id_to_class_map
