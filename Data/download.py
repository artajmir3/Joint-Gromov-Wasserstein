import urllib.request

file = open("names.txt", "r")
lines = file.readlines()
for line in lines:
	url = "https://files.rcsb.org/download/%s.cif"%(line.strip(),)
	urllib.request.urlretrieve(url, "%s.cif"%(line.strip(),))
	print(url)
