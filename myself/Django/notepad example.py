import urllib.request
print(urllib.request.urlopen("http://www.example.com").read().decode('utf-8'))