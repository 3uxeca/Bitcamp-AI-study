


interval = 6
x = []
y = []
temps = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,18,20,21,22,23,24]
for i in range(len(temps)):
	if i < interval: continue
	y.append(temps[i])
	xa = []
	for p in range(interval):
		d = i + p - interval
		xa.append(temps[d])
	x.append(xa)
print("x:", x)
print("y:", y)