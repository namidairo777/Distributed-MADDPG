from multiprocessing import Pool, Array

def test(a):
	q[0] = q[5]
	return 1

if __name__ == "__main__":
	q = Array("i", [1,2,3,4,5,6])
	with Pool(processes=1) as pool:
		pool.map(test, range(4))
		print(q)