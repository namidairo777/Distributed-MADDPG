import threading
import time

from queue import Queue

def job1(l,q):
    for i in l:
        print("job1 is running")
        print(time.time())
        q.put(i)

def job2(l,q):
    
    
    for i in l:
        print("job2 is running")
        print(time.time())
        q.put(i)

def multithreading():
    q =Queue()
    threads = []
    data1 = [[1,1,1],[2,2,2],[3,3,3]]
    data2 = [[4,4,4],[5,5,5],[6,6,6]]
    t = threading.Thread(target=job1,args=(data1,q))
    t.start()
    t = threading.Thread(target=job2,args=(data2,q))
    t.start()
    threads.append(t)
    for thread in threads:
        thread.join()
    results = []
    for _ in range(q.qsize()):
        results.append(q.get())
    print(results)

if __name__=='__main__':
    multithreading()