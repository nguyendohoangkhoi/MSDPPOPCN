# -*- coding: utf-8 -*-
import time
from multiprocessing import Process
def Temp(i):
#     for j in range(1000):
        print("process ",i," start")
        time.sleep(5)
        print("process ",i," end")
p1 = Process(target=Temp,args=(1,))
p2 = Process(target=Temp,args=(2,))
if __name__ == '__main__':
    p1.start()
    p2.start()