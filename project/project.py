# -*- coding: utf-8 -*-
"""
Created on Mon May  2 16:27:51 2022

@author: owenw
"""

import random
import time
import timeit

class Point:
    
    def __init__(self, miles, date, lat, lon):
        self.miles = miles
        self.date = date
        self.lat = lat
        self.lon = lon
        
    def toString(self):
        print(self.miles, self.date, self.lat, self.lon)


if __name__ == '__main__':
    
    data = []
    
    for i in range(8):
        num = 1 * 10 ** i
        
        for i in range(num):
            # if i % 1000000 == 0: print('working')
            p = Point(random.randint(0, 1000),
                              time.time(),
                              random.randint(0, 100) / 90,
                              random.randint(0, 100) / 180)
            
            data.append(p)
        
        start = timeit.default_timer()
        data.sort(key=lambda x: x.lon)
        data.sort(key=lambda x: x.lat)
        data.sort(key=lambda x: x.date)
        data.sort(key=lambda x: x.miles)
        stop = timeit.default_timer()
        diff = stop - start
        print(f'Sort Operations on {num} objects in {diff:6.9f} seconds')  
    
    

    # for i in range(4):
    #     print(f"{i}")
    #     random.shuffle(data)
        
    #     start = timeit.default_timer()
    #     if i >= 3 : data.sort(key=lambda x: x.lon)
    #     if i >= 2 : data.sort(key=lambda x: x.lat)
    #     if i >= 1 : data.sort(key=lambda x: x.date)
    #     if i >= 0 : data.sort(key=lambda x: x.miles)
    #     stop = timeit.default_timer()
    #     diff = stop - start
    #     print(f'Sort Operations on {num} objects in {diff:6.9f} seconds')  
    
    # for i in range(num):
    #     data[i].toString()