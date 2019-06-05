# -*- coding: utf-8 -*-


import xlwt
from tempfile import TemporaryFile
import numpy as np

def saveTrainResult(filename,titles,trainResults):
#    trainResult = np.load('trainResult.npy')
    book = xlwt.Workbook()
    sheet1 = book.add_sheet('sheet1',cell_overwrite_ok=True)
    
    supersecretdata = trainResults
    sheet1.write(0,0,titles[0])
    sheet1.write(0,1,titles[1])
    sheet1.write(0,2,titles[2])
    sheet1.write(0,3,titles[3])
    for i,e in enumerate(supersecretdata):
        for j,d in enumerate(e):
            sheet1.write(i+1,j,d)
    
    name = filename
    book.save(name)
    book.save(TemporaryFile())