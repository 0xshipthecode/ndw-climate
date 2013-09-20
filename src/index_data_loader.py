# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:14:18 2013

@author: hartman
"""
import csv
import numpy as np

def loadPSDstar(input_filename, start_year = None, end_year = None):
    
    data = []
    ts = []
    years = []
    with open(input_filename, 'rb') as f:
        reader = csv.reader(f, delimiter=' ', skipinitialspace = True)
        for row in reader:
            if (len(row) > 0):
                if (row[-1] == ''):
                    row = row[:-1]
                first = row[0]
                if ((not first.isdigit()) and len(first) > 5):
                    first = first[:4]                
                if (len(row) == 13 and first.isdigit()):
                    y = int(first)
                    if ((start_year == None or y >= start_year) and (end_year == None or y < end_year)):
                        row_data = map(float,row[1:])
                        data.append(row_data)
                        ts.extend(row_data)
                        years.append(y)
    
    resulting_data = {}
    resulting_data['ts'] = np.array(ts)
    resulting_data['data'] = np.array(data)
    resulting_data['years'] = np.array(years)
    return resulting_data    

def loadPSD(input_filename, start_year = None, end_year = None):
    
    data = []
    ts = []
    years = []
    with open(input_filename, 'rb') as f:
        reader = csv.reader(f, delimiter=' ', skipinitialspace = True)
        for row in reader:
            if (len(row) > 0):
                if (row[-1] == ''):
                    row = row[:-1]
                if (len(row) == 13 and row[0].isdigit()):
                    y = int(row[0])
                    if ((start_year == None or y >= start_year) and (end_year == None or y < end_year)):
                        row_data = map(float,row[1:])
                        data.append(row_data)
                        ts.extend(row_data)
                        years.append(y)
    
    resulting_data = {}
    resulting_data['ts'] = np.array(ts)
    resulting_data['data'] = np.array(data)
    resulting_data['years'] = np.array(years)
    return resulting_data    

def load_two_column(input_filename, start_year = None, end_year = None):
    
    data = []
    ts = []
    years = []
    with open(input_filename, 'rb') as f:
        reader = csv.reader(f, delimiter=' ', skipinitialspace = True)
        monthly_data = []
        for row in reader:
            if (len(row) == 2 and row[0].isdigit()):
                r = row[0]
                #print r
                y = int(r[:4])
                m = int(r[4:])
                val = float(row[1])
                if ((start_year == None or y >= start_year) and (end_year == None or y < end_year)):
                    if (len(monthly_data) < 11):
                        monthly_data.append(val)
                    else:
                        monthly_data.append(val)
                        data.append(monthly_data)
                        monthly_data = []
                        years.append(y)
                        
                    ts.append(val)                    
    
    resulting_data = {}
    resulting_data['ts'] = np.array(ts)
    resulting_data['data'] = np.array(data)
    resulting_data['years'] = np.array(years)
    return resulting_data    

def load_three_column(input_filename, start_year = None, end_year = None):
    
    data = []
    ts = []
    years = []
    with open(input_filename, 'rb') as f:
        reader = csv.reader(f, delimiter=' ', skipinitialspace = True)
        monthly_data = []
        for row in reader:
            if (len(row) == 3 and row[0].isdigit()):
                y = int(row[0])
                m = int(row[1])
                val = float(row[2])
                if ((start_year == None or y >= start_year) and (end_year == None or y < end_year)):
                    if (len(monthly_data) < 11):
                        monthly_data.append(val)
                    else:
                        monthly_data.append(val)
                        data.append(monthly_data)
                        monthly_data = []
                        years.append(y)
                        
                    ts.append(val)                    
    
    resulting_data = {}
    resulting_data['ts'] = np.array(ts)
    resulting_data['data'] = np.array(data)
    resulting_data['years'] = np.array(years)
    return resulting_data    


