# importing required libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# reading csv file and extracting class column to y. 
x = pd.read_csv("C:\...\cancer.csv") 
a = np.array(x) 
y = a[:,30] # classes having 0 and 1 

# extracting two features 
x = np.column_stack((x.malignant,x.benign)) 
x.shape # 569 samples and 2 features 

print (x),(y) 

