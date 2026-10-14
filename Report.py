import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages 
 
# Generating the report 

with PdfPages('report.pdf') as pdf:
    fig = plt.figure(figsize=[8.27, 11.69]) 
    #Page one
    fig.suptitle('Exploratory Data Analysis Report ',fontsize=16, weight='bold' ) #Title
    #Introduction
    plt.text(0.5,0.5, 'Introduction')
    #Dataset Overview
    #Data Cleaning and Loading
    
    #Page Two
    #Data Summary and Statistics
    #

    
    
    pdf.savefig(fig)
    plt.close(fig)