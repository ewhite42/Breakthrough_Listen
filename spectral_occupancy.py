import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import turbo_seti.find_event as find
import glob
import argparse

parser = argparse.ArgumentParser(description="generates a histogram of the spectral occupancy from a given set of .dat files")
parser.add_argument("folder", help="directory to L-Band .dat files")
parser.add_argument("bin_width", help="width of bin in Mhz", type=float)
parser.add_argument("-GBT", help="data was collected from Green Bank Telescope", action="store_true", default=False)
args = parser.parse_args()

dat_files = glob.glob(args.folder+"*.dat")


def calculate_hist(dat_file, bin_width=1): 
    """calculates a histogram of the number of hits for a single .dat file"""
    #read the file into a pandas dataframe
    tbl = find.read_dat(dat_file)

    #make the bins for the histogram
    min_freq = int(min(tbl["Freq"]))
    max_freq = np.round(max(tbl["Freq"]))
    bins = np.linspace(min_freq, max_freq, int((max_freq-min_freq)/bin_width) , endpoint=True)
    hist, bin_edges = np.histogram(tbl["Freq"], bins=bins)
    return hist, bin_edges


def calculate_proportion(file_list, bin_width=1, GBT=False):
    """Takes in a list of .dat files and makes a true/false table of hits in a frequency bin"""
    edges = []
    histograms = []
    min_freq = 0
    max_freq = 1e9
    
    #calculate histogram for the .dat file and check the boundaries on the data
    for file in file_list:
        hist, bin_edges = calculate_hist(file, bin_width)
        if min(bin_edges) > min_freq:
            min_freq = min(bin_edges)
        if max(bin_edges) < max_freq:
            max_freq = max(bin_edges)
        edges.append(bin_edges)
        histograms.append(hist)
    
    #make sure all lists are within the boundaries
    for i in range(len(edges)):
        within_boundaries = np.where( (edges[i] >= min_freq) & (edges[i] <= max_freq) ) #get the boundaries of the tightest frequency range
        edges[i] = edges[i][within_boundaries] # take only the entries within that range
        freq_boundaries = within_boundaries[0][:-1] # since the bins list has one more entry than frequencies, I will drop the last entry. the hit count will correspond with the frequency at the start of its bin
        histograms[i] = histograms[i][freq_boundaries] # take only the entries within that range
        
    #create the dataframe and add the frequency bins to column 0
    df = pd.DataFrame()
    df.insert(0, "freq", edges[0][:-1])
    
    #check if there is a hit in the frequency bin and insert value to dataframe
    for i in range(len(histograms)):
        colname = "file"+str(i)
        found_hit = histograms[i] > 0
        df.insert(len(df.columns), colname, found_hit.astype(int))
    
    #exclude entries in the GBT data due to the notch filter exclusion
    #bin_edges = np.arange(min_freq, max_freq+.1, bin_width)
    bin_edges = np.linspace(min_freq, max_freq, int((max_freq-min_freq)/bin_width), endpoint=True)
    if GBT:
        df = df[(df["freq"] < 1200) | (df["freq"] > 1341)]
        first_edge = np.arange(min_freq, 1200, bin_width)
        second_edge= np.arange(1341, max_freq, bin_width) #may or may not need max_freq+1
        bin_edges = np.append(first_edge, second_edge)
    
        
    # sum up the number of entries that have a hit and divide by the number of .dat files
    data_labels = df.columns[2:]
    total = df["file0"]
    for label in data_labels:
        total = total + df[label]
    
    return bin_edges, total/len(file_list) 

bin_edges, prob_hist = calculate_proportion(dat_files, bin_width=args.bin_width, GBT=args.GBT)

plt.figure(figsize=(20, 10))
plt.bar(bin_edges[:-1], prob_hist)#, width = 1) 
plt.xlabel("Frequency [Mhz]")
plt.ylabel("Probability of hit")
plt.title("GBT L-Band Spectral Occupancy")
plt.savefig("spectral_occupancy.pdf")
