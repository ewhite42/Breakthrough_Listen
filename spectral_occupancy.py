import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import turbo_seti.find_event as find
import glob
import argparse

def read_txt(text_file):
    """
    reads a text file with one filepath per
    line and returns a python list where
    each entry is a filepath
    
    Arguments
    ----------
    text_file : str
        A string indicating the location of the 
        text file pointing to the dat files 
    """
    with open(text_file) as open_file:
        lines = open_file.readlines()
    
    for i in range(len(lines)):
        lines[i] = lines[i].replace("\n", "")
    return lines

def calculate_hist(dat_file, bin_width=1): 
    """
    calculates a histogram of the number of hits for a single .dat file
    
    Arguments
    ----------
    dat_file : str
        filepath to the .dat file
    bin_width : float
        width of the hisrogram bins in units of MHz
        The default is 1 Mhz
        
    Returns
    --------
    hist : numpy.ndarray 
        the count of hits in each bin
    bin_edges : numpy.ndarray
        the edge values of each bin. Has length Nbins+1
    """
    #read the file into a pandas dataframe
    tbl = find.read_dat(dat_file)

    #make the bins for the histogram
    min_freq = int(min(tbl["Freq"]))
    max_freq = np.round(max(tbl["Freq"]))
    bins = np.linspace(min_freq, max_freq, int((max_freq-min_freq)/bin_width) , endpoint=True)
    hist, bin_edges = np.histogram(tbl["Freq"], bins=bins)
    return hist, bin_edges

def calculate_proportion(file_list, bin_width=1, GBT_L=False, GBT_S=False):
    """
    Takes in a list of .dat files and makes a true/false table of hits in a frequency bin
    
    Arguments
    ----------
    file_list : list
        A python list containing the filepaths to .dat 
        files that will be used to calculate the 
        spcetral occupancy
    bin_width : float
        width of the hisrogram bins in MHz
    GBT_L : bool
        Indicates whether the data comes from the L-band
        of Green Bank Telescope. If set to true, the 
        data between 1200-1341 Mhz will be excluded due
        to the notch filter in that range
    GBT_S : bool
        Indicates whether the data comes from the S-band
        of Green Bank Telescope. If set to true, the 
        data between 2300-2360 Mhz will be excluded due
        to the notch filter in that range
    """
    edges = []
    histograms = []
    min_freq = 0
    max_freq = 1e9
    
    print("Calculating histograms...",end="")
    #calculate histogram for the .dat file and check the boundaries on the data
    for file in file_list:
        hist, bin_edges = calculate_hist(file, bin_width)
        if min(bin_edges) > min_freq:
            min_freq = min(bin_edges)
        if max(bin_edges) < max_freq:
            max_freq = max(bin_edges)
        edges.append(bin_edges)
        histograms.append(hist)
    print("Done.")
    
    #make sure all lists are within the boundaries
    print("Trimming edges...",end="")
    for i in range(len(edges)):
        within_boundaries = np.where( (edges[i] >= min_freq) & (edges[i] <= max_freq) ) #get the boundaries of the tightest frequency range
        edges[i] = edges[i][within_boundaries] # take only the entries within that range
        freq_boundaries = within_boundaries[0][:-1] # since the bins list has one more entry than frequencies, I will drop the last entry. the hit count will correspond with the frequency at the start of its bin
        histograms[i] = histograms[i][freq_boundaries] # take only the entries within that range
    print("Done.")    
    
    #create the dataframe and add the frequency bins to column 0
    df = pd.DataFrame()
    df.insert(0, "freq", edges[0][:-1])
    
    #check if there is a hit in the frequency bin and insert value to dataframe
    for i in range(len(histograms)):
        colname = "file"+str(i)
        found_hit = histograms[i] > 0
        df.insert(len(df.columns), colname, found_hit.astype(int))
    
    #exclude entries in the GBT data due to the notch filter exclusion
    bin_edges = np.linspace(min_freq, max_freq, int((max_freq-min_freq)/bin_width), endpoint=True)
    if GBT_L:
        print("Excluding hits in the range 1200-1341 MHz")
        df = df[(df["freq"] < 1200) | (df["freq"] > 1341)]
        first_edge = np.arange(min_freq, 1200, bin_width)
        second_edge= np.arange(1341, max_freq, bin_width) #may or may not need max_freq+1
        bin_edges = np.append(first_edge, second_edge)
    
    if GBT_S:
        print("Excluding hits in the range 2300-2360 MHz")
        df = df[(df["freq"] < 2300) | (df["freq"] > 2360)]
        first_edge = np.arange(min_freq, 2300, bin_width)
        second_edge= np.arange(2360, max_freq, bin_width) #may or may not need max_freq+1
        bin_edges = np.append(first_edge, second_edge)

     
    # sum up the number of entries that have a hit and divide by the number of .dat files
    data_labels = df.columns[2:]
    total = df["file0"]
    for label in data_labels:
        total = total + df[label]
    
    return bin_edges, total/len(file_list)  



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generates a histogram of the spectral occupancy from a given set of .dat files")
    parser.add_argument("-folder", "-f", help="directory .dat files are held in")
    parser.add_argument("-t", help="a .txt file to read the filepaths of the .dat files", action=None)
    parser.add_argument("-bin_width", "-b", help="width of bin in Mhz", type=float, default=1)
    parser.add_argument("-GBTL", help="data was collected from Green Bank Telescope L-band", action="store_true")
    parser.add_argument("-GBTS", help="data was collected from Green Bank Telescope S-band", action="store_true")
    args = parser.parse_args()
    
    print("Gathering files...",end="")
    if args.t == None:
        dat_files = glob.glob(args.folder+"/*.dat")
    else:
        dat_files = read_txt(args.t)
    print("Done.")
    
    bin_edges, prob_hist = calculate_proportion(dat_files, bin_width=args.bin_width, GBT_L=args.GBTL, GBT_S=args.GBTS)
    
    print("Saving plot...",end="")
    plt.figure(figsize=(20, 10))
    plt.bar(bin_edges[:-1], prob_hist, width = .99) 
    plt.xlabel("Frequency [Mhz]")
    plt.ylabel("Fraction with Hits")
    plt.title("Spectral Occupancy")
    plt.savefig("spectral_occupancy.pdf")
    print("Done")