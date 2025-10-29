def generate_boxplot(df,cat,val,outlier=True): # John

    # INPUTS:
    # df (pandas dataframe) -> the dataframe that contains the data
    # cat (string) -> Name of column. Each category(discrete) gets a boxplot
    # val (string) -> Name of column. Value(continuous) for which the boxplot is created
    # outlier (boolean) -> True or False. whether or not the outliers should be plotted (True by default)

    # RETURNS
    # returns nothing as such but it prints the boxplot

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import sklearn
    # computes median value per genre
    genre_medians = df.groupby(cat)[val].median().sort_values(ascending=False)
    
    plt.figure(figsize=(20, 10))
    sns.boxplot(
        data=df,
        x=cat,
        y=val,
        order=genre_medians.index, # sort by median
        showfliers=outlier,
        color='red'
    )
    
    plt.xlabel(cat, fontsize=14)
    plt.ylabel(val, fontsize=14)
    plt.title("Boxplot of "+val+" by "+cat+" (Sorted by Median)", fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(val+" vs "+cat+" sorted.png")
    plt.show()

def Ftest(df,level,val,alpha=0.05): # John

    # INPUTS
    # df (pandas dataframe) -> dataframe that contains the data
    # level (string) -> Name of column. Discrete data
    # val (string) -> Name of column. Continuous data
    # alpha (float) -> level of significance (0.05 by default)

    # RETURNS
    # the F-statistic, the critical value of F and the sum of squared errors which is needed by the LSD test

    from scipy.stats import f

    df=df[df[level]!='missing']
    
    a=df[level].nunique()
    N=df.shape[0]
    
    y_dot_dot_bar=df[val].mean()
    
    genres=list(set(df[level]))
    SST=0
    SSE=0
    for genre in genres:
        data=df[df[level]==genre]
        ni=data.shape[0]
        yi_dot_bar=sum(data[val])/ni
        SST+=ni*((yi_dot_bar-y_dot_dot_bar)**2)
        sse=0
        ratings=list(data[val])
        for r in ratings:
            sse+=(r-yi_dot_bar)**2
        SSE+=sse
    f_stat=(SST/(a-1))/(SSE/(N-a))
    f_crit=f.ppf(1-alpha,a-1,N-a)
    return f_stat,f_crit,SSE

def LSD_test(df,level,val,SSE,alpha=0.05): # John

    # INPUTS
    # df (pandas dataframe) -> dataframe that contains the data
    # level (string) -> name of the column (discrete data)
    # val (string) -> name of the column (continuous data)
    # SSE (float) -> Sum of Squared Errors from the F-test model
    # alpha (float) -> level of significance (default 0.05)

    # RETURNS
    # dictionary that contains each level as its keys and lists as its values. These lists contain the names of the genres for which the key genre has a statisctically higher average value 

    a=df[level].nunique()
    N=df.shape[0]
    genres=list(set(df[level]))
    df=df[df[level]!='missing']
    from scipy.stats import t
    from math import sqrt
    t_val=t.ppf(1-alpha/2,N-a)
    MSE=SSE/(N-a)
    dic={}
    for genre in genres:
        dic[genre]=[]
    for i in range(len(genres)):
        for j in range(i+1,len(genres)):
            g1=genres[i]
            g2=genres[j]
            if g1!=g2:
                n1=df[df[level]==g1].shape[0]
                n2=df[df[level]==g2].shape[0]
                LSD=t_val*sqrt(MSE*(1/n1+1/n2))
                y1_bar=sum(df[df[level]==g1][val])/n1
                y2_bar=sum(df[df[level]==g2][val])/n2
                if abs(y1_bar-y2_bar)>LSD:
                    if y1_bar>y2_bar:
                        dic[g1].append(g2)
                    else:
                        dic[g2].append(g1)
    return dic