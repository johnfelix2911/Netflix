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

def pie_by_count(   #Daksh
    df,
    column="type",
    title="Catalog Composition",
    *,
    colors=None,
    figsize=(8, 6),
    text_color="white",
    font_size=12,
    wedge_edgecolor="white",
    wedge_linewidth=1,
    labeldistance=0.30,
    pctdistance=0.80,
    autopct="%1.1f%%",
    startangle=90,
    show=True,
):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import numpy as np
    """
    Draws a pie chart from value counts of a dataframe column.
    """
    if column not in df.columns:
        raise KeyError(f"'{column}' not in dataframe columns")

    counts = df[column].value_counts(dropna=False)
    labels = counts.index.astype(str)

    fig, ax = plt.subplots(figsize=figsize)
    ax.pie(
        counts.values,
        labels=labels,
        colors=colors,
        textprops={"color": text_color, "fontsize": font_size},
        wedgeprops={"edgecolor": wedge_edgecolor, "linewidth": wedge_linewidth},
        labeldistance=labeldistance,
        pctdistance=pctdistance,
        autopct=autopct,
        startangle=startangle,
    )
    ax.set_title(title)
    ax.axis("equal")
    if show:
        plt.show()
def barh_top_counts_series(   #Daksh
    s,
    *,
    top_n: int | None = None,
    title: str = "Top Categories (Count & Share)",
    xlabel: str = "Count",
    color: str = "#E50914",
    fontsize: int = 10,
    x_margin: float = 0.12,
    total: int | None = None,   # denominator for percentages; default = s.sum()
    show: bool = True,
    save_path: str | None = None,
):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import seaborn as sns
    """
    Draw a horizontal bar chart from a counts Series (index=labels, values=counts).
    """
    if not isinstance(s, pd.Series):
        raise TypeError("Expected a pandas Series of counts.")

    # Use provided Series; optionally keep only top_n
    s_plot = s.sort_values(ascending=False)
    if top_n is not None:
        s_plot = s_plot.head(top_n)

    denom = total if total is not None else int(s_plot.sum())

    # Reverse for largest at top in barh
    labels = s_plot.index[::-1].astype(str)
    values = s_plot.values[::-1]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(labels, values, color=color)
    ax.margins(x=x_margin)

    # Annotate: "count (pct)"
    for i, v in enumerate(values):
        pct = (v / denom) if denom else 0.0
        ax.text(v, i, f" {v} ({pct:.1%})", va="center", color="#111", fontsize=fontsize)

    ax.set(title=title, xlabel=xlabel, ylabel="")
    plt.tight_layout()

    if show:
        plt.show()