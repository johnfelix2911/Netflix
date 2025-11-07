import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re

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

def generate_boxplot_interactive(df, cat, val, outlier=True):  # John
    """
    INPUTS:
    df (pandas DataFrame): The dataframe that contains the data
    cat (string): Name of column. Each category (discrete) gets a boxplot
    val (string): Name of column. Value (continuous) for which the boxplot is created
    outlier (boolean): Whether or not the outliers should be plotted (True by default)

    RETURNS:
    Displays an interactive Plotly boxplot
    """
    import pandas as pd
    import plotly.express as px

    # Compute median values per category and sort
    cat_medians = df.groupby(cat)[val].median().sort_values(ascending=False)

    # Create Plotly boxplot
    fig = px.box(
        df,
        x=cat,
        y=val,
        category_orders={cat: cat_medians.index},  # Sort categories by median
        points='outliers' if outlier else False,
        color_discrete_sequence=['#E50914'],  # Netflix red
    )

    fig.update_layout(
        title=f"Boxplot of {val} by {cat} (Sorted by Median)",
        xaxis_title=cat,
        yaxis_title=val,
        template='plotly_dark',  # Dark Netflix-style theme
        title_font=dict(size=22, color='white'),
        xaxis=dict(title_font=dict(size=16), tickfont=dict(size=12)),
        yaxis=dict(title_font=dict(size=16), tickfont=dict(size=12)),
        plot_bgcolor='#141414',
        paper_bgcolor='#141414',
    )

    fig.show()

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
def tv_genre_rating_sentiment_scatter( #Daksh
    df,
    x_col="avg_rating",
    y_col="avg_sent",
    size_col="tv_show_count",
    label_col="genres",
    title="TV Genres on Netflix: Rating vs Sentiment (VADER)",
    width=1000,
    height=650,
):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import numpy as np
    import plotly.express as px
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        size=size_col,
        color=y_col,
        color_continuous_scale=["#E50914", "#B20710", "#FFFFFF"],
        hover_data={
            label_col: True,
            x_col: ":.2f",
            y_col: ":.3f",
            size_col: True,
        },
        title=title,
    )

    fig.update_traces(
        marker=dict(line=dict(width=1, color="white")),
        selector=dict(mode="markers"),
    )

    fig.update_layout(
        width=width,
        height=height,
        template="plotly_dark",
        title_font=dict(size=22, color="white"),
        xaxis_title="Average Rating",
        yaxis_title="Average Sentiment (compound)",
        plot_bgcolor="#141414",
        paper_bgcolor="#141414",
        coloraxis_colorbar=dict(title="Avg Sentiment", tickfont=dict(color="white")),
    )
    return fig
def tv_genre_popularity_sentiment_scatter(#Daksh
    df,
    x_col="avg_popularity",
    y_col="avg_sent",
    size_col="tv_show_count",
    label_col="genres",
    title="TV Show Genres on Netflix: Popularity vs Sentiment (VADER)",
    width=1000,
    height=650,
):
    """
    Bubble scatter for TV genres: popularity vs sentiment.
    df must contain [x_col, y_col, size_col, label_col].
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import numpy as np
    import plotly.express as px
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        size=size_col,
        color=y_col,
        color_continuous_scale=["#E50914", "#B20710", "#FFFFFF"],  # netflix-ish
        hover_data={
            label_col: True,
            x_col: ":.2f",
            y_col: ":.3f",
            size_col: True,
        },
        title=title,
    )

    fig.update_traces(
        marker=dict(line=dict(width=1, color="white")),
        selector=dict(mode="markers"),
    )

    fig.update_layout(
        width=width,
        height=height,
        template="plotly_dark",
        title_font=dict(size=22, color="white"),
        xaxis_title="Average Popularity",
        yaxis_title="Average Sentiment (compound)",
        plot_bgcolor="#141414",
        paper_bgcolor="#141414",
        coloraxis_colorbar=dict(title="Avg Sentiment", tickfont=dict(color="white")),
    )
    return fig
def genre_rating_sentiment_scatter( #Daksh
    df,
    x_col="avg_rating",
    y_col="avg_sent",
    size_col="movie_count",
    label_col="genres",
    title="Movie Genres on Netflix: Rating vs Sentiment (VADER)",
    width=1000,
    height=650,
):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import numpy as np
    import plotly.express as px
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        size=size_col,
        color=y_col,
        color_continuous_scale=["#E50914", "#B20710", "#FFFFFF"],  # netflix-ish
        hover_data={
            label_col: True,
            x_col: ":.2f",
            y_col: ":.3f",
            size_col: True,
        },
        title=title,
    )

    fig.update_traces(
        marker=dict(line=dict(width=1, color="white")),
        selector=dict(mode="markers"),
    )

    fig.update_layout(
        width=width,
        height=height,
        template="plotly_dark",
        title_font=dict(size=22, color="white"),
        xaxis_title="Average Rating",
        yaxis_title="Average Sentiment (compound)",
        plot_bgcolor="#141414",
        paper_bgcolor="#141414",
        coloraxis_colorbar=dict(title="Avg Sentiment", tickfont=dict(color="white")),
    )
    return fig
def genre_popularity_sentiment_scatter(#Daksh
    df,
    x_col="avg_popularity",
    y_col="avg_sent",
    size_col="movie_count",
    label_col="genres",
    title="Movie Genres on Netflix: Popularity vs Sentiment (VADER)",
    width=1000,
    height=650,
):
    """
    Build a bubble scatter of genre popularity vs sentiment.

    Parameters
    ----------
    df : DataFrame with columns [x_col, y_col, size_col, label_col]
    x_col : str, column for x-axis (e.g., "avg_popularity")
    y_col : str, column for y-axis (e.g., "avg_sent")
    size_col : str, bubble size column (e.g., "movie_count")
    label_col : str, label shown on hover (e.g., "genres")
    title : str, chart title
    width, height : int, figure size

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import numpy as np
    import plotly.express as px
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        size=size_col,
        color=y_col,
        color_continuous_scale=["#E50914", "#B20710", "#FFFFFF"],  # netflix-ish
        hover_data={
            label_col: True,
            x_col: ":.2f",
            y_col: ":.3f",
            size_col: True,
        },
        title=title,
    )

    fig.update_traces(
        marker=dict(line=dict(width=1, color="white")),
        selector=dict(mode="markers"),
    )

    fig.update_layout(
        width=width,
        height=height,
        template="plotly_dark",
        title_font=dict(size=22, color="white"),
        xaxis_title="Average Popularity",
        yaxis_title="Average Sentiment (compound)",
        plot_bgcolor="#141414",
        paper_bgcolor="#141414",
        coloraxis_colorbar=dict(title="Avg Sentiment", tickfont=dict(color="white")),
    )
    return fig
def pie_by_count(   #Daksh
    df,
    column="type",
    title="Catalog Composition",
    *,
    colors=["#E50914", "#000000"],
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
    """
    Draws a pie chart from value counts of a dataframe column.
    Inputs:-
    1)df : pd.DataFrame
        Dataframe containing the data.
    2)column : string
        Name of the column to plot.
    3)title : string
        Title of the pie chart.
    """
     
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import numpy as np

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
    plt.figtext(0.5, 0.08, title, 
            wrap=True, horizontalalignment='center', fontsize=12)
    ax.axis("equal")
    if show:
        plt.show()

def barh_top_counts_series(   #Daksh
    s,
    *,
    title: str = "Top Categories (Count & Share)",
    figtitle,
    xlabel: str = "Count",
    color: str = "#E50914",
    fontsize: int = 10,
    x_margin: float = 0.12,
    total: int | None = None,   # denominator for percentages; default = s.sum()
    show: bool = True,
    save_path: str | None = None,
):
    """
    Draw a horizontal bar chart from a counts Series (index=labels, values=counts).
    Inputs:-
    1)s : pd.Series
        Series of counts (index=labels, values=counts).
    2)title : string
    3)xlabel : string
    4)color : string
    5)figtitle : string
        Figure title.
        Bar color.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import seaborn as sns
    if not isinstance(s, pd.Series):
        raise TypeError("Expected a pandas Series of counts.")

    # Use provided Series;
    s_plot = s.sort_values(ascending=False)

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
    plt.figtext(0.5, 0, figtitle, 
            wrap=True, horizontalalignment='center', fontsize=12)
    plt.tight_layout()

    if show:
        plt.show()

def barh_top_counts_series_black_background(   #Daksh
    s,
    *,
    title: str = "Top Categories (Count & Share)",
    # figtitle,
    xlabel: str = "Count",
    color: str = "#E50914",
    fontsize: int = 10,
    x_margin: float = 0.12,
    total: int | None = None,   # denominator for percentages; default = s.sum()
    show: bool = True,
    save_path: str | None = None,
):
    """
    Draw a horizontal bar chart from a counts Series (index=labels, values=counts).
    Inputs:-
    1)s : pd.Series
        Series of counts (index=labels, values=counts).
    2)title : string
    3)xlabel : string
    4)color : string
    5)figtitle : string
        Figure title.
        Bar color.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import seaborn as sns
    if not isinstance(s, pd.Series):
        raise TypeError("Expected a pandas Series of counts.")

    # Use provided Series;
    s_plot = s.sort_values(ascending=False)

    denom = total if total is not None else int(s_plot.sum())

    # Reverse for largest at top in barh
    labels = s_plot.index[::-1].astype(str)
    values = s_plot.values[::-1]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(labels, values, color=color)
    ax.margins(x=x_margin)

    #set background to black
    ax.set_facecolor("#000000")  
    fig.patch.set_facecolor("#000000")

    # Annotate: "count (pct)"
    for i, v in enumerate(values):
        pct = (v / denom) if denom else 0.0
        ax.text(v, i, f" {v} ({pct:.1%})", va="center", color="white", fontsize=fontsize)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.set_title(title,color="white") 
    ax.set_xlabel(xlabel,color="white")
    # plt.figtext(0.5, 0, figtitle, 
    #         wrap=True, horizontalalignment='center', fontsize=12)
    plt.tight_layout()

    if show:
        plt.show()
def plot_sentiment_by_genre(sent_pivot,title): #Daksh
    """
    Plot stacked sentiment distribution across genres (Netflix style).
    sent_pivot: index=genre, columns=['positive','neutral','negative'], values=%
    """
    import matplotlib.pyplot as plt
    sent_pivot = sent_pivot[["positive", "neutral", "negative"]]

    ax = sent_pivot.plot(
        kind="bar",
        stacked=True,
        figsize=(12, 6),
        color=["#E50914", "#221F1F", "#B3B3B3"],  # netflix-ish
        edgecolor="white",
    )

    plt.ylabel("% of titles")
    plt.title(title, fontsize=14, weight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Sentiment", bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    plt.show()
    
def genre_wordcloud(df, col='genres', title='Genre Popularity Word Cloud'): # John
    """
    Generate and display a word cloud of popular genres (or any categorical feature).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the genre or categorical column.
    col : str, optional
        Column name containing genre or categorical data (default='genres').
    title : str, optional
        Title of the plot (default='Genre Popularity Word Cloud').

    Returns
    -------
    WordCloud
        The generated WordCloud object (for further use or saving).
    """

    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    from collections import Counter

    # --- Step 1: Extract and clean genre data ---
    # Ensure each entry is a list
    genres_list = []
    for entry in df[col]:
        if isinstance(entry, str):
            # Split by comma if stored as string
            genres_list.extend([g.strip() for g in entry.split(',') if g.strip()])
        elif isinstance(entry, list):
            genres_list.extend(entry)
        # ignore missing/invalid entries silently

    # --- Step 2: Count frequency of each genre ---
    genre_counts = Counter(genres_list)

    if not genre_counts:
        print("No valid genre data found. Please check your column format.")
        return None

    # --- Step 3: Generate word cloud ---
    wc = WordCloud(
        width=1200,
        height=600,
        background_color='white',
        colormap='plasma',
        prefer_horizontal=0.9
    ).generate_from_frequencies(genre_counts)

    # --- Step 4: Display ---
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=18)
    plt.tight_layout()
    plt.show()

    return wc

def generate_treemap(df, cat, val): # John
    """
    Creates an interactive treemap showing contribution of each category to a continuous variable.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the data.
    cat : str
        The name of the categorical column (e.g. 'genres').
    val : str
        The name of the continuous column (e.g. 'revenue').

    Returns
    -------
    Displays an interactive treemap (Plotly figure).
    """
    import pandas as pd
    import plotly.express as px

    # Aggregate data
    grouped = (
        df.groupby(cat, dropna=True)[val]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    # Create treemap
    fig = px.treemap(
        grouped,
        path=[cat],
        values=val,
        color=val,
        color_continuous_scale=['#E50914', '#B20710', '#831010'],  # Netflix-themed red shades
        title=f"Contribution of {cat} to Total {val}",
    )

    # Update layout for Netflix-style aesthetics
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#141414',
        plot_bgcolor='#141414',
        title_font=dict(size=22, color='white'),
    )

    fig.show()


# def generate_interactive_scatter(df, x, y, color=None, hover=None):
#     """
#     Creates an interactive Netflix-themed scatter plot using Plotly.

#     Parameters
#     ----------
#     df : pandas.DataFrame
#         The dataframe containing the data.
#     x : str
#         The name of the column for the x-axis (e.g. 'budget').
#     y : str
#         The name of the column for the y-axis (e.g. 'revenue').
#     color : str, optional
#         The name of the categorical column to color the points by (e.g. 'genres').
#     hover : str or list, optional
#         Column(s) to display when hovering (e.g. 'title' or ['title', 'rating']).

#     Returns
#     -------
#     Displays an interactive scatter plot.
#     """
#     import pandas as pd
#     import plotly.express as px

#     # Drop rows with missing required values
#     cols = [x, y]
#     if color:
#         cols.append(color)
#     if hover:
#         if isinstance(hover, list):
#             cols.extend(hover)
#         else:
#             cols.append(hover)
#     df_clean = df.dropna(subset=cols)

#     # Create interactive scatter plot
#     fig = px.scatter(
#         df_clean,
#         x=x,
#         y=y,
#         color=color,
#         hover_data=hover,
#         title=f"{y} vs {x}" + (f" colored by {color}" if color else ""),
#         color_discrete_sequence=['#E50914', '#B81D24', '#831010', '#221f1f'],  # Netflix reds
#     )

#     # Style for Netflix dark theme
#     fig.update_layout(
#         template='plotly_dark',
#         paper_bgcolor='#141414',
#         plot_bgcolor='#141414',
#         font=dict(color='white'),
#         title_font=dict(size=22),
#         xaxis_title=x.title(),
#         yaxis_title=y.title(),
#         legend_title=color.title() if color else '',
#     )

#     # Smooth marker style
#     fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='white')))

#     fig.show()

def generate_interactive_scatter(df, x, y, color=None, hover=None): # John
    """
    Creates an interactive scatter plot using Plotly (dark theme + distinct colors).

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the data.
    x : str
        The name of the column for the x-axis (e.g. 'budget').
    y : str
        The name of the column for the y-axis (e.g. 'revenue').
    color : str, optional
        The name of the categorical column to color the points by (e.g. 'genres').
    hover : str or list, optional
        Column(s) to display when hovering (e.g. 'title' or ['title', 'rating']).

    Returns
    -------
    Displays an interactive scatter plot.
    """
    import pandas as pd
    import plotly.express as px

    # Drop rows with missing required values
    cols = [x, y]
    if color:
        cols.append(color)
    if hover:
        if isinstance(hover, list):
            cols.extend(hover)
        else:
            cols.append(hover)
    df_clean = df.dropna(subset=cols)

    # Use a distinct, high-contrast qualitative color palette
    color_palette = px.colors.qualitative.Set3 + px.colors.qualitative.Bold + px.colors.qualitative.Safe

    # Create interactive scatter plot
    fig = px.scatter(
        df_clean,
        x=x,
        y=y,
        color=color,
        hover_data=hover,
        title=f"{y} vs {x}" + (f" colored by {color}" if color else ""),
        color_discrete_sequence=color_palette,
    )

    # Dark Netflix-like theme but with colorful genres
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#141414',
        plot_bgcolor='#141414',
        font=dict(color='white'),
        title_font=dict(size=22),
        xaxis_title=x.title(),
        yaxis_title=y.title(),
        legend_title=color.title() if color else '',
    )

    # Style markers for better readability
    fig.update_traces(marker=dict(size=9, opacity=0.8, line=dict(width=0.5, color='white')))

    fig.show()

def bar_stacked(                    #Sourendra
    df,
    *,
    title: str = "Stacked Bar Chart",
    xlabel: str = "",
    ylabel: str = "Count",
    color: list | None = None,
    legend_title: str = "Type",
    figsize: tuple = (20, 10),
    rotation: int = 45,
    show: bool = True,
    save_path: str | None = None,
):
    """
    Draws a stacked bar chart from a given DataFrame, styled with Netflix brand colors
    on a clean white background for better contrast.

    Inputs:-
    ----------
    1) df : pd.DataFrame
        DataFrame where each column represents a category to be stacked.
        The index will be used as x-axis labels.

    2) title : str, default = "Stacked Bar Chart"
        Title of the bar chart.

    3) xlabel, ylabel : str
        Labels for the x-axis and y-axis.

    4) color : list, optional
        List of color codes to use for the bars.
        If None, the function uses Netflix brand colors on white background:
        ["#E50914" (Netflix Red), "#000000" (Black), "#555555" (Gray), "#B3B3B3" (Light Gray)]

    5) legend_title : str, default = "Type"
        Title displayed on the legend.

    6) figsize : tuple, default = (20, 10)
        Figure size for the chart.

    7) rotation : int, default = 45
        Rotation angle for x-axis tick labels.

    8) show : bool, default = True
        Whether to display the chart immediately.

    9) save_path : str, optional
        If provided, saves the plot to the given path.

    Output:-
    ----------
    Returns the matplotlib Axes object of the plot.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected input 'df' to be a pandas DataFrame.")

    # --- Default Netflix brand colors on white background ---
    if color is None:
        color = ["#E50914", "#000000", "#555555", "#B3B3B3"]

    # --- Set white background for clarity ---
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'

    # --- Plot the stacked bar chart ---
    ax = df.plot(kind="bar", stacked=True, figsize=figsize, color=color[:len(df.columns)], edgecolor='black')

    ax.set_title(title, fontsize=16, color="#000000", pad=10, fontweight="bold")
    ax.set_xlabel(xlabel, color="#000000")
    ax.set_ylabel(ylabel, color="#000000")
    ax.legend(
        title=legend_title,
        title_fontsize=12,
        fontsize=10,
        facecolor="white",
        edgecolor="#000000",
        labelcolor="#000000"
    )

    plt.xticks(rotation=rotation, color="#000000", fontsize=10)
    plt.yticks(color="#000000", fontsize=10)
    plt.grid(color="#DDDDDD", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    # --- Save or show ---
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300, facecolor="white")
    if show:
        plt.show()

    return ax

def bar_chart_vertical(             #Sourendra
    s,
    *,
    title: str = "Bar Chart",
    xlabel: str = "",
    ylabel: str = "Count",
    color: str = "#E50914",
    figsize: tuple = (12, 6),
    annotate: bool = True,
    rotation: int = 0,
    show: bool = True,
    save_path: str | None = None,
):
    """
    Generalized vertical bar chart helper for categorical or time-like count Series.

    Parameters
    ----------
    s : pd.Series
        Index = categories (e.g. months, genres), values = counts.
    title : str
        Chart title.
    xlabel : str
        Label for x-axis.
    ylabel : str
        Label for y-axis.
    color : str
        Color for bars.
    figsize : tuple
        Figure size.
    annotate : bool
        Whether to show count annotations above bars.
    rotation : int
        Rotation angle for x-axis tick labels.
    show : bool
        Whether to display the plot immediately.
    save_path : str
        Optional path to save the figure.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    if not isinstance(s, pd.Series):
        raise TypeError("Expected a pandas Series (index=labels, values=counts).")

    # --- Sort months in natural order if applicable ---
    month_order = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    month_abbr = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    idx_str = s.index.astype(str)

    if all(x in month_order for x in idx_str):
        s_plot = s.reindex(month_order).dropna()
    elif all(x in month_abbr for x in idx_str):
        s_plot = s.reindex(month_abbr).dropna()
    else:
        s_plot = s.sort_index()

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(s_plot.index.astype(str), s_plot.values, color=color)

    # Annotate each bar
    if annotate:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height, f"{int(height)}",
                    ha='center', va='bottom', fontsize=10)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=rotation)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

def heatmap_by_category(            #Sourendra
    df,
    row_col="year_added",
    col_col="month_added",
    value_col="show_id",
    figsize=(20, 10),
    title="Heatmap of Content Added by Year and Month"
):
    """
    Plots a heatmap with a white background and a black-to-red color scale for the cells.
    This function is self-contained and its styling will not affect other plots.
    The Netflix red and black theme is now built-in.

    INPUTS:
        df (pd.DataFrame): DataFrame containing the data.
        row_col (str): Column for Y-axis (default: 'year_added').
        col_col (str): Column for X-axis (default: 'month_added').
        value_col (str): Column whose count is used (default: 'show_id').
        figsize (tuple): Figure size (default: (20, 10)).
        title (str): Plot title (default: 'Heatmap of Content Added by Year and Month').

    RETURNS:
        None (Displays a heatmap)
    """
    
    # --- 1. Self-Contained Imports ---
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap

    # --- 2. Data Preparation ---
    month_order = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    if col_col.lower().startswith("month") and df[col_col].dtype == object:
        df[col_col] = pd.Categorical(df[col_col], categories=month_order, ordered=True)

    pivot_table = (
        df.groupby([row_col, col_col])[value_col]
        .count()
        .unstack(fill_value=0)
        .sort_index(ascending=True)
    )

    # --- 3. Style and Color Definition (Hardcoded) ---
    background_color = 'white'
    text_color = 'black'
    netflix_red = '#E50914'
    
    # The custom black-to-red colormap is now the only option.
    # Low values will be black, high values will be bright red.
    netflix_cmap = LinearSegmentedColormap.from_list(
        "netflix_custom_theme", ["#000000", netflix_red], N=256
    )

    # --- 4. Plotting with Isolated Styling ---
    # Use a 'with' context to ensure style changes are temporary and local.
    with plt.style.context('default'):
        plt.rcParams['figure.facecolor'] = background_color
        plt.rcParams['axes.facecolor'] = background_color
        
        plt.figure(figsize=figsize)
        
        ax = sns.heatmap(
            pivot_table,
            cmap=netflix_cmap,  # Use the hardcoded Netflix colormap
            annot=False,
            linewidths=0.5,
            linecolor='white',
            cbar=True,
            cbar_kws={'label': 'Count'}
        )

        # --- 5. Formatting ---
        plt.title(title, fontsize=18, color=netflix_red, pad=20, fontweight="bold")
        plt.xlabel(col_col.replace("_", " ").title(), color=text_color, fontsize=12, labelpad=10)
        plt.ylabel(row_col.replace("_", " ").title(), color=text_color, fontsize=12, labelpad=10)
        
        plt.xticks(rotation=45, color=text_color)
        plt.yticks(rotation=0, color=text_color)

        cbar = ax.collections[0].colorbar
        cbar.set_label('Count', color=text_color)
        cbar.ax.yaxis.set_tick_params(color=text_color)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=text_color)

        plt.tight_layout()
        plt.show()

def generate_multi_line_chart(      #Sourendra
    data_dict,
    *,
    title="Line Chart Comparison",
    xlabel="X-axis",
    ylabel="Y-axis",
    figsize=(12, 6),
    marker="o",
    linestyle="-",
    grid_alpha=0.3,
):
    """
    Plot multiple lines on the same chart for comparison, styled with a clean white theme
    and red & black lines.

    Parameters
    ----------
    data_dict : dict
        Dictionary where keys are labels (e.g., 'Movies', 'TV Shows')
        and values are pandas Series or lists with numeric indexes (like years).
    title : str
        Title of the plot.
    xlabel : str
        Label for the X-axis.
    ylabel : str
        Label for the Y-axis.
    figsize : tuple
        Figure size of the plot.
    marker : str
        Marker style for each line.
    linestyle : str
        Line style for all plots.
    grid_alpha : float
        Transparency of the grid.
    """
    # --- 1. Self-Contained Imports ---
    import matplotlib.pyplot as plt
    import pandas as pd
    from itertools import cycle

    # --- 2. White Theme Color Palette ---
    background_color = '#FFFFFF'
    text_color = '#000000'
    netflix_red = '#E50914'
    grid_color = '#CCCCCC' # Light grey for the grid
    
    # Define a color cycle for the lines to alternate between red and black
    line_colors = cycle([netflix_red, text_color])

    # --- 3. Plotting with Isolated Styling ---
    # Use plt.style.context() to apply the theme ONLY within this 'with' block.
    with plt.style.context('default'):
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set background colors
        fig.set_facecolor(background_color)
        ax.set_facecolor(background_color)

        # Loop through each dataset in the dictionary
        for label, series in data_dict.items():
            # Convert to Series if list is passed
            if not isinstance(series, pd.Series):
                series = pd.Series(series)
            
            ax.plot(
                series.index,
                series.values,
                marker=marker,
                linestyle=linestyle,
                label=label,
                color=next(line_colors) # Cycle through red and black
            )

        # --- 4. Formatting and Final Touches ---
        # Set title and labels with theme colors
        ax.set_title(title, fontsize=16, color=netflix_red, fontweight='bold', pad=20)
        ax.set_xlabel(xlabel, fontsize=12, color=text_color)
        ax.set_ylabel(ylabel, fontsize=12, color=text_color)

        # Customize grid
        ax.grid(True, alpha=grid_alpha, color=grid_color, linestyle='--')
        
        # Customize ticks and spines (the plot border)
        ax.tick_params(axis='x', colors=text_color)
        ax.tick_params(axis='y', colors=text_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(text_color)
            
        # Customize legend
        legend = ax.legend()
        legend.get_frame().set_facecolor(background_color)
        for text in legend.get_texts():
            text.set_color(text_color)

        plt.tight_layout()
        plt.show()

def generate_heatmap_flexible(      #Sourendra
    df,
    index_col,
    column_col,
    value_col=None,
    aggfunc=None,
    cmap="Reds",
    figsize=(12, 5),
    annot=True,
    fmt=".0f",
    orientation="vertical",
    title=None,
    xlabel=None,
    ylabel=None,
    cbar_label=None,
):
    """
    A flexible and generalized heatmap generator for exploring relationships between
    any two categorical or temporal variables. Supports aggregation, custom colormaps,
    and annotation formatting.

    INPUTS:
        df (pd.DataFrame): Input dataset.
        index_col (str): Column for Y-axis (rows).
        column_col (str): Column for X-axis (columns).
        value_col (str, optional): Column to aggregate. If None, counts entries.
        aggfunc (str or callable, optional): Aggregation function ('count', 'mean', 'sum', etc.).
                                             If None, auto-detects based on value_col type.
        cmap (str): Matplotlib colormap (default 'Reds').
        figsize (tuple): Figure size (default (12, 5)).
        annot (bool): Display numeric annotations (default True).
        fmt (str): String format for annotations (default '.0f').
        orientation (str): Orientation of colorbar ('vertical' or 'horizontal').
        title (str): Custom plot title (auto-generated if None).
        xlabel (str): X-axis label (auto-generated if None).
        ylabel (str): Y-axis label (auto-generated if None).
        cbar_label (str): Colorbar label (auto-generated if None).

    RETURNS:
        None (Displays a heatmap)
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # --- Auto-detect aggregation function ---
    if aggfunc is None:
        if value_col is None:
            aggfunc = "count"
        elif pd.api.types.is_numeric_dtype(df[value_col]):
            aggfunc = "mean"
        else:
            aggfunc = "count"

    # --- Smart labeling ---
    if title is None:
        title = f"Heatmap of {value_col or 'Counts'} by {index_col} and {column_col}"
    if xlabel is None:
        xlabel = column_col.replace("_", " ").title()
    if ylabel is None:
        ylabel = index_col.replace("_", " ").title()
    if cbar_label is None:
        cbar_label = f"{aggfunc.title()} of {value_col or 'Entries'}"

    # --- Pivot table creation ---
    if value_col is None:
        pivot_table = df.groupby([index_col, column_col]).size().unstack(fill_value=0)
    else:
        pivot_table = (
            df.groupby([index_col, column_col])[value_col]
            .agg(aggfunc)
            .unstack(fill_value=0)
        )

    # --- Month or day ordering (optional heuristic) ---
    import calendar
    if column_col.lower().startswith("month"):
        month_order = list(calendar.month_abbr)[1:]  # ['Jan', ..., 'Dec']
        # Handle if numeric months
        if pivot_table.columns.dtype.kind in "iufc":
            month_order = range(1, 13)
        pivot_table = pivot_table[[col for col in month_order if col in pivot_table.columns]]

    # --- Plotting ---
    plt.figure(figsize=figsize)
    sns.heatmap(
        pivot_table,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"orientation": orientation, "label": cbar_label},
    )

    plt.title(title, fontsize=16, pad=12, weight="bold")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    #     print(f"✅ Bar chart race saved as: {filename}")
def generate_line_chart( #Daksh
    s,
    *,
    title="Content Added Over Time",
    xlabel="Month",
    ylabel="Number of Titles Added",
    figsize=(10, 5),
    figtitle,
    marker=".",
    color="red", 
    markerfacecolor="black",
    markeredgecolor="black",
    grid_alpha=0.3,
):
    """
    Plot a time-series of content volume.
    Expects a pandas Series with a datetime-like index (e.g. year_month) and counts as values.
    Inputs:-
    1)s : pd.Series
        Series with datetime-like index and counts as values.
    2)title : string
        Title of the line chart.
    3)xlabel : string
        Label for the x-axis.
    4)ylabel : string
        Label for the y-axis. 
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import seaborn as sns
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        s.index, s.values, marker=marker,color=color,markerfacecolor=markerfacecolor,
        markeredgecolor=markeredgecolor
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=grid_alpha)
    plt.figtext(0.5, 0, figtitle, 
            wrap=True, horizontalalignment='center', fontsize=12)
    fig.tight_layout()
    plt.show()

def plot_treemap_from_series(
    s,
    *,
    # title="Top 15 Genres / Categories",
    caption="Figure 1: Genre Treemap (Netflix-style)",
    figsize=(10, 6),
):
    """
    Plot a treemap from a pandas Series of counts.
    s.index -> labels
    s.values -> sizes
    """
    import matplotlib.pyplot as plt
    import squarify
    # netflix-ish palette (light → dark)
    colors = ["#E50914", "#B20710", "#831010", "#6B0F0F", "#4A0C0C"] * 5

    labels = [f"{lbl}\n{s[lbl]}" for lbl in s.index]
    sizes = s.values

    fig, ax = plt.subplots(figsize=figsize)
    # black background for Netflix vibe
    ax.set_facecolor("#141414")
    fig.patch.set_facecolor("#141414")

    squarify.plot(
        sizes=sizes,
        label=labels,
        color=colors[:len(sizes)],
        alpha=1.0,
        text_kwargs={"color": "white", "fontsize": 10}
    )
    # ax.set_title(title, color="white", fontsize=14, pad=12)
    ax.axis("off")

    # make room at bottom for caption
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # caption at bottom, centered
    fig.text(
        0.5, 0.01,
        caption,
        ha="center",
        va="bottom",
        color="white",
        fontsize=9,
    )

    plt.show()
def chi_square_test(df, col1, col2):#Taniya
    """
    Performs a Chi-Square Test of Independence between two categorical variables.


    INPUTS:
        df (pandas.DataFrame) -> Dataset containing both categorical columns.
        col1 (str) -> First categorical variable (e.g., 'director').
        col2 (str) -> Second categorical variable (e.g., 'rating').


    RETURNS:
        chi2 (float) -> Chi-square statistic
        p (float) -> p-value
        dof (int) -> Degrees of freedom
        contingency (pd.DataFrame) -> Contingency table used in the test


    PURPOSE:
        To determine whether there is a statistically significant relationship 
        between two categorical variables. Highly modular — usable for 
        director-rating, actor-category, country-type, etc.
    """
    import pandas as pd
    from scipy.stats import chi2_contingency


    # Drop missing or unknown entries for the two columns
    data = df.dropna(subset=[col1, col2])
    if data[col1].dtype == 'object':
        data = data[data[col1].str.lower() != 'unknown']
    if data[col2].dtype == 'object':
        data = data[data[col2].str.lower() != 'unknown']


    # Build contingency table
    contingency = pd.crosstab(data[col1], data[col2])


    # Perform Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency)


    return chi2, p, dof, contingency
def plot_chi_square_heatmap(contingency, var1_name="Variable 1", var2_name="Variable 2", top_n=10):  # Taniya
    """
    Plots a heatmap for the top N categories from a Chi-Square contingency table.


    INPUTS:
        contingency (pd.DataFrame) -> Contingency table (output from chi_square_test)
        var1_name (str) -> Label for the first variable
        var2_name (str) -> Label for the second variable
        top_n (int) -> Number of top categories (rows) to visualize


    RETURNS:
        None (displays heatmap)
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib as mpl


    # Take top N categories by row totals for clarity
    top_rows = contingency.sum(axis=1).sort_values(ascending=False).head(top_n).index
    subset = contingency.loc[top_rows]


    # Create Netflix-style red gradient colormap
    netflix_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "netflix_red",
        ["#221F1F", "#8B0000", "#E50914"]
    )


    plt.figure(figsize=(10, 6))
    sns.heatmap(
        subset,
        cmap=netflix_cmap,
        annot=True,
        fmt='d',
        linewidths=0.5,
        cbar_kws={"label": "Frequency"},
        annot_kws={"color": "white", "fontsize": 10}
    )


    # Netflix-inspired styling
    plt.title(f"{var1_name} vs {var2_name} Distribution (Top {top_n})",
              fontsize=15, color="#E50914", fontweight="bold", pad=15)
    plt.xlabel(var2_name, fontsize=12, color="white")
    plt.ylabel(var1_name, fontsize=12, color="white")


    plt.gca().set_facecolor("#141414")
    plt.gcf().patch.set_facecolor("#141414")
    plt.xticks(color="white", rotation=45, ha="right")
    plt.yticks(color="white")


    plt.tight_layout()
    plt.show()


def anova_test(df, group_col, value_col):#Taniya
    # """
    # Performs a one-way ANOVA test to determine whether the mean of a continuous 
    # variable differs significantly across groups.


    # INPUTS:
    #     df (pandas.DataFrame) -> Dataset containing categorical and continuous columns
    #     group_col (str) -> Column name for the categorical variable (e.g., 'director')
    #     value_col (str) -> Column name for the continuous variable (e.g., 'duration')


    # RETURNS:
    #     F-statistic (float), p-value (float)


    # PURPOSE:
    #     Tests whether the average of a numeric column (e.g., duration) 
    #     differs significantly across categories (e.g., directors).
    # """
    import pandas as pd
    from scipy.stats import f_oneway


    # Drop missing values
    data = df.dropna(subset=[group_col, value_col])


    # Convert duration to numeric if it’s a string (e.g., '90 min', '2 Seasons')
    if data[value_col].dtype == 'object':
        data[value_col] = (
            data[value_col]
            .astype(str)
            .str.extract(r'(\d+)')  # extract the numeric part
            .astype(float)
        )


    # Prepare samples grouped by the categorical variable
    groups = [
        group[value_col].dropna().values
        for _, group in data.groupby(group_col)
        if len(group[value_col].dropna()) > 1
    ]


    # Perform one-way ANOVA
    if len(groups) > 1:
        F_stat, p_val = f_oneway(*groups)
        return F_stat, p_val
    else:
        print("Not enough groups for ANOVA.")
        return None, None





def compute_network_centrality(G):#Taniya
    # """
    # Computes key centrality measures for a NetworkX graph.


    # INPUTS:
    #     G (networkx.Graph) -> Collaboration network (e.g., Director–Actor)


    # RETURNS:
    #     pd.DataFrame -> DataFrame containing:
    #                     [node, degree_centrality, betweenness_centrality, 
    #                      closeness_centrality, eigenvector_centrality]


    # PURPOSE:
    #     Quantifies the most connected and influential creators in the network.
    # """
    import networkx as nx
    import pandas as pd


    # Compute all major centrality measures
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, normalized=True)
    closeness_centrality = nx.closeness_centrality(G)
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=500)
    except nx.PowerIterationFailedConvergence:
        eigenvector_centrality = {n: 0 for n in G.nodes()}


    # Combine results
    centrality_df = pd.DataFrame({
        'node': list(G.nodes()),
        'degree_centrality': [degree_centrality[n] for n in G.nodes()],
        'betweenness_centrality': [betweenness_centrality[n] for n in G.nodes()],
        'closeness_centrality': [closeness_centrality[n] for n in G.nodes()],
        'eigenvector_centrality': [eigenvector_centrality[n] for n in G.nodes()],
    })


    return centrality_df.sort_values(by='degree_centrality', ascending=False)
def plot_top_central_nodes(centrality_df, metric='degree_centrality', top_n=10):#Taniya
    # """
    # Plots the top N most central nodes by a chosen centrality metric.
    # """
    import seaborn as sns
    import matplotlib.pyplot as plt


    top_nodes = centrality_df.sort_values(by=metric, ascending=False).head(top_n)


    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_nodes, x=metric, y='node', color='skyblue')
    plt.title(f"Top {top_n} Nodes by {metric.replace('_', ' ').title()}")
    plt.xlabel(metric.replace('_', ' ').title())
    plt.ylabel("Node (Creator)")
    plt.tight_layout()
    plt.show()


def dataset_vanity_checks(df):#Taniya-may remove depending on whether the cleaning needs vanity checks
    print("Total rows:", len(df))
    print("Missing directors:", (df['director'] == 'Unknown').sum())
    print("Missing cast:", (df['cast'] == 'Unknown').sum())
    print("Unique directors:", df['director'].nunique())
    print("Unique actors:", df['cast'].nunique())
    print("Unique genres:", df['listed_in'].nunique())
    print("Year range:", df['release_year'].min(), "-", df['release_year'].max())
def get_top_creators(df, column, n=20):#Taniya
    
    # INPUTS:
    #     df (pandas.DataFrame)  -> The cleaned Netflix dataset.
    #     column (str)           -> Column name ('director' or 'cast').
    #     n (int, optional)      -> Number of top creators to return (default = 20).


    # RETURNS:
    #     pandas.Series -> Top 'n' creators and their respective counts.


    # PURPOSE:
    #     Identifies which creators (directors or actors) appear most frequently
    #     in the catalog, excluding 'Unknown'.
    
    subset = df[df[column].str.lower() != 'unknown']
    top_creators = subset[column].value_counts().head(n)
    return top_creators


def plot_top_creators(series, title):  # Taniya
    """
    Visualizes top creators by count using a horizontal bar chart
    with a Netflix-themed red gradient.


    INPUTS:
        series (pandas.Series) -> Index = creator names; values = counts.
        title (str)            -> Chart title.


    RETURNS:
        None (displays the bar plot)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib as mpl


    # Netflix red gradient colormap (dark → bright red)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "netflix_red", ["#8B0000", "#E50914"]
    )


    # Normalize bar values to the colormap range
    norm = mpl.colors.Normalize(vmin=min(series.values), vmax=max(series.values))
    colors = cmap(norm(series.values))


    # Plot bars
    plt.figure(figsize=(10, 6))
    bars = plt.barh(series.index, series.values, color=colors)


    # Title & labels
    plt.title(title, fontsize=15, color="#E50914", fontweight="bold")
    plt.xlabel("Number of Titles", color="white", fontsize=12)
    plt.ylabel("", color="white")
    plt.gca().invert_yaxis()  # Highest value at top
    plt.grid(axis='x', linestyle='--', alpha=0.3, color='gray')


    # Netflix dark background
    plt.gca().set_facecolor("#141414")
    plt.gcf().patch.set_facecolor("#141414")


    # Make axis tick labels white
    plt.xticks(color="white", fontsize=10)
    plt.yticks(color="white", fontsize=10)


    plt.tight_layout()
    plt.show()

def build_collaboration_network(df):#Taniya
    # """
    # INPUTS:
    #     df (pandas.DataFrame) -> Dataset containing 'director' and 'cast' columns.


    # RETURNS:
    #     networkx.Graph -> Undirected bipartite graph of directors and actors.


    # PURPOSE:
    #     Constructs a collaboration network showing which directors have worked
    #     with which actors, excluding 'Unknown' entries.
    # """
    import networkx as nx
    import pandas as pd


    valid_df = df[(df['director'] != 'Unknown') & (df['cast'] != 'Unknown')]
    G = nx.from_pandas_edgelist(valid_df, source='director', target='cast')
    return G


def plot_network(G, max_nodes=150):  # Taniya
    # """
    # INPUTS:
    #     G (networkx.Graph) -> Collaboration network from build_collaboration_network().
    #     max_nodes (int)    -> Maximum nodes to show (default = 150).


    # RETURNS:
    #     None (displays Netflix-themed collaboration visualization with visible edges).


    # PURPOSE:
    #     Visualizes the director–actor network with Netflix-inspired aesthetics:
    #     - Bright red edges for visibility
    #     - Red directors, grey-white actors
    #     - Black cinematic background
    #     - Node sizes scaled by degree
    #     - Top collaborators labeled in bright red
    # """
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np


    # Limit graph size for readability
    if len(G.nodes) > max_nodes:
        G = G.subgraph(list(G.nodes)[:max_nodes])


    plt.figure(figsize=(14, 10))
    plt.style.use("dark_background")


    # Identify directors and actors
    directors = [n for n in G.nodes if ' ' in n and len(n.split()) <= 3]
    actors = list(set(G.nodes) - set(directors))


    # Netflix color palette
    director_color = "#E50914"  # Netflix red
    actor_color = "#B3B3B3"     # Muted white-grey
    edge_color = "#E50914"      # Bright red edges


    # Colors and node sizes
    degrees = dict(G.degree)
    node_colors = [director_color if n in directors else actor_color for n in G.nodes]
    node_sizes = [120 + 4 * degrees[n] for n in G.nodes]


    # Layout
    pos = nx.spring_layout(G, k=0.3, iterations=40, seed=42)


    # Draw edges — brighter and more visible now
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, alpha=0.6, width=1.3)


    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, linewidths=0.5)


    # Label top 10 most connected nodes
    top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:10]
    label_nodes = [n for n, _ in top_nodes if n in pos]
    labels = {n: n for n in label_nodes}


    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color="#FF4C4C", font_weight="bold")


    # Style the plot
    plt.title("🎬 Director–Actor Collaboration Network",
              fontsize=15, color="#E50914", fontweight="bold", pad=15)
    plt.gca().set_facecolor("#000000")  # pure black
    plt.gcf().patch.set_facecolor("#000000")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_heatmap(matrix, row_label="Row", col_label="Column", top_rows=15, title=None):  # Taniya
    """
    Plots a Netflix-themed heatmap for any 2D pivot table (entity–category relationships).


    INPUTS:
        matrix (pd.DataFrame) -> Pivot table or cross-tab (rows = entities, columns = categories)
        row_label (str) -> Label for rows (e.g., 'Director', 'Actor', 'Country')
        col_label (str) -> Label for columns (e.g., 'Genre', 'Category', 'Rating')
        top_rows (int) -> Number of top rows (entities) to display (default = 15)
        title (str or None) -> Custom title; if None, auto-generates one


    RETURNS:
        None (displays the heatmap)
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import pandas as pd


    # Validate input
    if not isinstance(matrix, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame pivot table.")


    # Select top entities (rows)
    top_entities = matrix.sum(axis=1).sort_values(ascending=False).head(top_rows).index
    subset = matrix.loc[top_entities]


    # Sort columns by overall frequency
    subset = subset[subset.sum().sort_values(ascending=False).index]


    # Generate title if not provided
    if title is None:
        title = f"{row_label}–{col_label} Relationship Heatmap (Top {top_rows})"


    # Netflix-style color map
    netflix_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "netflix_red",
        ["#221F1F", "#8B0000", "#E50914"]
    )


    # Plot
    plt.figure(figsize=(14, 7))
    sns.heatmap(
        subset,
        cmap=netflix_cmap,
        linewidths=0.4,
        linecolor="#2a2a2a",
        cbar_kws={'label': 'Count'},
        annot=False
    )


    # Dark Netflix background
    plt.gca().set_facecolor("#141414")
    plt.gcf().patch.set_facecolor("#141414")


    # Titles and labels
    plt.title(title, fontsize=16, color="#E50914", fontweight="bold", pad=15)
    plt.xlabel(col_label, fontsize=12, color="white")
    plt.ylabel(row_label, fontsize=12, color="white")


    # Axis styling
    plt.xticks(rotation=45, ha="right", color="white", fontsize=10)
    plt.yticks(color="white", fontsize=10)


    # Colorbar styling
    cbar = plt.gca().collections[0].colorbar
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.tick_params(colors="white")


    plt.tight_layout()
    plt.show()


def director_genre_matrix(df, min_titles=3):#Taniya
    """
    INPUTS:
        df (pandas.DataFrame) -> Dataset with 'director' and 'listed_in' columns.
        min_titles (int)      -> Minimum number of titles per director to include.


    RETURNS:
        pandas.DataFrame -> Pivot table (directors × genres, counts of titles).


    PURPOSE:
        Builds a cross-tab of how many titles each director has in each genre,
        useful for specialization or heatmap visualization.
    """
    import pandas as pd


    pivot = (df[df['director'] != 'Unknown']
             .groupby(['director', 'listed_in'])
             .size()
             .unstack(fill_value=0))
    pivot = pivot[pivot.sum(axis=1) >= min_titles]
    return pivot


def plot_creator_country_distribution(df, creator_col='director'):  # Taniya
    """
    INPUTS:
        df (pandas.DataFrame)  -> Dataset with 'country' and creator column.
        creator_col (str)      -> Column to analyze ('director' or 'cast').


    RETURNS:
        None (displays Netflix-themed bar chart).


    PURPOSE:
        Shows top countries by count of unique creators to study
        international vs domestic talent distribution with a Netflix-style aesthetic.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib as mpl


    # Prepare data
    country_counts = (
        df[['country', creator_col]]
        .drop_duplicates()
        .country.value_counts()
        .head(15)
    )


    # Netflix red gradient colormap
    netflix_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "netflix_red", ["#8B0000", "#E50914"]
    )
    colors = [netflix_cmap(i / (len(country_counts) - 1)) for i in range(len(country_counts))]


    # Plot
    plt.figure(figsize=(8, 6))
    plt.style.use("dark_background")
    sns.barplot(
        x=country_counts.values,
        y=country_counts.index,
        palette=colors
    )


    # Netflix-styled title & labels
    plt.title("🌍 Top 15 Countries by Creator Count", fontsize=15, color="#E50914", fontweight="bold", pad=15)
    plt.xlabel("Number of Creators", fontsize=12, color="white")
    plt.ylabel("", color="white")


    # Backgrounds
    plt.gca().set_facecolor("#000000")
    plt.gcf().patch.set_facecolor("#000000")


    # Ticks and spines
    plt.xticks(color="white")
    plt.yticks(color="white")
    for spine in plt.gca().spines.values():
        spine.set_visible(False)


    plt.tight_layout()
    plt.show()


def director_rating_significance(df, val_col='vote_average'):  # Taniya
    import pandas as pd
    import numpy as np
    from itertools import combinations
    from scipy.stats import f, t


    # Drop missing values for safety
    df = df.dropna(subset=['director', val_col])


    # Group by director and compute mean and count
    groups = df.groupby('director')[val_col].apply(list)
    k = len(groups)  # Number of directors
    n_total = len(df)  # Total observations
    overall_mean = df[val_col].mean()


    # ---- F-test (One-way ANOVA) ----
    # Between-group sum of squares (SSB)
    ssb = sum([len(vals) * (np.mean(vals) - overall_mean) ** 2 for vals in groups])


    # Within-group sum of squares (SSW)
    ssw = sum([sum((np.array(vals) - np.mean(vals)) ** 2) for vals in groups])


    dfb = k - 1
    dfw = n_total - k
    msb = ssb / dfb
    msw = ssw / dfw
    f_stat = msb / msw
    f_crit = f.ppf(0.95, dfb, dfw)  # α = 0.05


    print(f"F-statistic = {f_stat:.3f}, F-critical = {f_crit:.3f}")
    if f_stat <= f_crit:
        print("Fail to reject H₀ → No significant difference between directors.")
        return None


    print("Reject H₀ → Significant difference detected. Proceeding with LSD test...")


    # ---- LSD (Least Significant Difference) ----
    result = {}
    means = groups.apply(np.mean)
    sizes = groups.apply(len)
    se = np.sqrt(msw * (1/sizes.values[:, None] + 1/sizes.values))


    # t-critical value (two-tailed, α=0.05)
    t_crit = t.ppf(1 - 0.05/2, dfw)
    lsd_results = []


    for (d1, d2) in combinations(groups.index, 2):
        diff = abs(means[d1] - means[d2])
        se_pair = np.sqrt(msw * (1/sizes[d1] + 1/sizes[d2]))
        lsd = t_crit * se_pair
        significant = diff > lsd
        result[(d1, d2)] = {
            'Mean_Diff': diff,
            'LSD': lsd,
            'Significant': significant
        }
        lsd_results.append((d1, d2, diff, lsd, significant))


    # Print summary of significant differences
    sig_pairs = [pair for pair, vals in result.items() if vals['Significant']]
    if sig_pairs:
        print("\nSignificant director pairs (mean difference > LSD):")
        for pair in sig_pairs:
            print(f"  {pair[0]} vs {pair[1]} → Δ={result[pair]['Mean_Diff']:.3f}")
    else:
        print("\nNo significant pairwise differences found in LSD test.")


    return result

def plot_international_vs_domestic(df, creator_col='director', home_country='India'):  # Taniya
    """
    INPUTS:
        df (pandas.DataFrame)  -> Dataset with 'country' and creator column.
        creator_col (str)      -> Column to analyze ('director' or 'cast').
        home_country (str)     -> Country considered "domestic" (default = 'United States').


    RETURNS:
        pandas.DataFrame -> Summary table of domestic vs international creator counts.


    PURPOSE:
        Compares how many unique creators are domestic vs international,
        using a Netflix-themed red–black visualization to highlight global talent diversity.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd


    # Drop unknowns
    df = df[df['country'].str.lower() != 'unknown']


    # Classify creators
    df['talent_origin'] = df['country'].apply(
        lambda x: 'Domestic' if home_country.lower() in x.lower() else 'International'
    )


    # Count unique creators in each group
    unique_creators = df[[creator_col, 'talent_origin']].drop_duplicates()
    summary = unique_creators['talent_origin'].value_counts().reset_index()
    summary.columns = ['Talent Type', 'Creator Count']


    # Define Netflix-style colors
    palette = {
        'Domestic': '#E50914',       # Netflix red
        'International': '#B3B3B3'   # Muted grey-white
    }


    # Plot
    plt.figure(figsize=(6, 5))
    plt.style.use("dark_background")
    sns.barplot(
        data=summary,
        x='Talent Type',
        y='Creator Count',
        palette=palette
    )


    # Title & styling
    plt.title(
        f"🌍 International vs Domestic {creator_col.capitalize()}s ({home_country})",
        fontsize=15,
        color="#E50914",
        fontweight="bold",
        pad=15
    )
    plt.xlabel("")
    plt.ylabel("Number of Unique Creators", fontsize=11, color="white")


    # Backgrounds
    plt.gca().set_facecolor("#000000")
    plt.gcf().patch.set_facecolor("#000000")


    # Text & ticks
    plt.xticks(color="white", fontsize=11)
    plt.yticks(color="white")


    # Remove borders
    for spine in plt.gca().spines.values():
        spine.set_visible(False)


    plt.tight_layout()
    plt.show()


    return summary
def plot_cast_frequency_distribution(df):#Taniya
    
    # INPUTS:
    #     df (pandas.DataFrame) -> Dataset with 'cast' column.


    # RETURNS:
    #     None (displays histogram and optional bubble chart).


    # PURPOSE:
    #     Visualizes how frequently actors appear across Netflix titles.
    #     Helps understand whether a few actors dominate the catalog
    #     or if the presence is evenly distributed.
    
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt


    # Remove 'Unknown' and count frequency
    cast_counts = df[df['cast'].str.lower() != 'unknown']['cast'].value_counts()


    # Optional: Bubble chart of top actors
    top_cast = cast_counts.head(30).reset_index()
    top_cast.columns = ['Actor', 'Appearances']


    plt.figure(figsize=(10,7))
    sns.scatterplot(
        data=top_cast,
        x='Appearances',
        y='Actor',
        size='Appearances',
        sizes=(100, 1000),
        alpha=0.7,
        color='#E50914'
    )
    plt.title("Top 30 Actor Appearance Bubble Chart", fontsize=14)
    plt.xlabel("Number of Titles Appeared In")
    plt.ylabel("Actor")
    plt.tight_layout()
    plt.show()


def plot_creator_timeline(df, creator_col='director', top_n=5):  # Taniya
    
    # INPUTS:
    #     df (pandas.DataFrame) -> Dataset containing creator names and release years
    #     creator_col (str) -> Column to analyze ('director' or 'cast')
    #     top_n (int) -> Number of top creators to include in the timeline


    # RETURNS:
    #     None (displays lineplot)


    # PURPOSE:
    #     Visualizes how the content output of top creators changes over time.
    #     Helps identify career trends and Netflix’s collaborations over years.
    
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt


    # Drop missing and unknown values
    data = df[df[creator_col].str.lower() != 'unknown']
    data = data.dropna(subset=[creator_col, 'release_year'])


    # Select top N creators
    top_creators = data[creator_col].value_counts().head(top_n).index


    # Filter dataset for only these creators
    data_top = data[data[creator_col].isin(top_creators)]


    # Group by year and creator
    timeline = (
        data_top.groupby(['release_year', creator_col])
        .size()
        .reset_index(name='count')
    )


    plt.figure(figsize=(10,6))
    sns.lineplot(
        data=timeline,
        x='release_year',
        y='count',
        hue=creator_col,
        marker='o',
        palette='tab10'
    )


    plt.title(f"Yearly Count of Works by Top {top_n} {creator_col.title()}s", fontsize=14)
    plt.xlabel("Release Year")
    plt.ylabel("Number of Titles")


    # ✅ Legend INSIDE the plot (top-left corner)
    plt.legend(
        title=creator_col.title(),
        loc='upper left',
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        facecolor='black',
        framealpha=0.9,
        fontsize=9,
        title_fontsize=10
    )


    plt.tight_layout()
    plt.show()


def compute_entropy(df, entity_col, category_col):#Taniya
    # """
    # Computes Shannon Entropy for any categorical pair (e.g., Director–Genre, Actor–Genre, Country–Category).


    # INPUTS:
    #     df (pandas.DataFrame) -> dataset containing both categorical columns
    #     entity_col (str) -> column name representing the entity (e.g., 'director', 'actor', 'country')
    #     category_col (str) -> column name representing the category (e.g., 'listed_in', 'genre', 'category')


    # RETURNS:
    #     pd.DataFrame -> DataFrame with columns:
    #                     [entity_col, 'entropy', 'num_records']


    # PURPOSE:
    #     Quantifies specialization or diversity for each entity.
    #     - Low entropy → specialized in fewer categories
    #     - High entropy → diversified across many categories
    # """
    import pandas as pd
    import numpy as np


    # Drop missing or unknown values
    data = df.dropna(subset=[entity_col, category_col])
    if data[entity_col].dtype == 'object':
        data = data[data[entity_col].str.lower() != 'unknown']
    if data[category_col].dtype == 'object':
        data = data[data[category_col].str.lower() != 'unknown']


    # Split comma-separated category entries if present
    #data = data.assign(*{category_col: data[category_col].astype(str).str.split(',\s')})
    data = data.explode(category_col)


    # Compute counts
    combo_counts = data.groupby([entity_col, category_col]).size().reset_index(name='count')
    total_counts = combo_counts.groupby(entity_col)['count'].sum().reset_index(name='total')


    # Merge to calculate probabilities
    merged = combo_counts.merge(total_counts, on=entity_col)
    merged['p'] = merged['count'] / merged['total']


    # Compute entropy
    entropy_df = (
        merged.groupby(entity_col)
        .apply(lambda x: -np.sum(x['p'] * np.log2(x['p'])))
        .reset_index(name='entropy')
    )


    # Add total number of records (for filtering)
    num_records = data.groupby(entity_col).size().reset_index(name='num_records')
    entropy_df = entropy_df.merge(num_records, on=entity_col, how='left')


    return entropy_df.sort_values(by='entropy', ascending=False)


def plot_entropy(entropy_df, entity_col, top_n=10):  # Taniya
    
    # Plots top and bottom N entities based on entropy (diversity) 
    # in a Netflix-themed color scheme.


    # INPUTS:
    #     entropy_df (pd.DataFrame) -> DataFrame returned from compute_entropy()
    #     entity_col (str) -> entity column name (e.g., 'director', 'actor')
    #     top_n (int) -> number of top and bottom entities to visualize


    # RETURNS:
    #     None (displays two Netflix-themed bar charts)
   
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib as mpl


    # Prepare data
    top_diverse = entropy_df.head(top_n)
    top_specialized = entropy_df.tail(top_n)


    # Netflix gradients
    red_gradient = mpl.colors.LinearSegmentedColormap.from_list(
        "netflix_red", ["#8B0000", "#E50914"]
    )
    dark_gradient = mpl.colors.LinearSegmentedColormap.from_list(
        "netflix_darkred", ["#333333", "#8B0000"]
    )


    # --- Plot 1: Top Diverse Entities (High Entropy) ---
    plt.figure(figsize=(10, 6))
    plt.style.use("dark_background")
    sns.barplot(
        data=top_diverse,
        x='entropy',
        y=entity_col,
        palette=[red_gradient(i / top_n) for i in range(top_n)]
    )
    plt.title(f"🔥 Top {top_n} Most Diverse {entity_col.title()}s (High Entropy)",
              fontsize=15, color="#E50914", fontweight="bold", pad=15)
    plt.xlabel("Entropy (Diversity)", fontsize=12, color="white")
    plt.ylabel(entity_col.title(), fontsize=12, color="white")


    # Style adjustments
    plt.gca().set_facecolor("#000000")
    plt.gcf().patch.set_facecolor("#000000")
    plt.xticks(color="white")
    plt.yticks(color="white")
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    plt.show()

def generate_styled_boxplot(df, cat, val, outlier=True): # Taniya
   
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


    # Netflix red color
    netflix_red = '#E50914'


    sns.boxplot(
        data=df,
        x=cat,
        y=val,
        order=genre_medians.index,   # sort by median
        showfliers=outlier,
        boxprops=dict(color=netflix_red),
        whiskerprops=dict(color=netflix_red),
        capprops=dict(color=netflix_red),
        medianprops=dict(color=netflix_red),
        flierprops=dict(markerfacecolor=netflix_red, markeredgecolor=netflix_red),
    )


    # Styling text in white
    plt.xlabel("", color="white")
    plt.xticks([], color="white")
    plt.ylabel(val, fontsize=14, color="white")
    plt.title("Boxplot of "+val+" by "+cat+" (Sorted by Median)", fontsize=16, color="white")
    plt.yticks(fontsize=12, color="white")


    plt.show()

def plot_top_countries_by_shows(df, top_n): # Aditya
    """
    Plots the top N countries by the number of unique shows.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing 'country' and 'show_id' columns.
        top_n (int): Number of top countries to display
    """
    # Count unique show_id for each country
    import matplotlib.pyplot as plt
    country_counts = (
        df.groupby('country')['show_id']
        .nunique()
        .sort_values(ascending=False)
    )

    # Plot top N countries
    plt.figure(figsize=(12, 6))
    country_counts.head(top_n).plot(kind='bar', color='skyblue')

    plt.title(f"Top {top_n} Countries by Number of Unique Shows", fontsize=14)
    plt.xlabel("Country")
    plt.ylabel("Number of Unique Shows")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_category_frequency_per_country(df, top_n): # Aditya
    import seaborn as sns
    import matplotlib.pyplot as plt
    """
    Plots the frequency of each category per country using unique show IDs.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing 'country', 'category', and 'show_id' columns.
        top_n (int): Number of top countries (by number of unique shows) to include in the plot.
    """
    # Count frequency of each category per country using unique show IDs
    country_category_counts = (
        df.groupby(['country', 'category'])['show_id']
          .nunique()
          .reset_index(name='count')
    )

    # Focus on top N countries with most shows
    top_countries = (
        df.groupby('country')['show_id']
          .nunique()
          .sort_values(ascending=False)
          .head(top_n)
          .index
    )

    filtered = country_category_counts[country_category_counts['country'].isin(top_countries)]

    # Sort countries by total count (for better visual order)
    filtered['country'] = pd.Categorical(filtered['country'], categories=top_countries, ordered=True)

    # Plot
    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=filtered,
        x='country',
        y='count',
        hue='category',
        palette='tab10'
    )

    plt.title("Category Frequency per Country", fontsize=14)
    plt.xlabel("Country")
    plt.ylabel("Number of Unique Shows")
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_top_countries_by_type(df, top_n): # Aditya
    """
    Plots the top N countries by the number of Movies and TV Shows.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing at least the columns: 'country', 'type', 'show_id'.
    top_n : int, optional
    """
    import matplotlib.pyplot as plt
    # --- Group by country and type ---
    country_type_counts = (
        df.groupby(['country', 'type'])['show_id']
          .nunique()
          .reset_index(name='count')
    )

    # --- Separate for Movies and TV Shows ---
    top_movies = (
        country_type_counts[country_type_counts['type'].str.lower() == 'movie']
        .sort_values('count', ascending=False)
        .head(top_n)
    )

    top_tvshows = (
        country_type_counts[country_type_counts['type'].str.lower() == 'tv show']
        .sort_values('count', ascending=False)
        .head(top_n)
    )

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Movies Plot
    axes[0].bar(top_movies['country'], top_movies['count'], color='skyblue')
    axes[0].set_title(f"Top {top_n} Countries by Number of Movies", fontsize=14)
    axes[0].set_xlabel("Country")
    axes[0].set_ylabel("Number of Unique Movies")
    axes[0].tick_params(axis='x', rotation=90)

    # TV Shows Plot
    axes[1].bar(top_tvshows['country'], top_tvshows['count'], color='lightgreen')
    axes[1].set_title(f"Top {top_n} Countries by Number of TV Shows", fontsize=14)
    axes[1].set_xlabel("Country")
    axes[1].tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.show()


    

def plot_category_frequency_by_country(df, top_n): #Aditya
    """
    Plots category frequency per country (Movies and TV Shows separately)
    using a Netflix-style dark theme.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing at least 'country', 'type', 'category', and 'show_id'.
    top_n : int, optional
        Number of top countries to display.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    # --- Validate columns ---
    required_cols = {'country', 'type', 'category', 'show_id'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    # --- Clean data ---
    df = df.dropna(subset=['country', 'type', 'category', 'show_id'])

    # --- Group by country, type, and category ---
    country_category_counts = (
        df.groupby(['country', 'type', 'category'])['show_id']
          .nunique()
          .reset_index(name='count')
    )

    # --- Select top N countries by total show count ---
    top_countries = (
        df.groupby('country')['show_id']
          .nunique()
          .sort_values(ascending=False)
          .head(top_n)
          .index
    )

    filtered = country_category_counts[country_category_counts['country'].isin(top_countries)]

    # --- Split by type ---
    movies = filtered[filtered['type'].str.lower() == 'movie']
    tvshows = filtered[filtered['type'].str.lower() == 'tv show']

    # --- Netflix-style color palette ---
    netflix_palette = ['#E50914', "#970000", '#b81d24', '#f5f5f1', '#737373']

    # --- Apply dark theme ---
    sns.set_theme(style="darkgrid", rc={'axes.facecolor': "#FFFDFD", 'figure.facecolor': '#141414'})
    
    # --- Create Subplots ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 14), sharex=True)

    # --- Movies Plot ---
    sns.barplot(
        data=movies,
        x='country',
        y='count',
        hue='category',
        palette=netflix_palette,
        ax=axes[0]
    )
    axes[0].set_title("🎬 Category Frequency per Country (Movies)", fontsize=14, color='white')
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Number of Unique Movies", color='white')
    axes[0].tick_params(axis='x', rotation=45, colors='white')
    axes[0].tick_params(axis='y', colors='white')
    axes[0].legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left', facecolor='#141414', labelcolor='white')

    # --- TV Shows Plot ---
    sns.barplot(
        data=tvshows,
        x='country',
        y='count',
        hue='category',
        palette=netflix_palette,
        ax=axes[1]
    )
    axes[1].set_title("📺 Category Frequency per Country (TV Shows)", fontsize=14, color='white')
    axes[1].set_xlabel("Country", color='white')
    axes[1].set_ylabel("Number of Unique TV Shows", color='white')
    axes[1].tick_params(axis='x', rotation=45, colors='white')
    axes[1].tick_params(axis='y', colors='white')
    axes[1].legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left', facecolor='#141414', labelcolor='white')

    # --- Final Touches ---
    plt.tight_layout()
    plt.show()

def plot_top_countries_by_type(df, top_n): # Aditya
    """
    Plots the top N countries by the number of Movies and TV Shows.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing at least the columns: 'country', 'type', 'show_id'.
    top_n : int, optional
    """

    # --- Group by country and type ---
    country_type_counts = (
        df.groupby(['country', 'type'])['show_id']
          .nunique()
          .reset_index(name='count')
    )

    # --- Separate for Movies and TV Shows ---
    top_movies = (
        country_type_counts[country_type_counts['type'].str.lower() == 'movie']
        .sort_values('count', ascending=False)
        .head(top_n)
    )

    top_tvshows = (
        country_type_counts[country_type_counts['type'].str.lower() == 'tv show']
        .sort_values('count', ascending=False)
        .head(top_n)
    )

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Movies Plot
    axes[0].bar(top_movies['country'], top_movies['count'], color='skyblue')
    axes[0].set_title(f"Top {top_n} Countries by Number of Movies", fontsize=14)
    axes[0].set_xlabel("Country")
    axes[0].set_ylabel("Number of Unique Movies")
    axes[0].tick_params(axis='x', rotation=90)

    # TV Shows Plot
    axes[1].bar(top_tvshows['country'], top_tvshows['count'], color='lightgreen')
    axes[1].set_title(f"Top {top_n} Countries by Number of TV Shows", fontsize=14)
    axes[1].set_xlabel("Country")
    axes[1].tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.show()

def plot_avg_movie_duration_by_country(df, top_n): # Aditya
    """
    Plots the average runtime of movies per country using a Netflix-inspired theme.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least ['country', 'type', 'duration'] columns.
        'duration' should contain strings like '90 min'.
    top_n : int, optional
        Number of top countries to show.
    """



   # --- Step 1: Filter for Movies only ---
    movies_df = df[df['type'].str.lower() == 'movie'].copy()

    # --- Step 2: Convert 'duration' (like '90 min') to numeric minutes ---
    movies_df['duration_minutes'] = (
        movies_df['duration']
        .astype(str)
        .apply(lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else None)
    )

    # Drop missing or invalid duration rows
    movies_df = movies_df.dropna(subset=['duration_minutes'])

    # --- Step 3: Ensure uniqueness (country + show_id) ---
    movies_df = movies_df.drop_duplicates(subset=['country', 'show_id'])

    # --- Step 4: Compute average duration per country ---
    avg_duration = (
        movies_df.groupby('country')['duration_minutes']
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    # --- Step 5: Take top N countries ---
    top_countries = avg_duration.head(top_n)

    # --- Step 6: Netflix-inspired palette ---
    netflix_palette = ['#E50914', '#b81d24', '#221f1f', '#737373', '#7c0f00']

    # --- Step 7: Plot ---
    sns.set_theme(style="whitegrid", rc={'axes.facecolor': 'white', 'figure.facecolor': 'white'})

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=top_countries,
        x='country',
        y='duration_minutes',
        palette=netflix_palette
    )

    # --- Step 8: Customize ---
    plt.title("Average Movie Duration by Country", fontsize=16, color='#E50914', weight='bold')
    plt.xlabel("Country", fontsize=12, color='black')
    plt.ylabel("Average Duration (minutes)", fontsize=12, color='black')
    plt.xticks(rotation=75, ha='right', color='black')
    plt.yticks(color='black')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


def plot_avg_seasons_by_country(df, top_n): #Aditya
    """
    Plots the average number of TV show seasons per country using a Netflix-inspired palette.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least ['show_id', 'country', 'type', 'duration'] columns.
        'duration' should contain strings like '1 Season' or '2 Seasons'.
    top_n : int, optional
        Number of top countries to display in the bar plot.
    """

    # --- Step 1: Keep unique shows only ---
    df_unique = df.drop_duplicates(subset=['show_id']).copy()

    # --- Step 2: Filter for TV Shows only ---
    tv_df = df_unique[df_unique['type'].str.lower() == 'tv show'].copy()

    # --- Step 3: Extract numeric season counts ---
    tv_df['seasons'] = (
        tv_df['duration']
        .astype(str)
        .apply(lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else None)
    )

    # Drop rows without valid season info or country
    tv_df = tv_df.dropna(subset=['seasons', 'country'])

    # --- Step 4: Compute average seasons per country ---
    avg_seasons = (
        tv_df.groupby('country')['seasons']
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    # --- Step 5: Select top N countries ---
    top_countries = avg_seasons.head(top_n)

    # --- Step 6: Netflix-inspired palette ---
    netflix_palette = ['#E50914', '#b81d24', '#221f1f', '#737373', "#7c0f00"]

    # --- Step 7: Plot setup ---
    sns.set_theme(style="whitegrid", rc={'axes.facecolor': 'white', 'figure.facecolor': 'white'})

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=top_countries,
        x='country',
        y='seasons',
        palette=netflix_palette
    )

    # --- Step 8: Styling ---
    plt.title("Average Number of TV Show Seasons by Country", fontsize=16, color='#E50914', weight='bold')
    plt.xlabel("Country", fontsize=12, color='black')
    plt.ylabel("Average Number of Seasons", fontsize=12, color='black')
    plt.xticks(rotation=90, ha='right', color='black')
    plt.yticks(color='black')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


def plot_avg_seasons_by_country(df, top_n): #Aditya
    """
    Plots the average number of TV show seasons per country using a Netflix-inspired palette.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least ['show_id', 'country', 'type', 'duration'] columns.
        'duration' should contain strings like '1 Season' or '2 Seasons'.
    top_n : int, optional
        Number of top countries to display in the bar plot.
    """

    # --- Step 1: Keep unique shows only ---
    df_unique = df.drop_duplicates(subset=['show_id']).copy()

    # --- Step 2: Filter for TV Shows only ---
    tv_df = df_unique[df_unique['type'].str.lower() == 'tv show'].copy()

    # --- Step 3: Extract numeric season counts ---
    tv_df['seasons'] = (
        tv_df['duration']
        .astype(str)
        .apply(lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else None)
    )

    # Drop rows without valid season info or country
    tv_df = tv_df.dropna(subset=['seasons', 'country'])

    # --- Step 4: Compute average seasons per country ---
    avg_seasons = (
        tv_df.groupby('country')['seasons']
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    # --- Step 5: Select top N countries ---
    top_countries = avg_seasons.head(top_n)

    # --- Step 6: Netflix-inspired palette ---
    netflix_palette = ['#E50914', '#b81d24', '#221f1f', '#737373', "#7c0f00"]

    # --- Step 7: Plot setup ---
    sns.set_theme(style="whitegrid", rc={'axes.facecolor': 'white', 'figure.facecolor': 'white'})

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=top_countries,
        x='country',
        y='seasons',
        palette=netflix_palette
    )

    # --- Step 8: Styling ---
    plt.title("Average Number of TV Show Seasons by Country", fontsize=16, color='#E50914', weight='bold')
    plt.xlabel("Country", fontsize=12, color='black')
    plt.ylabel("Average Number of Seasons", fontsize=12, color='black')
    plt.xticks(rotation=90, ha='right', color='black')
    plt.yticks(color='black')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


def plot_movie_coproduction_heatmap(df, top_n): # Aditya
    """
    Plots a Netflix-themed heatmap showing how many unique movies 
    are shared (co-produced) between pairs of countries.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ['show_id', 'type', 'country'] columns.
        Each (show_id, country) pair represents one country’s involvement in a movie.
    top_n : int
        Number of top countries (by unique movie count) to display.
    """

    # --- Step 1: Filter only Movies ---
    movies_df = df[df['type'].str.lower() == 'movie'].copy()

    # --- Step 2: Get top countries by number of unique movies ---
    top_countries = (
        movies_df.groupby('country')['show_id']
        .nunique()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    # --- Step 3: Keep only those top countries ---
    movies_df = movies_df[movies_df['country'].isin(top_countries)]

    # --- Step 4: Build co-production matrix ---
    pairs_count = {}

    # For each show_id, get all countries involved
    for show_id, group in movies_df.groupby('show_id'):
        countries = sorted(group['country'].unique())
        # Create all combinations of country pairs for that show_id
        for c1, c2 in itertools.combinations(countries, 2):
            pairs_count[(c1, c2)] = pairs_count.get((c1, c2), 0) + 1

    # --- Step 5: Create symmetric matrix (DataFrame) ---
    matrix = pd.DataFrame(0, index=top_countries, columns=top_countries)

    for (c1, c2), count in pairs_count.items():
        matrix.loc[c1, c2] += count
        matrix.loc[c2, c1] += count

    # Diagonal entries = unique movies per country
    solo_counts = movies_df.groupby('country')['show_id'].nunique()
    for c in top_countries:
        matrix.loc[c, c] = solo_counts.get(c, 0)

    # --- Step 6: Plot Heatmap (Netflix style) ---
    sns.set_theme(style="white")
    plt.figure(figsize=(20, 16))
    netflix_red = "#E50914"
    netflix_palette = sns.color_palette(["#000000", netflix_red, "#B81D24"])

    sns.heatmap(
        matrix,
        cmap=netflix_palette,
        annot=True,
        fmt=".0f",
        linewidths=0.5,
        cbar_kws={'label': 'Number of Shared Unique Movies'}
    )

    plt.title(
        "Netflix Co-Production Heatmap: Shared Unique Movies Between Countries",
        fontsize=16,
        color=netflix_red,
        weight='bold'
    )
    plt.xlabel("Country", fontsize=12)
    plt.ylabel("Country", fontsize=12)
    plt.xticks(rotation=75, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_wordcloud(df,col,save=False): # John
    # plots the word cloud of the discrete data column "col"
    # df : dataframe
    # col : column for which you want the wordcloud to be plotted (discrete)
    # save : set as true if you want the figure to be saved in your current working directory
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    # Combine all genres into a single string
    text = ' '.join([genre for sublist in df[col] 
                    for genre in ([sublist] if isinstance(sublist, str) else sublist)])

    # Define a custom color function (Netflix red)
    def netflix_red_color_func(*args, **kwargs):
        return "#E50914"

    # Create the WordCloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',  # Black background makes red pop
        color_func=netflix_red_color_func
    ).generate(text)

    # Plot the WordCloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('', fontsize=16, color='#E50914')
    if save:
        plt.savefig(col+'_wordcloud.png')
    plt.show()

def plot_categorywise_corr(df,col1,col2,cat,save=False,sorted=False): # John
    # plots the correlation coeff between col1 and col2 for each cat
    # df : dataframe
    # col1 : column 1 (continuous)
    # col2 : column 2 (continuous)
    # cat : category column (discrete)
    # save : set as true if you want to save the fig
    # sorted : set as true if you want the plot to be sorted in ascending order
    import seaborn as sns
    import matplotlib.pyplot as plt
    import plotly.express as px
    genre_corr = (
        df
        .groupby(cat)
        .apply(lambda x: x[col1].corr(x[col2]))
        .sort_values(ascending=sorted)
    )
    plt.figure(figsize=(12, 6))
    genre_corr.plot(kind='bar', color='red', edgecolor='black')

    plt.title("Correlation between "+col1+" and "+col2+" per "+cat, fontsize=16)
    plt.xlabel(cat, fontsize=14)
    plt.ylabel("Pearson Correlation", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    if save:
        plt.savefig('correlation between '+col1+' and '+col2+' per '+cat+'.png')
    plt.show()

def plot_change_over_time(df,var,time,cat): # John
    # plots the change over time of var per cat
    # df : dataframe
    # var : continous data column
    # time : column name that contains data related to time
    # cat : category column 

    import seaborn as sns
    import matplotlib.pyplot as plt
    import plotly.express as px
    popularity_trend = (
        df
        .groupby([cat, time])[var]
        .mean()
        .reset_index()
    )

    fig = px.line(
        popularity_trend,
        x=time,
        y=var,
        color=cat,
        markers=True, 
        title="Change in "+var+" Over Time by "+cat,
        color_discrete_sequence=px.colors.qualitative.Light24 
    )

    fig.update_layout(
        width=1100,
        height=650,
        template='plotly_dark',
        title_font=dict(size=22, color='white'),
        xaxis_title=time,
        yaxis_title='Average '+var,
        legend_title=cat,
        plot_bgcolor='#141414',
        paper_bgcolor='#141414',
        legend=dict(
            title=cat,
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.02
        )
    )

    fig.update_xaxes(
        tickmode='linear',
        tick0=popularity_trend[time].min(),
        dtick=1
    )

    fig.show()

def plot_number_across_time(df,cat,time): # John 
    # plots the number of occurances of a particular categrory (cat) for every category across time
    # df : dataframe
    # cat : category column name
    # time : column that contains data related to time

    import seaborn as sns
    import matplotlib.pyplot as plt
    import plotly.express as px
    movie_counts = (
        df.groupby([time, cat])
        .size()
        .reset_index(name='num_movies')
    )

    fig = px.line(
        movie_counts,
        x=time,
        y='num_movies',
        color=cat, 
        title='Number of Movies Released per Year by '+cat,
        color_discrete_sequence=px.colors.qualitative.Set3 
    )

    fig.update_layout(
        width=1000,
        height=600,
        template='plotly_dark',
        title_font=dict(size=22, color='white'),
        xaxis_title=time,
        yaxis_title='Number of Movies',
        legend_title=cat,
        plot_bgcolor='#141414',
        paper_bgcolor='#141414',
    )

    fig.show()

def cumulative_number_plot(df,time,cat): # John
    # plots the cumulative number of occurances of a particular category over time for every catgeory
    # df : dataframe
    # time : column that contains data related to time
    # cat : column name of the categorical data

    import seaborn as sns
    import matplotlib.pyplot as plt
    import plotly.express as px
    movie_counts = (
        df.groupby([time, cat])
        .size()
        .reset_index(name='num_movies')
    )

    movie_counts['cumulative_movies'] = (
        movie_counts
        .groupby(cat)['num_movies']
        .cumsum()  
    )

    fig = px.line(
        movie_counts,
        x=time,
        y='cumulative_movies',
        color=cat,
        markers=True,
        title='Cumulative Number of Movies Released Over Time by '+cat,
        color_discrete_sequence=px.colors.qualitative.Light24  
    )

    fig.update_layout(
        width=1100,
        height=650,
        template='plotly_dark',
        title_font=dict(size=22, color='white'),
        xaxis_title=time,
        yaxis_title='Cumulative Number of Movies (till that year)',
        legend_title=cat,
        plot_bgcolor='#141414',
        paper_bgcolor='#141414',
        legend=dict(
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.02
        )
    )

    fig.update_xaxes(
        tickmode='linear',
        tick0=movie_counts[time].min(),
        dtick=1
    )

    fig.show()

def sunburst_plot(df,cat1,cat2): # John 
    # plots a sunburst plot which shows hierarchial relationships between two categorical variables
    # cat1 : the categorical variable that has upper hierarchy in the sunburst plot
    # cat2 : the categorical variable that has lower hierarchy in the sunburst plot

    import seaborn as sns
    import matplotlib.pyplot as plt
    import plotly.express as px

    # If the column contains lists, explode it
    if df[cat1].apply(lambda x: isinstance(x, list)).any():
        df = df.explode(cat1).reset_index(drop=True)

    if df[cat2].apply(lambda x: isinstance(x, list)).any():
        df = df.explode(cat2).reset_index(drop=True)

    genre_lang_counts = (
        df.groupby([cat1, cat2])
        .size()
        .reset_index(name='num_movies')
    )

    fig = px.sunburst(
        genre_lang_counts,
        path=[cat1, cat2], 
        values='num_movies',
        color=cat1,
        color_discrete_sequence=[
            '#E50914', '#B20710', '#F40612', '#FF0A16', '#A60311',
            '#CC0E14', '#99000D', '#FF1E22', '#B81D24', '#E50914'
        ],
        title='Movies by '+cat1+' and '+cat2
    )

    fig.update_layout(
        width=1000,
        height=1000,
        template='plotly_dark',
        title_font=dict(size=24, color='white', family='Arial Black'),
        font=dict(color='white'),
        paper_bgcolor='#141414', 
        plot_bgcolor='#141414',
        margin=dict(t=80, l=0, r=0, b=0)
    )

    fig.update_traces(
        hoverlabel=dict(bgcolor='white', font_color='black'),
        textfont=dict(color='white', size=12),
        insidetextorientation='radial'
    )

    fig.show()

def plot_top20(dic,cat,var): # John
    import seaborn as sns
    import matplotlib.pyplot as plt
    import plotly.express as px
    df_genres = pd.DataFrame(list(dic.items()), columns=[cat+' Combination', 'Average_'+var])

    df_top20 = df_genres.sort_values(by='Average_'+var, ascending=False).head(20)

    fig = px.bar(
        df_top20,
        x=cat+' Combination',
        y='Average_'+var,
        text='Average_'+var,
        title='Top 20 '+cat+' by Average '+var,
    )

    fig.update_traces(
        marker_color='#E50914',    
        texttemplate='%{text:.2f}',
        textposition='outside'
    )

    fig.update_layout(
        width=1000,
        height=600,
        template='plotly_dark',
        title_font=dict(size=24, color='white', family='Arial Black'),
        xaxis_title=cat+' Combination',
        yaxis_title='Average '+var,
        plot_bgcolor='#141414',
        paper_bgcolor='#141414',
        xaxis=dict(tickangle=45, tickfont=dict(size=12, color='white')),
        yaxis=dict(tickfont=dict(color='white')),
    )

    fig.show()

