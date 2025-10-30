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
    # plt.savefig(val+" vs "+cat+" sorted.png")
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
    ax.set_title(title)
    ax.axis("equal")
    if show:
        plt.show()

def barh_top_counts_series(   #Daksh
    s,
    *,
    title: str = "Top Categories (Count & Share)",
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
    plt.tight_layout()

    if show:
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

# def generate_bar_chart_race(df, category, value, time_col, title=None, n_bars=10, filename='bar_chart_race.mp4'):
#     """
#     Creates a bar chart race animation showing how a value changes over time across categories.

#     Parameters
#     ----------
#     df : pandas.DataFrame
#         DataFrame containing the data.
#     category : str
#         Name of the categorical column (e.g. 'genres').
#     value : str
#         Name of the numeric column (e.g. 'popularity').
#     time_col : str
#         Name of the time column (e.g. 'release_year').
#     title : str, optional
#         Title for the chart (default: auto-generated).
#     n_bars : int, optional
#         Number of top bars to display (default: 10).
#     filename : str, optional
#         Output file name (e.g. 'bar_chart_race.mp4' or '.gif').

#     Returns
#     -------
#     Displays and saves a bar chart race animation.
#     """

#     import pandas as pd
#     import bar_chart_race as bcr

#     # Clean data: remove missing values and aggregate by year and category
#     df_clean = (
#         df.dropna(subset=[category, value, time_col])
#         .groupby([time_col, category])[value]
#         .mean()
#         .reset_index()
#     )

#     # Pivot the table to get years as index and categories as columns
#     pivot_df = df_clean.pivot(index=time_col, columns=category, values=value).fillna(0)

#     # Sort columns alphabetically for consistency
#     pivot_df = pivot_df.sort_index(axis=1)

#     # Create a nice default title if not provided
#     if not title:
#         title = f"Change in {value.title()} over Time by {category.title()}"

#     # Create the bar chart race
#     bcr.bar_chart_race(
#         df=pivot_df,
#         n_bars=n_bars,
#         sort='desc',
#         title=title,
#         filename=filename,
#         filter_column_colors=True,
#         steps_per_period=10,
#         period_length=800,
#         interpolate_period=False,
#         cmap='Set2',  # colorful yet readable palette
#         bar_size=.95,
#         figsize=(6, 4),
#         period_label={'x': .95, 'y': .15, 'ha': 'right', 'va': 'center'},
#         period_summary_func=lambda v, r: {'x': .99, 'y': .05,
#                                           's': f'Total = {v.sum():,.0f}',
#                                           'ha': 'right', 'size': 8},
#         shared_fontdict={'family': 'Arial', 'weight': 'bold', 'color': 'gray'}
#     )

#     print(f"✅ Bar chart race saved as: {filename}")
def generate_line_chart( #Daksh
    s,
    *,
    title="Content Added Over Time",
    xlabel="Month",
    ylabel="Number of Titles Added",
    figsize=(10, 5),
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
    fig.tight_layout()
    plt.show()