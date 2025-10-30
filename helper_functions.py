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
def plot_chi_square_heatmap(contingency, var1_name="Variable 1", var2_name="Variable 2", top_n=10):#Taniya
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

    # Take top N categories by row totals for clarity
    top_rows = contingency.sum(axis=1).sort_values(ascending=False).head(top_n).index
    subset = contingency.loc[top_rows]

    plt.figure(figsize=(10, 6))
    sns.heatmap(subset, cmap='YlGnBu', annot=True, fmt='d')
    plt.title(f"{var1_name} vs {var2_name} Distribution (Top {top_n})", fontsize=14)
    plt.xlabel(var2_name)
    plt.ylabel(var1_name)
    plt.tight_layout()
    plt.show()
def anova_test(df, group_col, value_col):#Taniya
    """
    Performs a one-way ANOVA test to determine whether the mean of a continuous 
    variable differs significantly across groups.

    INPUTS:
        df (pandas.DataFrame) -> Dataset containing categorical and continuous columns
        group_col (str) -> Column name for the categorical variable (e.g., 'director')
        value_col (str) -> Column name for the continuous variable (e.g., 'duration')

    RETURNS:
        F-statistic (float), p-value (float)

    PURPOSE:
        Tests whether the average of a numeric column (e.g., duration) 
        differs significantly across categories (e.g., directors).
    """
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
def plot_top_creators(series, title, color='steelblue'):#Taniya
    # """
    # INPUTS:
    #     series (pandas.Series) -> Index = creator names; values = counts.
    #     title (str)            -> Chart title.
    #     color (str, optional)  -> Bar color (default = 'steelblue').

    # RETURNS:
    #     None (displays the bar plot).

    # PURPOSE:
    #     Visualizes top creators by count using a horizontal bar chart.
    # """
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    sns.barplot(x=series.values, y=series.index, color=color)
    plt.title(title, fontsize=15)
    plt.xlabel("Number of Titles")
    plt.ylabel("")
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

def plot_network(G, max_nodes=150):#Taniya
    """
    INPUTS:
        G (networkx.Graph) -> Collaboration network from build_collaboration_network().
        max_nodes (int)    -> Maximum nodes to show (default = 150).

    RETURNS:
        None (displays enhanced bipartite collaboration visualization).

    PURPOSE:
        Improves the director–actor network visualization by:
        - Coloring directors and actors differently
        - Scaling node sizes by degree (collaboration frequency)
        - Using a cleaner spring layout
    """
    import networkx as nx
    import matplotlib.pyplot as plt

    # Limit graph size for readability
    if len(G.nodes) > max_nodes:
        G = G.subgraph(list(G.nodes)[:max_nodes])

    plt.figure(figsize=(14, 10))

    # Identify directors and actors
    directors = [n for n in G.nodes if ' ' in n and len(n.split()) <= 3]
    actors = list(set(G.nodes) - set(directors))

    # Colors and node sizes
    node_colors = ['skyblue' if n in directors else 'salmon' for n in G.nodes]
    node_sizes = [100 + 3 * nx.degree(G, n) for n in G.nodes]

    # Layout
    pos = nx.spring_layout(G, k=0.3, iterations=40, seed=42)

    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.85)
    nx.draw_networkx_edges(G, pos, alpha=0.4)

    # Label top 10 most connected nodes
    top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:10]
    label_nodes = [n for n, _ in top_nodes if n in pos]
    labels = {n: n for n in label_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='black')

    plt.title("Director–Actor Collaboration Network (Enhanced View)", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# def plot_genre_heatmap(matrix, top_directors=15):#Taniya
#     """
#     INPUTS:
#         matrix (pd.DataFrame) -> Director–Genre pivot table from director_genre_matrix().
#         top_directors (int)   -> Number of top directors to display (default = 15).

#     RETURNS:
#         None (displays improved heatmap).

#     PURPOSE:
#         Visualizes directors and the genres they specialize in by showing
#         the count of titles per genre for each top director.
#     """
#     import seaborn as sns
#     import matplotlib.pyplot as plt

#     # Focus on top directors by total works
#     top = matrix.sum(axis=1).sort_values(ascending=False).head(top_directors).index
#     subset = matrix.loc[top]

#     # Sort genres by overall popularity
#     subset = subset[subset.sum().sort_values(ascending=False).index]

#     plt.figure(figsize=(14, 7))
#     sns.heatmap(
#         subset,
#         cmap="YlOrRd",
#         linewidths=0.3,
#         linecolor='white',
#         cbar_kws={'label': 'Number of Titles'}
#     )
#     plt.title("Director–Genre Specialization Map (Top Directors)", fontsize=15)
#     plt.xlabel("Genre")
#     plt.ylabel("Director")
#     plt.xticks(rotation=45, ha="right")
#     plt.tight_layout()
#     plt.show()
def plot_heatmap(matrix, row_label="Row", col_label="Column", top_rows=15, cmap="YlOrRd", title=None):#Taniya
    # """
    # Plots a generalized heatmap for any 2D pivot table (entity–category relationships).

    # INPUTS:
    #     matrix (pd.DataFrame) -> Pivot table or cross-tab (rows = entities, columns = categories)
    #     row_label (str) -> Label for rows (e.g., 'Director', 'Actor', 'Country')
    #     col_label (str) -> Label for columns (e.g., 'Genre', 'Category', 'Rating')
    #     top_rows (int) -> Number of top rows (entities) to display (default = 15)
    #     cmap (str) -> Color map for heatmap (default = 'YlOrRd')
    #     title (str or None) -> Custom title; if None, auto-generates one

    # RETURNS:
    #     None (displays the heatmap)

    # PURPOSE:
    #     Visualizes relationships between any two categorical dimensions in a dataset.
    #     Example uses:
    #     - Director vs Genre specialization
    #     - Actor vs Genre diversity
    #     - Country vs Category focus
    # """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    # Validate matrix
    if not isinstance(matrix, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame pivot table.")

    # Select top entities (rows)
    top_entities = matrix.sum(axis=1).sort_values(ascending=False).head(top_rows).index
    subset = matrix.loc[top_entities]

    # Sort columns (categories) by overall frequency
    subset = subset[subset.sum().sort_values(ascending=False).index]

    # Generate title if not provided
    if title is None:
        title = f"{row_label}–{col_label} Relationship Heatmap (Top {top_rows})"

    # Plot
    plt.figure(figsize=(14, 7))
    sns.heatmap(
        subset,
        cmap=cmap,
        linewidths=0.3,
        linecolor='white',
        cbar_kws={'label': 'Count'},
    )
    plt.title(title, fontsize=15)
    plt.xlabel(col_label)
    plt.ylabel(row_label)
    plt.xticks(rotation=45, ha="right")
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




def plot_creator_country_distribution(df, creator_col='director'):#Taniya
    # """
    # INPUTS:
    #     df (pandas.DataFrame)  -> Dataset with 'country' and creator column.
    #     creator_col (str)      -> Column to analyze ('director' or 'cast').

    # RETURNS:
    #     None (displays bar chart).

    # PURPOSE:
    #     Shows top countries by count of unique creators to study international
    #     versus domestic talent distribution.
    # """
    import seaborn as sns
    import matplotlib.pyplot as plt

    country_counts = (df[['country', creator_col]]
                      .drop_duplicates()
                      .country.value_counts()
                      .head(15))

    plt.figure(figsize=(8, 6))
    sns.barplot(x=country_counts.values, y=country_counts.index, palette="crest")
    plt.title("Top 15 Countries by Creator Count", fontsize=14)
    plt.xlabel("Number of Creators")
    plt.tight_layout()
    plt.show()

def director_rating_significance(df, val_col='vote_average'):#Taniya
    # """
    # INPUTS:
    #     df (pandas.DataFrame)  -> Dataset containing 'director' and rating column.
    #     val_col (str)          -> Continuous value to compare (default = 'vote_average').

    # RETURNS:
    #     dict or None -> Dictionary of significant pairwise differences (if F-test significant).

    # PURPOSE:
    #     Uses existing Ftest() and LSD_test() functions to check if directors differ
    #     significantly in average rating, then performs pairwise LSD comparisons.
    # """
    f_stat, f_crit, sse = Ftest(df, 'director', val_col)
    print(f"F-statistic = {f_stat:.3f}, F-critical = {f_crit:.3f}")
    if f_stat > f_crit:
        print("Reject H₀ → Significant difference between directors.")
        result = LSD_test(df, 'director', val_col, sse)
        return result
    else:
        print("Fail to reject H₀ → No significant difference detected.")

def plot_international_vs_domestic(df, creator_col='director', home_country='United States'):#Taniya
    # """
    # INPUTS:
    #     df (pandas.DataFrame)  -> Dataset with 'country' and creator column.
    #     creator_col (str)      -> Column to analyze ('director' or 'cast').
    #     home_country (str)     -> Country considered "domestic" (default = 'United States').

    # RETURNS:
    #     pandas.DataFrame -> Summary table of domestic vs international creator counts.

    # PURPOSE:
    #     Compares how many unique creators are domestic vs international,
    #     showing Netflix's global diversity of talent.
    # """
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

    # Plot
    plt.figure(figsize=(6, 5))
    sns.barplot(data=summary, x='Talent Type', y='Creator Count', palette='Set2')
    plt.title(f"International vs Domestic {creator_col.capitalize()}s ({home_country})", fontsize=14)
    plt.tight_layout()
    plt.show()

    return summary
def plot_cast_frequency_distribution(df):#Taniya
    """
    INPUTS:
        df (pandas.DataFrame) -> Dataset with 'cast' column.

    RETURNS:
        None (displays histogram and optional bubble chart).

    PURPOSE:
        Visualizes how frequently actors appear across Netflix titles.
        Helps understand whether a few actors dominate the catalog
        or if the presence is evenly distributed.
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Remove 'Unknown' and count frequency
    cast_counts = df[df['cast'].str.lower() != 'unknown']['cast'].value_counts()

    # Histogram of appearances
    plt.figure(figsize=(8,5))
    sns.histplot(cast_counts, bins=30, kde=True, color='teal')
    plt.title("Distribution of Actor Appearances on Netflix", fontsize=14)
    plt.xlabel("Number of Titles Appeared In")
    plt.ylabel("Number of Actors")
    plt.tight_layout()
    plt.show()

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
        color='coral'
    )
    plt.title("Top 30 Actor Appearance Bubble Chart", fontsize=14)
    plt.xlabel("Number of Titles Appeared In")
    plt.ylabel("Actor")
    plt.tight_layout()
    plt.show()
def plot_creator_timeline(df, creator_col='director', top_n=5):#Taniya
    """
    INPUTS:
        df (pandas.DataFrame) -> Dataset containing creator names and release years
        creator_col (str) -> Column to analyze ('director' or 'cast')
        top_n (int) -> Number of top creators to include in the timeline

    RETURNS:
        None (displays lineplot)

    PURPOSE:
        Visualizes how the content output of top creators changes over time.
        Helps identify career trends and Netflix’s collaborations over years.
    """
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
    plt.legend(title=creator_col.title(), bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
def compute_entropy(df, entity_col, category_col):#Taniya
    """
    Computes Shannon Entropy for any categorical pair (e.g., Director–Genre, Actor–Genre, Country–Category).

    INPUTS:
        df (pandas.DataFrame) -> dataset containing both categorical columns
        entity_col (str) -> column name representing the entity (e.g., 'director', 'actor', 'country')
        category_col (str) -> column name representing the category (e.g., 'listed_in', 'genre', 'category')

    RETURNS:
        pd.DataFrame -> DataFrame with columns:
                        [entity_col, 'entropy', 'num_records']

    PURPOSE:
        Quantifies specialization or diversity for each entity.
        - Low entropy → specialized in fewer categories
        - High entropy → diversified across many categories
    """
    import pandas as pd
    import numpy as np

    # Drop missing or unknown values
    data = df.dropna(subset=[entity_col, category_col])
    if data[entity_col].dtype == 'object':
        data = data[data[entity_col].str.lower() != 'unknown']
    if data[category_col].dtype == 'object':
        data = data[data[category_col].str.lower() != 'unknown']

    # Split comma-separated category entries if present
    data = data.assign(**{category_col: data[category_col].astype(str).str.split(',\s*')})
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

def plot_entropy(entropy_df, entity_col, top_n=10):#Taniya
    """
    Plots top and bottom N entities based on entropy (diversity).

    INPUTS:
        entropy_df (pd.DataFrame) -> DataFrame returned from compute_entropy()
        entity_col (str) -> entity column name (e.g., 'director', 'actor')
        top_n (int) -> number of top and bottom entities to visualize

    RETURNS:
        None (displays two bar charts)
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Highest and lowest entropy entities
    top_diverse = entropy_df.head(top_n)
    top_specialized = entropy_df.tail(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_diverse, x='entropy', y=entity_col, color='skyblue')
    plt.title(f"Top {top_n} Most Diverse {entity_col.title()}s (High Entropy)")
    plt.xlabel("Entropy (Diversity)")
    plt.ylabel(entity_col.title())
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_specialized, x='entropy', y=entity_col, color='salmon')
    plt.title(f"Top {top_n} Most Specialized {entity_col.title()}s (Low Entropy)")
    plt.xlabel("Entropy (Diversity)")
    plt.ylabel(entity_col.title())
    plt.tight_layout()
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