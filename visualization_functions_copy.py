import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re

#Recommendation Insights additon
import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import ttest_ind
import statsmodels.api as sm

import plotly.express as px
import plotly.graph_objects as go

# --- Font & Color Definitions ---
# As per the PDF document
HEADING_FONT = "Roboto, Arial, sans-serif"
BODY_FONT = "Merriweather, Georgia, serif"
NETFLIX_RED = '#E50914'
TEXT_COLOR = 'white'
TEMPLATE = 'plotly_dark'

#End of rec insights additions

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


def bar_stacked(                        #Sourendra
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

def bar_chart_vertical(                 #Sourendra
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

def heatmap_by_category(                #Sourendra
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

def generate_multi_line_chart(          #Sourendra
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

def generate_heatmap_flexible(          #Sourendra
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



def plot_niche_superstar_matrix(        #Sourendra
        df: pd.DataFrame,
        top_n_actors: int = 15, 
        top_n_genres: int = 10) -> go.Figure:
    """
    (ADVANCED) Generates a heatmap to identify actors who are exceptionally popular in specific genres.
    """
    # Explode dataframe for actors and genres
    actor_df = df.assign(actor=df['cast'].str.split(', ')).explode('actor')
    genre_df = actor_df.assign(genre=actor_df['listed_in'].str.split(', ')).explode('genre')
    
    # Find top actors and genres to keep the matrix focused
    top_actors = genre_df['actor'].value_counts().nlargest(top_n_actors).index
    top_genres = genre_df['genre'].value_counts().nlargest(top_n_genres).index
    
    # Filter for top talent and genres
    filtered_df = genre_df[genre_df['actor'].isin(top_actors) & genre_df['genre'].isin(top_genres)]
    
    # Create the pivot table (matrix) of average popularity scores
    pivot_table = pd.pivot_table(filtered_df, values='score', index='actor', columns='genre', aggfunc='mean')
    
    fig = px.imshow(pivot_table,
                    title=f"<b>The Niche Superstar Matrix</b>",
                    labels=dict(x="Genre", y="Actor", color="Avg. Score"),
                    color_continuous_scale=['#333333', NETFLIX_RED])
    
    fig.update_layout(template=TEMPLATE, title={'x':0.5, 'font': {'size': 20, 'family': HEADING_FONT}},
                      font=dict(family=BODY_FONT, color=TEXT_COLOR))
    return fig

def plot_content_strategy_evolution(    #Sourendra
        df: pd.DataFrame) -> go.Figure:
    """
    Generates a 100% stacked area chart showing the proportional shift in genre strategy over time.
    """
    df['date_added'] = pd.to_datetime(df['date_added'])
    df['year_added'] = df['date_added'].dt.year
    
    genre_df = df.assign(genre=df['listed_in'].str.split(', ')).explode('genre')
    top_genres = genre_df['genre'].value_counts().nlargest(8).index
    genre_df['genre_agg'] = np.where(genre_df['genre'].isin(top_genres), genre_df['genre'], 'Other')
    
    yearly_genre_counts = genre_df.groupby(['year_added', 'genre_agg']).size().unstack(fill_value=0)
    yearly_genre_percent = yearly_genre_counts.apply(lambda x: x*100/sum(x), axis=1)
    
    fig = px.area(yearly_genre_percent,
                  title="<b>Evolution of Content Strategy (Genre Mix)</b>",
                  labels={'year_added': 'Year Content Added', 'value': 'Percentage of Catalog (%)', 'genre_agg': 'Genre'},
                  color_discrete_sequence=px.colors.qualitative.Set1) # Using a distinct color palette
    
    fig.update_layout(template=TEMPLATE, title={'x':0.5, 'font': {'size': 20, 'family': HEADING_FONT}},
                      font=dict(family=BODY_FONT, color=TEXT_COLOR))
    return fig

def plot_content_age_sweet_spot(        #Sourendra
        df: pd.DataFrame) -> go.Figure:
    """
    Generates a scatter plot with a regression line to find the relationship between content age and popularity.
    """
    fig = px.scatter(df, x='release_year', y='score',
                     title='<b>The "Goldilocks Zone" of Content Age</b>',
                     labels={'release_year': 'Original Release Year', 'score': 'Popularity Score'},
                     opacity=0.6,
                     trendline='lowess', # A flexible trendline (Locally Weighted Scatterplot Smoothing)
                     trendline_color_override='white')
    fig.update_traces(marker=dict(color=NETFLIX_RED))
    
    fig.update_layout(template=TEMPLATE, title={'x':0.5, 'font': {'size': 20, 'family': HEADING_FONT}},
                      font=dict(family=BODY_FONT, color=TEXT_COLOR))
    return fig

def plot_genre_complexity_impact(       #Sourendra
        df: pd.DataFrame) -> go.Figure:
    """
    Generates a violin plot to analyze if having more genres affects a title's popularity.
    """
    df['genre_count'] = df['listed_in'].apply(lambda x: len(x.split(', ')))
    
    fig = px.violin(df[df['genre_count'] <= 5], # Cap at 5 for readability
                    x='genre_count',
                    y='score',
                    box=True, # Display a box plot inside the violin
                    points='all', # Show individual data points
                    title='<b>Impact of Genre Complexity on Popularity</b>',
                    labels={'genre_count': 'Number of Genres Assigned to Title', 'score': 'Popularity Score'},
                    color_discrete_sequence=[NETFLIX_RED])
                     
    fig.update_layout(template=TEMPLATE, title={'x':0.5, 'font': {'size': 20, 'family': HEADING_FONT}},
                      font=dict(family=BODY_FONT, color=TEXT_COLOR))
    return fig

def plot_director_efficiency_quadrant(  #Sourendra
        df: pd.DataFrame, 
        min_titles: int = 4) -> go.Figure:
    """
    (ADVANCED - Presentation Ready) Generates a scatter plot to identify director efficiency.
    Selectively labels key directors and color-codes quadrants for maximum readability.
    """
    director_stats = df.assign(director=df['director'].str.split(', ')).explode('director')
    director_agg = director_stats.groupby('director')['score'].agg(['mean', 'count']).reset_index()
    director_agg = director_agg[director_agg['count'] >= min_titles].reset_index(drop=True)

    # Calculate median lines for quadrants
    median_count = director_agg['count'].median()
    median_score = director_agg['mean'].median()

    # --- NEW: Assign each director to a quadrant for coloring ---
    conditions = [
        (director_agg['count'] >= median_count) & (director_agg['mean'] >= median_score),
        (director_agg['count'] < median_count) & (director_agg['mean'] >= median_score),
        (director_agg['count'] >= median_count) & (director_agg['mean'] < median_score),
        (director_agg['count'] < median_count) & (director_agg['mean'] < median_score)
    ]
    choices = ['Proven Hit-Makers', 'Niche Masters', 'Workhorses', 'Emerging Talent']
    director_agg['quadrant'] = np.select(conditions, choices, default='Other')
    
    # --- NEW: Create the plot with color-coded quadrants ---
    fig = px.scatter(director_agg,
                     x='count',
                     y='mean',
                     color='quadrant',  # Color by the new quadrant column
                     size='count',
                     hover_data=['director'], # Show director name on hover
                     title='<b>The Director Efficiency Quadrant</b>',
                     labels={'count': 'Volume (Number of Popular Titles)', 'mean': 'Impact (Average Popularity Score)'},
                     color_discrete_map={ # Assign specific, vibrant colors
                         'Proven Hit-Makers': 'lightgreen',
                         'Niche Masters': 'cyan',
                         'Workhorses': 'orange',
                         'Emerging Talent': 'grey'
                     })

    # Add median lines
    fig.add_vline(x=median_count, line_width=1, line_dash="dash", line_color="grey")
    fig.add_hline(y=median_score, line_width=1, line_dash="dash", line_color="grey")
    
    # --- NEW: Selectively add text labels for clarity ---
    # Define a list of key directors to label to avoid clutter
    directors_to_label = [
        'Martin Scorsese', 'Steven Spielberg', 'Tom Hooper', 'Christopher Nolan',
        'Quentin Tarantino', 'David Fincher', 'Edgar Wright', 'Bong Joon Ho',
        'Paul Thomas Anderson', 'Yorgos Lanthimos', 'Lana Wachowski',
        'Paul W.S. Anderson', 'McG' # Examples from each quadrant
    ]
    
    # Add annotations only for the selected directors
    for i, row in director_agg.iterrows():
        if row['director'] in directors_to_label:
            fig.add_annotation(
                x=row['count'], y=row['mean'],
                text=f"<b>{row['director']}</b>",
                showarrow=True, arrowhead=1, arrowcolor='white', ax=20, ay=-30,
                font=dict(family=BODY_FONT, size=11, color="white"),
                bgcolor="rgba(0,0,0,0.6)"
            )

    fig.update_layout(
        template=TEMPLATE,
        title={'x':0.5, 'font': {'size': 24, 'family': HEADING_FONT}},
        font=dict(family=BODY_FONT, color=TEXT_COLOR, size=12),
        legend_title_text='<b>Strategic Quadrant</b>'
    )
    return fig

def plot_audience_engagement_pyramid(   #Sourendra
        df: pd.DataFrame) -> go.Figure:
    """
    (ADVANCED) Generates a box plot to compare the popularity scores across strategic audience segments.
    This moves beyond catalog composition to analyze which audience segment drives the most engagement.
    """
    # Create a simplified audience segment based on content rating
    conditions = [
        df['rating'].isin(['TV-Y', 'TV-Y7', 'TV-G', 'G', 'PG']),
        df['rating'].isin(['TV-PG', 'TV-14', 'PG-13']),
        df['rating'].isin(['TV-MA', 'R', 'NC-17'])
    ]
    choices = ['Kids & Family', 'Teens & Young Adult', 'Mature Adults']
    df['audience_segment'] = np.select(conditions, choices, default='Unrated')
    
    # Filter out the 'Unrated' category for a cleaner plot
    plot_df = df[df['audience_segment'] != 'Unrated']
    
    fig = px.box(plot_df,
                 x='audience_segment',
                 y='score',
                 color='audience_segment',
                 title='<b>The Audience Engagement Pyramid</b>',
                 labels={'audience_segment': 'Strategic Audience Segment', 'score': 'Popularity Score'},
                 category_orders={'audience_segment': ['Kids & Family', 'Teens & Young Adult', 'Mature Adults']}, # Enforce logical order
                 color_discrete_map={
                     'Kids & Family': 'grey',
                     'Teens & Young Adult': '#B3B3B3',
                     'Mature Adults': NETFLIX_RED
                 })
                 
    fig.update_layout(template=TEMPLATE, title={'x':0.5, 'font': {'size': 20, 'family': HEADING_FONT}},
                      font=dict(family=BODY_FONT, color=TEXT_COLOR), showlegend=False)
    return fig

def plot_global_strategy_validation(    #Sourendra
        df: pd.DataFrame) -> go.Figure:
    """
    (STATISTICALLY-DRIVEN) Compares the performance of US-produced content vs. International content.
    A t-test validates if there's a significant difference, answering a key strategic question about global expansion.
    """
    # Label content as Domestic (US) or International
    df['origin'] = np.where(df['country'].str.contains("United States", na=False), "Domestic (US)", "International")
    
    # Separate the scores for the t-test
    domestic_scores = df[df['origin'] == 'Domestic (US)']['score']
    international_scores = df[df['origin'] == 'International']['score']
    
    # Perform the independent t-test
    stat, p_value = ttest_ind(domestic_scores, international_scores, equal_var=False, nan_policy='omit')
    
    fig = px.violin(df,
                    x='origin',
                    y='score',
                    color='origin',
                    box=True,
                    title=f"<b>Validating the Global Strategy (p-value: {p_value:.3f})</b>",
                    labels={'origin': 'Content Origin', 'score': 'Popularity Score'},
                    color_discrete_map={
                        'Domestic (US)': '#B3B3B3',
                        'International': NETFLIX_RED
                    })

    fig.update_layout(template=TEMPLATE, title={'x':0.5, 'font': {'size': 20, 'family': HEADING_FONT}},
                      font=dict(family=BODY_FONT, color=TEXT_COLOR), showlegend=False)
    return fig

def weighted_rating(                    #Sourendra
        x, 
        m: float, 
        C: float) -> float:
    """
    Calculates the IMDb weighted rating for a movie or show.
    
    Args:
        x: A row of a DataFrame containing 'numVotes' and 'averageRating'.
        m: The minimum number of votes required (the threshold).
        C: The mean rating across the whole dataset.
    
    Returns:
        The calculated weighted rating score.
    """
    v = x['numVotes']
    R = x['averageRating']
    return (v / (v + m)) * R + (m / (v + m)) * C

def plot_content_lag_sweet_spot(        #Sourendra
        df: pd.DataFrame) -> go.Figure:
    """
    (ADVANCED) Analyzes the relationship between "content lag" (time from release to Netflix addition) and popularity.
    This provides direct insights for the content acquisition team on the value of freshness.
    """
    # Create the 'content_lag' feature
    df_copy = df.copy()
    df_copy['date_added'] = pd.to_datetime(df_copy['date_added'])
    df_copy['year_added'] = df_copy['date_added'].dt.year
    df_copy['content_lag_years'] = df_copy['year_added'] - df_copy['release_year']
    
    # Filter for a reasonable range (e.g., content added within 30 years of release)
    plot_df = df_copy[(df_copy['content_lag_years'] >= 0) & (df_copy['content_lag_years'] <= 30)]

    fig = px.scatter(plot_df,
                     x='content_lag_years',
                     y='score',
                     title='<b>The Content Lag "Sweet Spot"</b>',
                     labels={'content_lag_years': 'Content Lag (Years from Release to Netflix Addition)', 'score': 'Popularity Score'},
                     opacity=0.6,
                     trendline='lowess', # Locally Weighted Scatterplot Smoothing is great for noisy data
                     trendline_color_override='white')
    
    # Update the markers (the dots) to be Netflix Red
    fig.update_traces(marker=dict(color=NETFLIX_RED))
    
    fig.update_layout(template=TEMPLATE, title={'x':0.5, 'font': {'size': 20, 'family': HEADING_FONT}},
                      font=dict(family=BODY_FONT, color=TEXT_COLOR))
    return fig


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

def plot_lossmakers(df,rev,bud,cat):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import plotly.express as px

    loss_df = df[df[rev] < df[bud]]

    loss_counts = (
        loss_df.groupby(cat)
        .size()
        .reset_index(name='num_loss_movies')
        .sort_values(by='num_loss_movies', ascending=True)
    )

    fig = px.bar(
        loss_counts,
        x=cat,
        y='num_loss_movies',
        text='num_loss_movies',
        title='Number of Movies per Genre with Revenue Less than Budget',
    )

    fig.update_traces(
        marker_color='#E50914',       
        texttemplate='%{text}', 
        textposition='outside'
    )

    fig.update_layout(
        width=1000,
        height=600,
        template='plotly_dark',
        title_font=dict(size=22, color='white', family='Arial Black'),
        xaxis_title=cat,
        yaxis_title='Number of Movies (Revenue < Budget)',
        plot_bgcolor='#141414',
        paper_bgcolor='#141414',
        xaxis=dict(
            tickangle=45,
            categoryorder='total ascending', 
            tickfont=dict(color='white')
        ),
        yaxis=dict(tickfont=dict(color='white')),
    )

    fig.show()

def fraction_lossmaking(df,cat,rev,bud):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import plotly.express as px

    total_counts = df.groupby(cat).size().reset_index(name='total_movies')

    loss_counts = (
        df[df[rev] < df[bud]]
        .groupby(cat)
        .size()
        .reset_index(name='loss_movies')
    )

    genre_stats = pd.merge(total_counts, loss_counts, on=cat, how='left').fillna(0)
    genre_stats['loss_fraction'] = genre_stats['loss_movies'] / genre_stats['total_movies']

    genre_stats = genre_stats.sort_values(by='loss_fraction', ascending=True)

    fig = px.bar(
        genre_stats,
        x=cat,
        y='loss_fraction',
        text=genre_stats['loss_fraction'].apply(lambda x: f"{x:.2%}"),
        title='Fraction of Movies per Genre that Made a Loss',
    )

    fig.update_traces(
        marker_color='#E50914',
        textposition='outside'
    )

    fig.update_layout(
        width=1000,
        height=600,
        template='plotly_dark',
        title_font=dict(size=22, color='white', family='Arial Black'),
        xaxis_title='Genre',
        yaxis_title='Fraction of Movies (Revenue < Budget)',
        plot_bgcolor='#141414',
        paper_bgcolor='#141414',
        xaxis=dict(
            tickangle=45,
            categoryorder='total ascending',
            tickfont=dict(color='white')
        ),
        yaxis=dict(
            tickformat='.0%',
            tickfont=dict(color='white')
        )
    )

    fig.show()

def gap_analysis(df,cat,var,movies=True):
    import pandas as pd
    import plotly.express as px

    # Copy and compute director stats
    df=df[df[cat]!='missing']

    directors_stats = (
        df.groupby(cat)
        .agg(
            avg_var=(var, 'mean'),
            movie_count=('title', 'nunique')
        )
        .reset_index()
    )

    # Filter directors with at least 5 movies
    directors_stats = directors_stats[directors_stats['movie_count'] >= 5]

    # Create interactive scatter plot
    if movies:
        fig = px.scatter(
            directors_stats,
            x='movie_count',
            y='avg_var',
            color='avg_var',
            color_continuous_scale=['#E50914', '#B20710', '#FFFFFF'],  # Netflix red → white
            hover_data={
                cat: True,
                'movie_count': True,
                'avg_var': ':.2f',  # two decimal places
            },
            title=cat+" on Netflix: Movie Count vs. Average "+var,
        )
    else:
        fig = px.scatter(
            directors_stats,
            x='movie_count',
            y='avg_var',
            color='avg_var',
            color_continuous_scale=['#E50914', '#B20710', '#FFFFFF'],  # Netflix red → white
            hover_data={
                cat: True,
                'movie_count': True,
                'avg_var': ':.2f',  # two decimal places
            },
            title=cat+" on Netflix: Show Count vs. Average "+var,
        )

    # Style the markers
    fig.update_traces(
        marker=dict(size=11, line=dict(width=1, color='white')),
        selector=dict(mode='markers'),
        text=None  # no text labels shown directly
    )

    # Update layout (Netflix-themed)
    if movies:
        fig.update_layout(
            width=1000,
            height=650,
            template='plotly_dark',
            title_font=dict(size=22, color='white'),
            xaxis_title='Number of Movies on Netflix',
            yaxis_title='Average '+var+' of Their Movies',
            plot_bgcolor='#141414',
            paper_bgcolor='#141414',
            coloraxis_colorbar=dict(title='Avg '+var, tickfont=dict(color='white')),
        )
    else:
        fig.update_layout(
            width=1000,
            height=650,
            template='plotly_dark',
            title_font=dict(size=22, color='white'),
            xaxis_title='Number of Shows on Netflix',
            yaxis_title='Average '+var+' of Their Shows',
            plot_bgcolor='#141414',
            paper_bgcolor='#141414',
            coloraxis_colorbar=dict(title='Avg '+var, tickfont=dict(color='white')),
        )

    fig.show()
    high_potential = directors_stats[
        (directors_stats['avg_var'] > directors_stats['avg_var'].mean()) &
        (directors_stats['movie_count'] < directors_stats['movie_count'].median())
    ]

    if movies:
        high_potential.rename(columns={'movie_count':'number of movies on Netflix'},inplace=True)
    else:
        high_potential.rename(columns={'movie_count':'number of shows on Netflix'},inplace=True)
    print("TOP 10 HIGH POTENTIAL UNDERREPRESENTED DIRECTORS")
    print(high_potential.sort_values('avg_var', ascending=False).head(10))

