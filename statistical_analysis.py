import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re

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

def chi2_cont_test(df,cat):
    from scipy.stats import chi2_contingency
    import pandas as pd

    movies=df.copy()
    movies['profitable'] = (movies['revenue'] > movies['budget']).astype(int)

    movies_exploded = movies.assign(genres=movies[cat].str.split(', ')).explode(cat)

    contingency_table = pd.crosstab(movies_exploded[cat], movies_exploded['profitable'])

    chi2, p, dof, expected = chi2_contingency(contingency_table)

    print("Chi-square Statistic:", chi2)
    print("Degrees of Freedom:", dof)
    print("P-value:", p)

    alpha = 0.05
    if p < alpha:
        print("\nReject the null hypothesis: Profitability depends on "+cat)
    else:
        print("\nFail to reject the null hypothesis: Profitability does not depend on "+cat)