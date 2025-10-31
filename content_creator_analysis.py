from helper_functions import *
# ============================================================
# Netflix Content Creator Analysis (Steps 3–5)
# ============================================================
# PURPOSE:
# This script analyzes Netflix creators (directors and actors)
# using a cleaned dataset. It produces top creator rankings,
# collaboration networks, genre specialization heatmaps,
# creator timelines, country diversity analysis, and F/LSD tests.
# ============================================================

# ---------- Imports ----------
import os
import pandas as pd

# ---------- Step 0: Load Dataset ----------
print("Loading dataset...")
df = pd.read_csv("final_cleaned_main.csv")


# Convert categorical Netflix ratings to numeric scores
rating_map = {
    'G': 1, 'TV-Y': 1, 'TV-Y7': 2,
    'PG': 2, 'TV-G': 2, 'TV-PG': 3,
    'PG-13': 3, 'TV-14': 4,
    'R': 4, 'NC-17': 5, 'TV-MA': 5
}
df['vote_average'] = df['rating'].map(rating_map).fillna(3)

# Create output directory if missing
os.makedirs("outputs", exist_ok=True)

print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")

# ============================================================
# STEP 3: CONTENT CREATOR ANALYSIS
# ============================================================

print("\n===== STEP 3: CONTENT CREATOR ANALYSIS =====")

# --- 3.1 Most Prolific Directors ---
top_directors = get_top_creators(df, "director", n=20)
print("\n🎬 Top 10 Directors:\n", top_directors.head(10))
plot_top_creators(top_directors, "Top 20 Most Prolific Directors on Netflix")
top_directors.to_csv("outputs/top_directors.csv")

# --- 3.2 Most Frequent Actors ---
top_actors = get_top_creators(df, "cast", n=20)
print("\n⭐ Top 10 Actors:\n", top_actors.head(10))
plot_top_creators(top_actors, "Top 20 Most Frequent Actors on Netflix")
top_actors.to_csv("outputs/top_actors.csv")

# --- 3.3 Director–Actor Collaboration Network ---
print("\n🕸️ Building collaboration network...")
G = build_collaboration_network(df)
print(f"Network contains {len(G.nodes())} nodes and {len(G.edges())} edges.")
plot_network(G)
print("Network plotted successfully!")

# ============================================================
# STEP 4: VISUALIZATIONS
# ============================================================

print("\n===== STEP 4: CREATOR VISUALIZATIONS =====")

# --- 4.1 Director–Genre Heatmap ---
print("\n🎭 Creating Director–Genre Specialization Heatmap...")

# Generate the Director–Genre matrix
matrix = director_genre_matrix(df, min_titles=3)

# Import and plot the heatmap
from helper_functions import plot_heatmap
plot_heatmap(
    matrix,
    row_label="Director",
    col_label="Genre",
    top_rows=15,
    title="Director–Genre Specialization Map"
)

# Save the matrix for later analysis
import os
os.makedirs("outputs", exist_ok=True)
matrix.to_csv("outputs/director_genre_matrix.csv", index=True)
print("✅ Director–Genre matrix successfully saved to 'outputs/director_genre_matrix.csv'.")

# --- 4.2 Creator Timelines (Yearly Trends) ---
print("\n📅 Plotting director timeline trends...")
plot_creator_timeline(df, creator_col="director", top_n=5)

print("\n📅 Plotting actor timeline trends...")
plot_creator_timeline(df, creator_col="cast", top_n=5)

# --- 4.3 Country Distribution ---
print("\n🌍 Analyzing country distribution of creators...")
plot_creator_country_distribution(df, creator_col="director")
print("\n🌍 Comparing international vs domestic creators...")
intl_summary = plot_international_vs_domestic(df, creator_col="director", home_country="United States")
print(intl_summary)

print("\n🌎 For actors as well...")
actor_intl_summary = plot_international_vs_domestic(df, creator_col="cast", home_country="United States")
print(actor_intl_summary)
# --- 4.4 Cast Frequency Distribution ---
print("\n🎭 Plotting actor appearance frequency distribution...")
plot_cast_frequency_distribution(df)


# ============================================================
# STEP 5: STATISTICAL TESTS
# ============================================================

print("\n===== STEP 5: STATISTICAL TESTS =====")
print("\n===== STEP 5.1: CHI-SQUARE TEST =====")
from helper_functions import chi_square_test, plot_chi_square_heatmap

# Example use case: Director vs Rating
chi2, p, dof, contingency = chi_square_test(df, 'director', 'rating')
print(f"Chi-square statistic = {chi2:.2f}, dof = {dof}, p-value = {p:.4f}")

if p < 0.05:
    print("✅ Reject H₀ → Significant relationship detected.")
    print("Certain directors tend to target specific rating categories.")
else:
    print("❌ Fail to reject H₀ → No strong relationship found.")

# Optional visualization
plot_chi_square_heatmap(contingency, var1_name="Director", var2_name="Rating", top_n=10)
print("\n===== STEP 5.4: ENTROPY (CREATOR SPECIALIZATION) =====")

# Compute specialization for directors across genres
entropy_df = compute_entropy(df, entity_col='director', category_col='listed_in')

print("\n📊 Sample (Top 5 Most Diverse Directors):")
print(entropy_df.head(5))

# print("\n📊 Sample (Top 5 Most Specialized Directors):")
# print(entropy_df.tail(5))

# Plot both views
plot_entropy(entropy_df, entity_col='director', top_n=10)
# print("\n===== STEP 5.5: CENTRALITY ANALYSIS (CREATOR NETWORK) =====")
# # Compute centrality measures
# centrality_df = compute_network_centrality(G)

# print("\n📊 Top 5 creators by degree centrality:")
# print(centrality_df.head(5))

# # Plot top creators by connectivity
# plot_top_central_nodes(centrality_df, metric='degree_centrality', top_n=10)

# --- 5.3 ANOVA (Duration by Director)
F_stat, p_val = anova_test(df, group_col='director', value_col='duration')

if F_stat is not None:
    print(f"ANOVA F-statistic = {F_stat:.3f}, p-value = {p_val:.4f}")
    if p_val < 0.05:
        print("✅ Reject H₀ → Significant difference in average duration among directors.")
    else:
        print("❌ Fail to reject H₀ → No significant difference in duration across directors.")
df['duration_num'] = (
    df['duration']
    .astype(str)
    .str.extract(r'(\d+)')
    .astype(float)
)

print("\n📦 Visualizing duration distribution by director...")
generate_boxplot(df, cat='director', val='duration_num', outlier=False)
# --- 5.2 F-Test & LSD-Test: Director vs Ratings ---
print("\n📊 Running F-test and LSD post-hoc comparison for directors (vote_average)...")
result = director_rating_significance(df, val_col="vote_average")

if result:
    print("\nSignificant differences found. Sample output (first 5 entries):")
    for key in list(result.keys())[:5]:
        print(f"{key} > {result[key]}")
else:
    print("\nNo significant rating difference detected across directors.")



# ============================================================
# END OF ANALYSIS
# ============================================================
print("\n✅ All analyses completed successfully! Check the 'outputs/' folder for saved results.")
