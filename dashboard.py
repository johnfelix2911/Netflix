# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---------------------------------------------------------------------
# CONFIG + THEME (Netflix-ish)
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Netflix Strategic Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

NETFLIX_RED = "#E50914"
NETFLIX_BG = "#141414"
NETFLIX_CARD = "#181818"
NETFLIX_TEXT = "#ffffff"

st.markdown(
    f"""
    <style>
    .stApp {{
        background: {NETFLIX_BG};
        color: {NETFLIX_TEXT};
    }}
    header[data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}
    section[data-testid="stSidebar"] {{
        background-color: #000;
    }}
    .kpi-card {{
        background: {NETFLIX_CARD};
        padding: 1.1rem 1rem 1rem 1rem;
        border-radius: 0.8rem;
        border-left: 4px solid {NETFLIX_RED};
        text-align: left;
        color: {NETFLIX_TEXT};
    }}
    .kpi-title {{ font-size: 0.9rem; opacity: 0.8; }}
    .kpi-value {{ font-size: 1.6rem; font-weight: 700; }}
    .kpi-sub {{ font-size: 0.7rem; opacity: 0.6; }}
    .netflix-title {{ font-size: 1.6rem; font-weight: 700; }}
    .stTabs [aria-selected="true"] {{
        background: {NETFLIX_RED}11;
        border-bottom: 3px solid {NETFLIX_RED};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------
# LOAD DATA (exploded version)
# ---------------------------------------------------------------------
@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)

    # your columns (exploded): country, cast, listed_in, director
    # parse dates
    if "date_added" in df.columns:
        df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
        df["year_added"] = df["date_added"].dt.year
        df["month_added"] = df["date_added"].dt.month
    else:
        df["year_added"] = np.nan
        df["month_added"] = np.nan

    # we already have atomic country, so just clean
    if "country" in df.columns:
        df["country_clean"] = df["country"].fillna("Unknown").str.strip()
    else:
        df["country_clean"] = "Unknown"

    # we already have atomic listed_in (1 row = 1 category)
    if "listed_in" in df.columns:
        df["category_clean"] = df["listed_in"].fillna("Unknown").str.strip()
    else:
        df["category_clean"] = "Unknown"

    # to avoid overcounting, create a deduped titles dataframe
    # (one row per show_id)
    titles_df = (
        df.sort_values("show_id")
          .drop_duplicates(subset=["show_id"])
          .reset_index(drop=True)
    )

    return df, titles_df

# change file name if needed
df_exp, titles_df = load_data("/Users/dakshj/Desktop/IIT KGP/Semesters/Sem 5/Open IIT DATA/final_cleaned_main.csv")

# ---------------------------------------------------------------------
# SIDEBAR FILTERS (these must work on exploded df)
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎬 Global Filters")

    # years (from titles level)
    years = titles_df["year_added"].dropna().astype(int).sort_values().unique().tolist()
    year_filter = st.multiselect(
        "Year added",
        years,
        default=years if years else []
    )

    # type
    type_opts = titles_df["type"].dropna().unique().tolist()
    type_sel = st.multiselect(
        "Type",
        sorted(type_opts),
        default=sorted(type_opts)
    )

    # country (exploded → easy)
    country_opts = ["All"] + sorted(df_exp["country_clean"].dropna().unique().tolist())
    country_sel = st.selectbox("Country", country_opts, index=0)

    # genre/category (exploded)
    genre_opts = ["All"] + sorted(df_exp["category_clean"].dropna().unique().tolist())
    genre_sel = st.selectbox("Category / Genre", genre_opts, index=0)

    st.markdown("---")
    st.markdown("Export current view ↓")
    dl_placeholder = st.empty()


def apply_filters(df_exp: pd.DataFrame, titles_df: pd.DataFrame):
    """
    We will:
    1. Filter exploded df on country/category
    2. From there, limit to show_ids that satisfy year + type (from titles_df)
    This avoids double counting and keeps logic clean.
    """
    dfe = df_exp.copy()

    # 1) explode-level filters
    if country_sel != "All":
        dfe = dfe[dfe["country_clean"] == country_sel]
    if genre_sel != "All":
        dfe = dfe[dfe["category_clean"] == genre_sel]

    # 2) now enforce year + type using titles_df
    td = titles_df.copy()
    if year_filter:
        td = td[td["year_added"].isin(year_filter)]
    if type_sel:
        td = td[td["type"].isin(type_sel)]

    valid_ids = set(td["show_id"].tolist())
    dfe = dfe[dfe["show_id"].isin(valid_ids)]

    # ALSO produce a deduped titles view of THIS filtered exploded df
    titles_filtered = (
        dfe.sort_values("show_id")
           .drop_duplicates(subset=["show_id"])
           .reset_index(drop=True)
    )

    return dfe, titles_filtered


df_f_exp, titles_f = apply_filters(df_exp, titles_df)

with st.sidebar:
    dl_placeholder.download_button(
        "⬇️ Download filtered titles",
        data=titles_f.to_csv(index=False).encode("utf-8"),
        file_name="netflix_filtered_titles.csv",
        mime="text/csv",
        use_container_width=True
    )


# ---------------------------------------------------------------------
# KPI helper
# ---------------------------------------------------------------------
def kpi_card(title, value, sub=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------------------
# TABS (7)
# ---------------------------------------------------------------------
tabs = st.tabs([
    "1. Executive Overview",
    "2. Content Explorer",
    "3. Trend Intelligence",
    "4. Geographic Insights",
    "5. Genre & Category Intelligence",
    "6. Creator & Talent Hub",
    "7. Strategic Recommendations",
])

# =========================================================
# TAB 1
# =========================================================
with tabs[0]:
    st.markdown('<p class="netflix-title">Executive Overview</p>', unsafe_allow_html=True)
    st.markdown("High-level KPIs based on **unique titles** .")

    total_titles = len(titles_f)
    movie_ct = (titles_f["type"] == "Movie").sum()
    tv_ct = (titles_f["type"] == "TV Show").sum()

    # growth calc: based on titles_f
    if titles_f["year_added"].notna().any():
        last_year = int(titles_f["year_added"].max())
        prev_year = last_year - 1
        ly_ct = titles_f.loc[titles_f["year_added"] == last_year, "show_id"].nunique()
        py_ct = titles_f.loc[titles_f["year_added"] == prev_year, "show_id"].nunique()
        if py_ct == 0:
            growth = 0.0
        else:
            growth = (ly_ct - py_ct) / py_ct * 100
    else:
        growth = 0.0

    # geo diversity → from exploded filtered
    geo_div = df_f_exp["country_clean"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Titles", f"{total_titles:,}", "Unique show_id")
    with c2:
        kpi_card("Movies / TV", f"{movie_ct} / {tv_ct}", "")
    with c3:
        kpi_card("Growth vs LY", f"{growth:,.1f}%", "based on year_added")
    with c4:
        kpi_card("Countries", geo_div, "content breadth")

    st.markdown("### Catalog Composition")
    left, right = st.columns([1.2, 1])

    with left:
        type_counts = titles_f["type"].value_counts().reset_index()
        fig = px.pie(
            type_counts,
            names="type",
            values="count",
            color="type",
            color_discrete_map={"Movie": NETFLIX_RED, "TV Show": "#000000"},
            hole=0.45,
            title="Movies vs TV Shows"
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        top_genres = (
            df_f_exp.groupby("category_clean")["show_id"]
            .nunique()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
            .rename(columns={"show_id": "title_count"})
        )
        fig2 = px.bar(
            top_genres,
            x="title_count",
            y="category_clean",
            orientation="h",
            title="Top Categories (unique titles)",
            color="title_count",
            color_continuous_scale=["#2a2a2a", NETFLIX_RED],
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            yaxis=dict(categoryorder="total ascending"),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Key Highlights")
    st.markdown(
        "- Netflix’s catalog consistently favors movies over TV shows.  \n"
        "- Core Content Stack: International (21.7%), Drama (21.4%), and Comedy (13.9%) make up 57% of the catalog,showing a mainstream, global-first strategy.  \n"
        "- Room to Grow: Kids/family balances mature content, but low-share genres (docs, sci-fi, anime, sports <3%) are clear expansion targets."
    )


# =========================================================
# TAB 2: CONTENT EXPLORER
# =========================================================
with tabs[1]:
    st.markdown('<p class="netflix-title">Content Explorer</p>', unsafe_allow_html=True)
    st.markdown("Search & export **unique titles**. (Exploded rows → grouped back.)")

    col1, col2, col3 = st.columns([1.4, 1, 1])
    with col1:
        q = st.text_input("🔎 Search title / director / cast / desc", "")
    with col2:
        sort_col = st.selectbox(
            "Sort by",
            ["title", "type", "year_added", "release_year", "rating", "country_clean"],
            index=0
        )
    with col3:
        asc = st.toggle("Ascending", True)

    # we will use titles_f (deduped) for explorer
    explorer = titles_f.copy()

    if q:
        ql = q.lower()
        explorer = explorer[
            explorer["title"].str.lower().str.contains(ql, na=False)
            | explorer["director"].fillna("").str.lower().str.contains(ql)
            | explorer["cast"].fillna("").str.lower().str.contains(ql)
            | explorer["description"].fillna("").str.lower().str.contains(ql)
        ]

    if sort_col in explorer.columns:
        explorer = explorer.sort_values(by=sort_col, ascending=asc)

    st.dataframe(
        explorer[
            [
                "show_id", "title", "type", "country", "year_added",
                "release_year", "rating", "listed_in", "director", "cast"
            ]
        ].reset_index(drop=True),
        use_container_width=True,
        height=470
    )

    st.download_button(
        "⬇️ Export table",
        data=explorer.to_csv(index=False).encode("utf-8"),
        file_name="content_explorer.csv",
        mime="text/csv"
    )

# =========================================================
# TAB 3: TREND INTELLIGENCE
# =========================================================
with tabs[2]:
    st.markdown('<p class="netflix-title">Trend Intelligence</p>', unsafe_allow_html=True)
    st.markdown("Time-series based on **unique titles** (not exploded).")

    if titles_f["year_added"].notna().any():
        trend = (
            titles_f.groupby("year_added")["show_id"]
            .nunique()
            .reset_index(name="titles")
            .sort_values("year_added")
        )
        fig = px.line(
            trend, x="year_added", y="titles",
            markers=True,
            title="Titles Added by Year",
        )
        fig.update_traces(line_color=NETFLIX_RED)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            xaxis_title="Year",
            yaxis_title="No. of Titles"
        )
        st.plotly_chart(fig, use_container_width=True)

        # by type over time
        trend_type = (
            titles_f.groupby(["year_added", "type"])["show_id"]
            .nunique()
            .reset_index(name="count")
            .sort_values(["year_added", "type"])
        )
        fig2 = px.area(
            trend_type,
            x="year_added",
            y="count",
            color="type",
            title="Movies vs TV Shows over Time",
            color_discrete_map={"Movie": NETFLIX_RED, "TV Show": "#000000"}
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
        )
        st.plotly_chart(fig2, use_container_width=True)

        # month pattern from titles_f
        if "month_added" in titles_f.columns:
            month_df = (
                titles_f.groupby("month_added")["show_id"]
                .nunique()
                .reset_index(name="titles")
                .sort_values("month_added")
            )
            fig3 = px.bar(
                month_df,
                x="month_added",
                y="titles",
                title="Additions by Month",
                color="titles",
                color_continuous_scale=["#1f1f1f", NETFLIX_RED]
            )
            fig3.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
            )
            st.plotly_chart(fig3, use_container_width=True)

    else:
        st.info("No year_added info to build trend.")

# =========================================================
# TAB 4: GEOGRAPHIC INSIGHTS
# =========================================================
with tabs[3]:
    st.markdown('<p class="netflix-title">Geographic Insights</p>', unsafe_allow_html=True)
    st.markdown("Exploded data → one row per (title, country) → perfect for geo counts.")

    geo = (
        df_f_exp.groupby("country_clean")["show_id"]
        .nunique()
        .reset_index(name="titles")
        .sort_values("titles", ascending=False)
    )

    top_n = st.slider("Top N countries", 5, 30, 15)
    fig = px.bar(
        geo.head(top_n),
        x="country_clean",
        y="titles",
        title="Top Content Countries (unique titles)",
        color="titles",
        color_continuous_scale=["#1f1f1f", NETFLIX_RED]
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        xaxis_title="Country",
        yaxis_title="Titles"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("*(Add choropleth here if you map to ISO codes.)*")

# =========================================================
# TAB 5: GENRE & CATEGORY INTELLIGENCE
# =========================================================
with tabs[4]:
    st.markdown('<p class="netflix-title">Genre & Category Intelligence</p>', unsafe_allow_html=True)

    # popularity from exploded (count unique titles per category)
    cat_pop = (
        df_f_exp.groupby("category_clean")["show_id"]
        .nunique()
        .reset_index(name="titles")
        .sort_values("titles", ascending=False)
    )
    fig = px.treemap(
        cat_pop,
        path=["category_clean"],
        values="titles",
        title="Category popularity (unique titles)",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, l=0, r=0, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Category Co-occurrence (rebuild from exploded)")
    # even though listed_in is exploded, we can group it back
    # each show_id → array of categories
    grouped_cats = (
        df_exp.groupby("show_id")["listed_in"]
        .apply(lambda x: sorted(set(x.dropna().tolist())))
    )

    # take top 15 cats overall (from cat_pop)
    top_cats = cat_pop.head(15)["category_clean"].tolist()
    # init matrix
    matrix = pd.DataFrame(0, index=top_cats, columns=top_cats, dtype=int)

    for cats in grouped_cats:
        # keep only top cats in this title
        cats = [c for c in cats if c in top_cats]
        for i in range(len(cats)):
            for j in range(i, len(cats)):
                ci, cj = cats[i], cats[j]
                matrix.loc[ci, cj] += 1
                if i != j:
                    matrix.loc[cj, ci] += 1

    fig_h = px.imshow(
        matrix,
        text_auto=True,
        aspect="auto",
        title="Top Category Co-occurrence Matrix",
        color_continuous_scale=["#1f1f1f", NETFLIX_RED]
    )
    fig_h.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
    )
    st.plotly_chart(fig_h, use_container_width=True)

    # quick "opportunity" list
    med = cat_pop["titles"].median()
    rare = cat_pop[cat_pop["titles"] < med]
    st.markdown("#### Opportunity (below-median categories)")
    st.dataframe(rare, use_container_width=True, height=220)

# =========================================================
# TAB 6: CREATOR & TALENT HUB
# =========================================================
with tabs[5]:
    st.markdown('<p class="netflix-title">Creator & Talent Hub</p>', unsafe_allow_html=True)
    st.markdown("Your df is exploded → one row per (title, person) but some cells may still be list-like, so we cast to str.")

    q_person = st.text_input("🔍 Search director / cast name", "")

    # 1) make sure these 2 columns are plain strings, not lists / objects
    df_dir_safe = df_f_exp.copy()
    df_dir_safe["director"] = df_dir_safe["director"].astype(str).str.strip()
    df_dir_safe["cast"] = df_dir_safe["cast"].astype(str).str.strip()

    # 2) Top directors (by occurrences in exploded df)
    top_dir = (
        df_dir_safe["director"]
        .fillna("Unknown")
        .value_counts()
        .head(15)
        .reset_index()
    )
    # after reset_index() the columns are ['index', 'director'] → rename to nicer names
    top_dir.columns = ["director", "count"]

    fig_d = px.bar(
        top_dir,
        x="count",
        y="director",
        orientation="h",
        title="Top 15 Directors",
        color="count",
        color_continuous_scale=["#1f1f1f", NETFLIX_RED],
    )
    fig_d.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        yaxis=dict(categoryorder="total ascending"),
    )
    st.plotly_chart(fig_d, use_container_width=True)

    # 3) Talent search
    if q_person:
        ql = q_person.lower()

        df_search = df_exp.copy()
        df_search["director"] = df_search["director"].astype(str)
        df_search["cast"] = df_search["cast"].astype(str)

        mask = (
            df_search["director"].str.lower().str.contains(ql, na=False)
            | df_search["cast"].str.lower().str.contains(ql, na=False)
        )
        talent_rows = df_search[mask]

        # dedupe by show_id because exploded
        talent_titles = (
            talent_rows.sort_values("show_id")
                        .drop_duplicates(subset=["show_id"])
        )

        st.markdown(f"**Portfolio for `{q_person}`** ({len(talent_titles)} unique titles)")
        st.dataframe(
            talent_titles[
                ["show_id", "title", "type", "year_added", "release_year",
                 "rating", "country", "listed_in"]
            ],
            use_container_width=True,
            height=300
        )
    else:
        st.info("Search an actor/director above to see all titles they appear in.")
# =========================================================
# TAB 7: STRATEGIC RECOMMENDATIONS
# =========================================================
with tabs[6]:
    st.markdown('<p class="netflix-title">Strategic Recommendations</p>', unsafe_allow_html=True)
    st.markdown(
        "- **Exploded-aware KPIs**: always dedupe at `show_id` for exec view.  \n"
        "- **Geo gaps**: check Tab 4 → prioritize low-volume countries with high strategic value.  \n"
        "- **Genre gaps**: check Tab 5 → below-median categories = diversification picks.  \n"
        "- **Talent**: Tab 6 → repeat names → consider multi-title campaigns.  \n"
        "- **Dashboarding**: keep dark / Netflix-like UI for stakeholder demo."
    )
