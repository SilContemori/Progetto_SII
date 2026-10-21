
from pathlib import Path
import os
import re
import urllib.request
import zipfile

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CONSTANTS
# ============================================================

DATA_DIR = Path("ml-latest-small")
ZIP_PATH = Path("ml-latest-small.zip")
DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

MIN_RATINGS_PER_MOVIE = 10
MIN_COMMON_USERS = 5
K_NEIGHBORS = 30
CONTENT_LIKE_THRESHOLD = 4.0
GENRE_WEIGHT = 1.5
TAG_WEIGHT = 1.0

TMDB_BEARER_TOKEN = os.getenv("TMDB_BEARER_TOKEN", "").strip()
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
POSTER_PLACEHOLDER = "https://placehold.co/500x750/171923/E6E8EF?text=No+Poster"


# ============================================================
# FIXED VISUAL STYLE — NO SIDEBAR, NO THEME MENU
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --bg: #0B1020;
        --panel: #111827;
        --panel-2: #151D30;
        --text: #F7F8FC;
        --muted: #9AA6BD;
        --accent: #8B6BFF;
        --accent2: #2BC4D9;
        --border: rgba(255,255,255,.08);
    }

    .stApp {
        background:
          radial-gradient(circle at 8% 0%, rgba(139,107,255,.13), transparent 30rem),
          radial-gradient(circle at 94% 0%, rgba(43,196,217,.08), transparent 27rem),
          var(--bg);
        color: var(--text);
    }

    .block-container {
        max-width: 1420px;
        padding-top: 1.25rem;
        padding-bottom: 4rem;
    }

    /* Force-hide Streamlit chrome/theme controls */
    [data-testid="stHeader"] {
        display: none !important;
        height: 0 !important;
    }
    [data-testid="stToolbar"] {
        display: none !important;
    }
    [data-testid="stSidebar"] {
        display: none !important;
    }
    #MainMenu {
        visibility: hidden !important;
        display: none !important;
    }
    footer {
        visibility: hidden !important;
        display: none !important;
    }

    /* Header */
    .topline {
        color: #C7BCFF;
        text-transform: uppercase;
        font-size: .74rem;
        letter-spacing: .14em;
        font-weight: 800;
        margin-bottom: .3rem;
    }

    .brand {
        font-size: clamp(2.2rem, 5vw, 4rem);
        font-weight: 900;
        letter-spacing: -.055em;
        line-height: .98;
        margin: 0;
        color: white;
    }

    .tagline {
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.55;
        max-width: 920px;
        margin-top: .7rem;
        margin-bottom: 1.2rem;
    }

    .version-badge {
        display: inline-block;
        font-size: .7rem;
        font-weight: 800;
        color: #B9F5FB;
        border: 1px solid rgba(43,196,217,.25);
        background: rgba(43,196,217,.08);
        padding: .22rem .45rem;
        border-radius: 999px;
        margin-left: .45rem;
        vertical-align: middle;
    }

    .control-strip {
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: .8rem .95rem .3rem .95rem;
        background: rgba(17,24,39,.76);
        box-shadow: 0 12px 30px rgba(0,0,0,.14);
        margin-bottom: 1.1rem;
    }

    .section-title {
        color: white;
        font-size: 1.45rem;
        font-weight: 850;
        margin: .1rem 0 .2rem 0;
    }

    .section-sub {
        color: var(--muted);
        font-size: .94rem;
        line-height: 1.5;
        margin-bottom: .85rem;
    }

    .step {
        display: inline-block;
        color: #D1C8FF;
        background: rgba(139,107,255,.10);
        border: 1px solid rgba(139,107,255,.28);
        border-radius: 999px;
        padding: .22rem .52rem;
        font-size: .72rem;
        font-weight: 800;
        margin-bottom: .4rem;
    }

    /* Recommendation cards: intentionally compact */
    .card-title {
        color: white;
        font-size: .92rem;
        font-weight: 800;
        line-height: 1.25;
        min-height: 2.3rem;
        margin-top: .35rem;
        margin-bottom: .15rem;
    }

    .card-meta {
        color: var(--muted);
        font-size: .74rem;
        line-height: 1.35;
        min-height: 2rem;
        margin-bottom: .4rem;
    }

    .rank-chip {
        display: inline-block;
        color: #D8D2FF;
        background: rgba(139,107,255,.11);
        border: 1px solid rgba(139,107,255,.25);
        border-radius: 999px;
        padding: .16rem .42rem;
        font-size: .7rem;
        font-weight: 800;
        margin-right: .3rem;
    }

    .score-chip {
        display: inline-block;
        color: #AEEFF6;
        background: rgba(43,196,217,.08);
        border: 1px solid rgba(43,196,217,.20);
        border-radius: 999px;
        padding: .16rem .42rem;
        font-size: .7rem;
        font-weight: 800;
    }

    /* User-profile panel */
    .profile-intro {
        border-left: 3px solid var(--accent);
        padding-left: .85rem;
        color: var(--muted);
        margin-bottom: 1rem;
    }

    div[data-testid="stButton"] > button,
    div[data-testid="stPopover"] > button {
        border-radius: 12px !important;
        font-weight: 750 !important;
    }

    div[data-testid="stTabs"] button {
        font-weight: 750;
    }

    /* Make charts visually integrated */
    [data-testid="stVegaLiteChart"] {
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: .35rem;
        background: rgba(17,24,39,.35);
    }

    @media (max-width: 900px) {
        .block-container {
            padding-left: .8rem;
            padding-right: .8rem;
        }
        .brand {
            font-size: 2.5rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA
# ============================================================

def ensure_dataset():
    if DATA_DIR.exists():
        return

    with st.spinner("Scarico MovieLens Latest Small..."):
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(".")


@st.cache_data(show_spinner=False)
def load_data():
    ensure_dataset()

    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    movies = pd.read_csv(DATA_DIR / "movies.csv")
    tags = pd.read_csv(DATA_DIR / "tags.csv")
    links = pd.read_csv(DATA_DIR / "links.csv")

    ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s")
    tags["tag_clean"] = tags["tag"].astype(str).str.lower().str.strip()

    return ratings, movies, tags, links


def clean_token(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


# ============================================================
# MODELS
# ============================================================

@st.cache_resource(show_spinner="Preparo i modelli...")
def build_models():
    ratings, movies, tags, links = load_data()

    # -------- Content-Based --------
    movies_content = movies[["movieId", "title", "genres"]].copy()

    movies_content["genres_text"] = (
        movies_content["genres"]
        .str.split("|")
        .apply(
            lambda vals: " ".join(
                clean_token(v)
                for v in vals
                if v != "(no genres listed)"
            )
        )
    )

    tag_documents = (
        tags.assign(tag_token=tags["tag_clean"].map(clean_token))
        .groupby("movieId")["tag_token"]
        .apply(lambda x: " ".join(x))
        .rename("tags_text")
        .reset_index()
    )

    movies_content = movies_content.merge(
        tag_documents,
        on="movieId",
        how="left",
    )
    movies_content["tags_text"] = movies_content["tags_text"].fillna("")

    genre_vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    tag_vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        min_df=2,
    )

    genre_matrix = genre_vectorizer.fit_transform(
        movies_content["genres_text"]
    )
    tag_matrix = tag_vectorizer.fit_transform(
        movies_content["tags_text"]
    )

    content_matrix = hstack(
        [
            genre_matrix * GENRE_WEIGHT,
            tag_matrix * TAG_WEIGHT,
        ],
        format="csr",
    )

    content_movie_ids = movies_content["movieId"].to_numpy()

    content_movie_to_pos = {
        int(mid): pos
        for pos, mid in enumerate(content_movie_ids)
    }

    # -------- Item-Based --------
    movie_counts = ratings.groupby("movieId").size()

    eligible_movies = movie_counts[
        movie_counts >= MIN_RATINGS_PER_MOVIE
    ].index

    item_ratings = ratings[
        ratings["movieId"].isin(eligible_movies)
    ].copy()

    user_item = item_ratings.pivot(
        index="userId",
        columns="movieId",
        values="rating",
    )

    centered = user_item.sub(
        user_item.mean(axis=1),
        axis=0,
    )

    item_movie_ids = user_item.columns.to_numpy()

    item_user_matrix = csr_matrix(
        centered.T
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )

    binary_item_user = csr_matrix(
        user_item.notna()
        .T
        .to_numpy(dtype=np.int8)
    )

    item_movie_to_pos = {
        int(mid): pos
        for pos, mid in enumerate(item_movie_ids)
    }

    tmdb_map = (
        links.dropna(subset=["tmdbId"])
        .assign(tmdbId=lambda x: x["tmdbId"].astype(int))
        .set_index("movieId")["tmdbId"]
        .to_dict()
    )

    return {
        "movie_lookup": movies.set_index("movieId"),
        "content_matrix": content_matrix,
        "content_movie_ids": content_movie_ids,
        "content_movie_to_pos": content_movie_to_pos,
        "item_movie_ids": item_movie_ids,
        "item_movie_to_pos": item_movie_to_pos,
        "item_user_matrix": item_user_matrix,
        "binary_item_user": binary_item_user,
        "user_item": user_item,
        "tmdb_map": tmdb_map,
    }


ratings, movies, tags, links = load_data()
models = build_models()
movie_lookup = models["movie_lookup"]


# ============================================================
# TMDB
# ============================================================

def tmdb_enabled():
    return bool(TMDB_BEARER_TOKEN or TMDB_API_KEY)


def tmdb_headers():
    headers = {"accept": "application/json"}
    if TMDB_BEARER_TOKEN:
        headers["Authorization"] = f"Bearer {TMDB_BEARER_TOKEN}"
    return headers


def tmdb_params(language="it-IT"):
    params = {"language": language}
    if TMDB_API_KEY and not TMDB_BEARER_TOKEN:
        params["api_key"] = TMDB_API_KEY
    return params


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_tmdb_movie(movie_id):
    tmdb_id = models["tmdb_map"].get(int(movie_id))

    fallback = {
        "poster_url": POSTER_PLACEHOLDER,
        "overview": "Trama non disponibile.",
        "release_year": "",
        "tmdb_rating": None,
    }

    if not tmdb_id or not tmdb_enabled():
        return fallback

    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"

    try:
        response = requests.get(
            url,
            headers=tmdb_headers(),
            params=tmdb_params("it-IT"),
            timeout=8,
        )

        if response.status_code != 200:
            return fallback

        data = response.json()
        overview = (data.get("overview") or "").strip()

        if not overview:
            response_en = requests.get(
                url,
                headers=tmdb_headers(),
                params=tmdb_params("en-US"),
                timeout=8,
            )
            if response_en.status_code == 200:
                overview = (
                    response_en.json().get("overview")
                    or ""
                ).strip()

        poster_path = data.get("poster_path")
        poster_url = (
            f"{TMDB_IMAGE_BASE}{poster_path}"
            if poster_path
            else POSTER_PLACEHOLDER
        )

        release_date = data.get("release_date") or ""
        year = release_date[:4] if release_date else ""

        tmdb_rating = data.get("vote_average")
        if tmdb_rating in (None, 0):
            tmdb_rating = None

        return {
            "poster_url": poster_url,
            "overview": overview or "Trama non disponibile.",
            "release_year": year,
            "tmdb_rating": tmdb_rating,
        }

    except requests.RequestException:
        return fallback


# ============================================================
# RECOMMENDER CORE
# ============================================================

def profile_from_existing_user(user_id):
    user_df = ratings[
        ratings["userId"] == user_id
    ]

    return {
        int(row.movieId): float(row.rating)
        for row in user_df.itertuples()
    }


def content_scores(profile):
    if not profile:
        return None

    s = pd.Series(profile, dtype=float)
    user_mean = s.mean()

    liked = s[
        s >= CONTENT_LIKE_THRESHOLD
    ]

    liked = liked[
        liked.index.isin(
            models["content_movie_to_pos"]
        )
    ]

    if liked.empty:
        liked = s[s >= user_mean]
        liked = liked[
            liked.index.isin(
                models["content_movie_to_pos"]
            )
        ]

    if liked.empty:
        return None

    positions = np.array([
        models["content_movie_to_pos"][int(mid)]
        for mid in liked.index
    ])

    weights = np.maximum(
        liked.to_numpy(dtype=np.float32)
        - user_mean,
        0,
    )

    if weights.sum() == 0:
        weights = liked.to_numpy(dtype=np.float32)

    weighted = (
        models["content_matrix"][positions]
        .multiply(weights[:, None])
    )

    profile_vector = csr_matrix(
        weighted.sum(axis=0)
        / weights.sum()
    )

    return cosine_similarity(
        profile_vector,
        models["content_matrix"],
    ).ravel()


def item_scores(profile):
    if not profile:
        return None

    usable = {
        mid: rating
        for mid, rating in profile.items()
        if mid in models["item_movie_to_pos"]
    }

    if len(usable) < 2:
        return None

    seen_ids = np.array(
        list(usable.keys()),
        dtype=int,
    )

    seen_ratings = np.array(
        [usable[mid] for mid in seen_ids],
        dtype=np.float32,
    )

    seen_positions = np.array([
        models["item_movie_to_pos"][int(mid)]
        for mid in seen_ids
    ])

    user_mean = float(
        seen_ratings.mean()
    )

    deviations = (
        seen_ratings - user_mean
    )

    similarities = cosine_similarity(
        models["item_user_matrix"],
        models["item_user_matrix"][
            seen_positions
        ],
    ).astype(np.float32)

    common_users = (
        models["binary_item_user"]
        @ models["binary_item_user"][
            seen_positions
        ].T
    ).toarray()

    similarities[
        common_users < MIN_COMMON_USERS
    ] = 0.0

    for col, pos in enumerate(
        seen_positions
    ):
        similarities[pos, col] = 0.0

    if similarities.shape[1] > K_NEIGHBORS:
        top_idx = np.argpartition(
            np.abs(similarities),
            -K_NEIGHBORS,
            axis=1,
        )[:, -K_NEIGHBORS:]

        top_sims = np.take_along_axis(
            similarities,
            top_idx,
            axis=1,
        )

        top_devs = deviations[
            top_idx
        ]

        numerator = (
            top_sims * top_devs
        ).sum(axis=1)

        denominator = np.abs(
            top_sims
        ).sum(axis=1)

    else:
        numerator = (
            similarities * deviations
        ).sum(axis=1)

        denominator = np.abs(
            similarities
        ).sum(axis=1)

    predictions = np.full(
        len(models["item_movie_ids"]),
        np.nan,
        dtype=np.float32,
    )

    valid = denominator > 0

    predictions[valid] = (
        user_mean
        + numerator[valid]
        / denominator[valid]
    )

    return np.clip(
        predictions,
        0.5,
        5.0,
    )


def minmax(values):
    values = np.asarray(
        values,
        dtype=np.float32,
    )

    out = np.full(
        values.shape,
        np.nan,
        dtype=np.float32,
    )

    valid = np.isfinite(values)

    if not valid.any():
        return out

    v = values[valid]
    lo = v.min()
    hi = v.max()

    if np.isclose(lo, hi):
        out[valid] = 0.5
    else:
        out[valid] = (
            v - lo
        ) / (
            hi - lo
        )

    return out


def recommend(
    profile,
    model_name,
    n=10,
    alpha=0.5,
):
    seen = set(profile.keys())
    cb = content_scores(profile)
    cf = item_scores(profile)
    rows = []

    if model_name == "Content-Based":
        if cb is None:
            return pd.DataFrame()

        for pos, score in enumerate(cb):
            mid = int(
                models["content_movie_ids"][pos]
            )

            if mid in seen:
                continue

            rows.append({
                "movieId": mid,
                "title": movie_lookup.loc[
                    mid,
                    "title",
                ],
                "genres": movie_lookup.loc[
                    mid,
                    "genres",
                ],
                "score": float(score),
                "content_score": float(score),
                "cf_prediction": np.nan,
            })

    elif model_name == "Item-Based":
        if cf is None:
            return pd.DataFrame()

        for pos, score in enumerate(cf):
            if not np.isfinite(score):
                continue

            mid = int(
                models["item_movie_ids"][pos]
            )

            if mid in seen:
                continue

            rows.append({
                "movieId": mid,
                "title": movie_lookup.loc[
                    mid,
                    "title",
                ],
                "genres": movie_lookup.loc[
                    mid,
                    "genres",
                ],
                "score": float(score),
                "content_score": np.nan,
                "cf_prediction": float(score),
            })

    else:
        if cb is None or cf is None:
            return pd.DataFrame()

        cb_on_cf = np.array([
            cb[
                models["content_movie_to_pos"][
                    int(mid)
                ]
            ]
            for mid in models["item_movie_ids"]
        ], dtype=np.float32)

        hybrid = (
            alpha * minmax(cf)
            + (1 - alpha)
            * minmax(cb_on_cf)
        )

        for pos, score in enumerate(hybrid):
            if not np.isfinite(score):
                continue

            mid = int(
                models["item_movie_ids"][pos]
            )

            if mid in seen:
                continue

            rows.append({
                "movieId": mid,
                "title": movie_lookup.loc[
                    mid,
                    "title",
                ],
                "genres": movie_lookup.loc[
                    mid,
                    "genres",
                ],
                "score": float(score),
                "content_score": float(
                    cb_on_cf[pos]
                ),
                "cf_prediction": float(
                    cf[pos]
                ),
            })

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    return (
        result
        .sort_values(
            "score",
            ascending=False,
        )
        .head(n)
        .reset_index(drop=True)
    )


def similar_movies(
    movie_id,
    mode="Content-Based",
    n=8,
):
    if mode == "Content-Based":
        if movie_id not in models[
            "content_movie_to_pos"
        ]:
            return pd.DataFrame()

        pos = models[
            "content_movie_to_pos"
        ][movie_id]

        sims = cosine_similarity(
            models["content_matrix"][pos],
            models["content_matrix"],
        ).ravel()

        sims[pos] = -1

        order = np.argsort(
            sims
        )[::-1][:n]

        return pd.DataFrame([
            {
                "movieId": int(
                    models[
                        "content_movie_ids"
                    ][p]
                ),
                "title": movie_lookup.loc[
                    int(
                        models[
                            "content_movie_ids"
                        ][p]
                    ),
                    "title",
                ],
                "genres": movie_lookup.loc[
                    int(
                        models[
                            "content_movie_ids"
                        ][p]
                    ),
                    "genres",
                ],
                "score": float(
                    sims[p]
                ),
                "content_score": float(
                    sims[p]
                ),
                "cf_prediction": np.nan,
            }
            for p in order
        ])

    if movie_id not in models[
        "item_movie_to_pos"
    ]:
        return pd.DataFrame()

    pos = models[
        "item_movie_to_pos"
    ][movie_id]

    sims = cosine_similarity(
        models["item_user_matrix"],
        models["item_user_matrix"][pos],
    ).ravel()

    common = (
        models["binary_item_user"]
        @ models[
            "binary_item_user"
        ][pos].T
    ).toarray().ravel()

    sims[
        common < MIN_COMMON_USERS
    ] = 0.0

    sims[pos] = -1

    rows = []

    for p in np.argsort(
        sims
    )[::-1]:
        if sims[p] <= 0:
            continue

        mid = int(
            models[
                "item_movie_ids"
            ][p]
        )

        rows.append({
            "movieId": mid,
            "title": movie_lookup.loc[
                mid,
                "title",
            ],
            "genres": movie_lookup.loc[
                mid,
                "genres",
            ],
            "score": float(
                sims[p]
            ),
            "content_score": np.nan,
            "cf_prediction": float(
                sims[p]
            ),
        })

        if len(rows) == n:
            break

    return pd.DataFrame(rows)


# ============================================================
# PRESENTATION HELPERS
# ============================================================

title_to_id = {
    f"{row.title} · ID {row.movieId}":
        int(row.movieId)
    for row in movies.itertuples()
}

display_to_title = list(
    title_to_id.keys()
)


def compact_genres(
    genres,
    max_items=3,
):
    parts = [
        g
        for g in str(
            genres
        ).split("|")
        if g
    ]

    if len(parts) <= max_items:
        return " · ".join(parts)

    return (
        " · ".join(
            parts[:max_items]
        )
        + "…"
    )


def score_label(
    row,
    model_name,
):
    if model_name == "Hybrid":
        return (
            f"Hybrid {row['score']:.3f}"
        )

    if model_name == "Item-Based":
        return (
            f"Pred. {row['score']:.2f}/5"
        )

    return (
        f"Content {row['score']:.3f}"
    )


def details_reason(
    row,
    model_name,
):
    if model_name == "Hybrid":
        return (
            "Il ranking finale combina il segnale "
            f"collaborativo (CF {row['cf_prediction']:.2f}) "
            f"e quello di contenuto (CB {row['content_score']:.3f})."
        )

    if model_name == "Item-Based":
        return (
            "Il film è stato posizionato in alto perché "
            "presenta un pattern di rating compatibile con "
            "quelli dei film già valutati dall'utente."
        )

    return (
        "Il film è vicino al profilo dell'utente nello "
        "spazio costruito a partire da generi e tag."
    )


def render_movie_grid(
    df,
    model_name,
    columns=5,
):
    if df.empty:
        st.warning(
            "Nessuna raccomandazione disponibile "
            "per questo profilo/modello."
        )
        return

    for start in range(
        0,
        len(df),
        columns,
    ):
        subset = df.iloc[
            start:start + columns
        ]

        cols = st.columns(
            columns
        )

        for col, (idx, row) in zip(
            cols,
            subset.iterrows(),
        ):
            details = fetch_tmdb_movie(
                int(row["movieId"])
            )

            with col:
                st.image(
                    details["poster_url"],
                    use_container_width=True,
                )

                st.markdown(
                    f"<div class='card-title'>"
                    f"<span class='rank-chip'>#{idx + 1}</span>"
                    f"{row['title']}</div>",
                    unsafe_allow_html=True,
                )

                meta_parts = []

                if details[
                    "release_year"
                ]:
                    meta_parts.append(
                        details[
                            "release_year"
                        ]
                    )

                meta_parts.append(
                    compact_genres(
                        row["genres"]
                    )
                )

                st.markdown(
                    f"<div class='card-meta'>"
                    f"{' · '.join(meta_parts)}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"<span class='score-chip'>"
                    f"{score_label(row, model_name)}"
                    f"</span>",
                    unsafe_allow_html=True,
                )

                with st.popover(
                    "Apri scheda",
                    use_container_width=True,
                ):
                    p1, p2 = st.columns(
                        [1, 1.25]
                    )

                    with p1:
                        st.image(
                            details[
                                "poster_url"
                            ],
                            use_container_width=True,
                        )

                    with p2:
                        st.markdown(
                            f"### {row['title']}"
                        )

                        if details[
                            "tmdb_rating"
                        ] is not None:
                            st.caption(
                                f"TMDb "
                                f"{details['tmdb_rating']:.1f}/10"
                            )

                        st.caption(
                            compact_genres(
                                row["genres"],
                                max_items=8,
                            )
                        )

                    st.markdown(
                        "**Trama**"
                    )

                    st.write(
                        details["overview"]
                    )

                    st.markdown(
                        "**Perché è in questa posizione**"
                    )

                    st.write(
                        details_reason(
                            row,
                            model_name,
                        )
                    )

                    if model_name == "Hybrid":
                        st.write(
                            f"**CF:** "
                            f"{row['cf_prediction']:.3f}"
                            f"  ·  **CB:** "
                            f"{row['content_score']:.3f}"
                            f"  ·  **Hybrid:** "
                            f"{row['score']:.3f}"
                        )


def user_charts(
    history,
):
    # 1. Distribution by rating
    rating_dist = (
        history[
            "rating"
        ]
        .value_counts()
        .sort_index()
        .rename_axis(
            "rating"
        )
        .reset_index(
            name="count"
        )
    )

    rating_dist[
        "percent"
    ] = (
        rating_dist["count"]
        / rating_dist[
            "count"
        ].sum()
        * 100
    )

    rating_dist[
        "label"
    ] = rating_dist.apply(
        lambda r:
            f"{int(r['count'])} "
            f"({r['percent']:.1f}%)",
        axis=1,
    )

    chart_rating = (
        alt.Chart(
            rating_dist
        )
        .mark_bar(
            cornerRadiusTopLeft=6,
            cornerRadiusTopRight=6,
        )
        .encode(
            x=alt.X(
                "rating:O",
                title="Rating",
                sort="ascending",
            ),
            y=alt.Y(
                "count:Q",
                title="Numero di valutazioni",
            ),
            tooltip=[
                alt.Tooltip(
                    "rating:O",
                    title="Rating",
                ),
                alt.Tooltip(
                    "count:Q",
                    title="Valutazioni",
                ),
                alt.Tooltip(
                    "percent:Q",
                    title="Percentuale",
                    format=".1f",
                ),
            ],
        )
        .properties(
            height=280,
            title="Distribuzione dei rating",
        )
    )

    labels = (
        alt.Chart(
            rating_dist
        )
        .mark_text(
            dy=-9,
            fontSize=11,
        )
        .encode(
            x=alt.X(
                "rating:O",
                sort="ascending",
            ),
            y="count:Q",
            text="label:N",
        )
    )

    # 2. Like / neutral / dislike donut
    buckets = pd.DataFrame({
        "categoria": [
            "Apprezzati (≥4)",
            "Neutri (3–3.5)",
            "Bassi (≤2.5)",
        ],
        "count": [
            int(
                (
                    history["rating"]
                    >= 4.0
                ).sum()
            ),
            int(
                (
                    (
                        history[
                            "rating"
                        ] >= 3.0
                    )
                    & (
                        history[
                            "rating"
                        ] < 4.0
                    )
                ).sum()
            ),
            int(
                (
                    history["rating"]
                    < 3.0
                ).sum()
            ),
        ],
    })

    donut = (
        alt.Chart(
            buckets
        )
        .mark_arc(
            innerRadius=72,
            outerRadius=118,
        )
        .encode(
            theta=alt.Theta(
                "count:Q",
            ),
            color=alt.Color(
                "categoria:N",
                legend=alt.Legend(
                    title=None,
                    orient="bottom",
                ),
            ),
            tooltip=[
                "categoria:N",
                "count:Q",
            ],
        )
        .properties(
            height=280,
            title="Come usa la scala di rating",
        )
    )

    # 3. Average rating by genre
    genre_df = (
        history
        .assign(
            genre_list=
                history[
                    "genres"
                ].str.split("|")
        )
        .explode(
            "genre_list"
        )
    )

    genre_df = genre_df[
        genre_df[
            "genre_list"
        ] != "(no genres listed)"
    ]

    genre_stats = (
        genre_df
        .groupby(
            "genre_list"
        )
        .agg(
            mean_rating=(
                "rating",
                "mean",
            ),
            n=(
                "rating",
                "size",
            ),
        )
        .reset_index()
    )

    genre_stats = (
        genre_stats[
            genre_stats[
                "n"
            ] >= 3
        ]
        .sort_values(
            [
                "n",
                "mean_rating",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(10)
    )

    genre_chart = (
        alt.Chart(
            genre_stats
        )
        .mark_bar(
            cornerRadiusEnd=5,
        )
        .encode(
            x=alt.X(
                "mean_rating:Q",
                title="Rating medio",
                scale=alt.Scale(
                    domain=[0, 5],
                ),
            ),
            y=alt.Y(
                "genre_list:N",
                title=None,
                sort="-x",
            ),
            tooltip=[
                alt.Tooltip(
                    "genre_list:N",
                    title="Genere",
                ),
                alt.Tooltip(
                    "mean_rating:Q",
                    title="Rating medio",
                    format=".2f",
                ),
                alt.Tooltip(
                    "n:Q",
                    title="Valutazioni",
                ),
            ],
        )
        .properties(
            height=280,
            title="Rating medio per genere",
        )
    )

    c1, c2 = st.columns(
        [1.5, 1]
    )

    with c1:
        st.altair_chart(
            chart_rating + labels,
            use_container_width=True,
        )

    with c2:
        st.altair_chart(
            donut,
            use_container_width=True,
        )

    st.altair_chart(
        genre_chart,
        use_container_width=True,
    )


# ============================================================
# HEADER — NO BOX AROUND CINEMATCH
# ============================================================

st.markdown(
    """
    <div class="topline">
        Hybrid recommendation engine
        <span class="version-badge">UI V3</span>
    </div>
    <div class="brand">CineMatch</div>
    <div class="tagline">
        Sistema di raccomandazione cinematografico che combina
        Collaborative Filtering Item-Based e Content-Based Filtering.
        Usa un profilo manuale oppure un utente MovieLens per osservare
        come cambia il ranking prodotto dai diversi modelli.
    </div>
    """,
    unsafe_allow_html=True,
)

if tmdb_enabled():
    st.caption(
        "Poster e trama provengono da TMDb e servono solo alla visualizzazione: "
        "non entrano nel calcolo dello score."
    )
else:
    st.warning(
        "TMDb non configurato: il recommender funziona, ma poster e trama useranno placeholder."
    )


# ============================================================
# HORIZONTAL CONTROLS — SIDEBAR REMOVED
# ============================================================

st.markdown(
    "<div class='control-strip'>",
    unsafe_allow_html=True,
)

cc1, cc2, cc3 = st.columns(
    [1.35, 1, 1.15]
)

with cc1:
    model_name = st.selectbox(
        "Modello",
        [
            "Hybrid",
            "Item-Based",
            "Content-Based",
        ],
        index=0,
    )

with cc2:
    top_n = st.slider(
        "Top-N",
        min_value=5,
        max_value=15,
        value=10,
        step=1,
    )

with cc3:
    if model_name == "Hybrid":
        alpha = st.slider(
            "Peso Collaborative α",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
        )
    else:
        alpha = 0.5
        st.write("")
        st.caption(
            "Il peso α compare solo per il modello Hybrid."
        )

st.markdown(
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# TABS
# ============================================================

tab_manual, tab_user, tab_similar, tab_diag = st.tabs(
    [
        "✨ Prova il recommender",
        "👤 Utente MovieLens",
        "🔎 Film simili",
        "📊 Diagnostica",
    ]
)


# ============================================================
# MANUAL PROFILE
# ============================================================

with tab_manual:
    st.markdown(
        "<span class='step'>STEP 1</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='section-title'>Costruisci il profilo</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='section-sub'>"
        "Scegli almeno 5 film che conosci, includendo possibilmente "
        "sia titoli che ami sia titoli che non ti sono piaciuti."
        "</div>",
        unsafe_allow_html=True,
    )

    selected_labels = st.multiselect(
        "Film",
        options=display_to_title,
        placeholder="Cerca un film...",
        label_visibility="collapsed",
    )

    selected_count = len(
        selected_labels
    )

    st.progress(
        min(
            selected_count / 5,
            1.0,
        )
    )

    st.caption(
        f"{selected_count}/5 film minimi selezionati"
    )

    manual_profile = {}

    if selected_labels:
        st.markdown(
            "<span class='step'>STEP 2</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='section-title'>Assegna i rating</div>",
            unsafe_allow_html=True,
        )

        cols = st.columns(2)

        for i, label in enumerate(
            selected_labels
        ):
            movie_id = title_to_id[
                label
            ]

            with cols[
                i % 2
            ]:
                rating = st.select_slider(
                    movie_lookup.loc[
                        movie_id,
                        "title",
                    ],
                    options=[
                        0.5,
                        1.0,
                        1.5,
                        2.0,
                        2.5,
                        3.0,
                        3.5,
                        4.0,
                        4.5,
                        5.0,
                    ],
                    value=4.0,
                    key=f"manual_rating_{movie_id}",
                )

                manual_profile[
                    movie_id
                ] = float(
                    rating
                )

    if st.button(
        "Genera raccomandazioni",
        type="primary",
        use_container_width=True,
        disabled=(
            selected_count < 5
        ),
    ):
        with st.spinner(
            "Calcolo il ranking..."
        ):
            st.session_state[
                "manual_recs"
            ] = recommend(
                manual_profile,
                model_name,
                n=top_n,
                alpha=alpha,
            )

            st.session_state[
                "manual_model"
            ] = model_name

    if "manual_recs" in st.session_state:
        st.divider()

        st.markdown(
            "<div class='section-title'>Consigliati per te</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div class='section-sub'>"
            "Le card mostrano solo le informazioni essenziali. "
            "Apri la scheda del film per trama e dettagli."
            "</div>",
            unsafe_allow_html=True,
        )

        render_movie_grid(
            st.session_state[
                "manual_recs"
            ],
            st.session_state[
                "manual_model"
            ],
            columns=5,
        )


# ============================================================
# USER MOVIELENS
# ============================================================

with tab_user:
    st.markdown(
        "<div class='section-title'>Profilo utente MovieLens</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='section-sub'>"
        "Prima di generare le raccomandazioni, analizziamo come l'utente "
        "ha realmente utilizzato la scala dei rating."
        "</div>",
        unsafe_allow_html=True,
    )

    selected_user = st.selectbox(
        "User ID",
        sorted(
            ratings[
                "userId"
            ].unique()
        ),
    )

    user_history = (
        ratings[
            ratings[
                "userId"
            ] == selected_user
        ]
        .merge(
            movies[
                [
                    "movieId",
                    "title",
                    "genres",
                ]
            ],
            on="movieId",
            how="left",
        )
        .sort_values(
            "datetime"
        )
    )

    st.markdown(
        f"<div class='profile-intro'>"
        f"L'utente <b>{selected_user}</b> ha espresso "
        f"<b>{len(user_history)}</b> valutazioni con un rating medio di "
        f"<b>{user_history['rating'].mean():.2f}/5</b>."
        f"</div>",
        unsafe_allow_html=True,
    )

    user_charts(
        user_history
    )

    with st.expander(
        "Mostra tutte le valutazioni dell'utente"
    ):
        st.dataframe(
            user_history[
                [
                    "title",
                    "genres",
                    "rating",
                    "datetime",
                ]
            ]
            .sort_values(
                [
                    "rating",
                    "datetime",
                ],
                ascending=[
                    False,
                    False,
                ],
            ),
            use_container_width=True,
            hide_index=True,
            height=360,
        )

    st.divider()

    st.markdown(
        "<div class='section-title'>Raccomandazioni</div>",
        unsafe_allow_html=True,
    )

    if st.button(
        "Genera raccomandazioni per questo utente",
        type="primary",
        use_container_width=True,
    ):
        profile = profile_from_existing_user(
            selected_user
        )

        with st.spinner(
            "Calcolo il ranking..."
        ):
            st.session_state[
                "user_recs"
            ] = recommend(
                profile,
                model_name,
                n=top_n,
                alpha=alpha,
            )

            st.session_state[
                "user_model"
            ] = model_name

    if "user_recs" in st.session_state:
        st.markdown(
            "<div class='section-sub'>"
            "Ogni card è volutamente compatta; la trama completa "
            "è disponibile nel popup “Apri scheda”."
            "</div>",
            unsafe_allow_html=True,
        )

        render_movie_grid(
            st.session_state[
                "user_recs"
            ],
            st.session_state[
                "user_model"
            ],
            columns=5,
        )


# ============================================================
# SIMILAR MOVIES
# ============================================================

with tab_similar:
    st.markdown(
        "<div class='section-title'>Film simili</div>",
        unsafe_allow_html=True,
    )

    selected_movie_label = st.selectbox(
        "Film di partenza",
        display_to_title,
    )

    similar_mode = st.radio(
        "Similarità",
        [
            "Item-Based",
            "Content-Based",
        ],
        horizontal=True,
    )

    similar_n = st.slider(
        "Numero risultati",
        5,
        15,
        10,
        1,
    )

    movie_id = title_to_id[
        selected_movie_label
    ]

    if st.button(
        "Trova film simili",
        type="primary",
    ):
        st.session_state[
            "similar_recs"
        ] = similar_movies(
            movie_id,
            mode=similar_mode,
            n=similar_n,
        )

        st.session_state[
            "similar_mode"
        ] = similar_mode

    if "similar_recs" in st.session_state:
        render_movie_grid(
            st.session_state[
                "similar_recs"
            ],
            st.session_state[
                "similar_mode"
            ],
            columns=5,
        )


# ============================================================
# DIAGNOSTICS
# ============================================================

with tab_diag:
    st.markdown(
        "<div class='section-title'>Diagnostica</div>",
        unsafe_allow_html=True,
    )

    density = (
        models[
            "user_item"
        ]
        .notna()
        .sum()
        .sum()
        /
        (
            models[
                "user_item"
            ].shape[0]
            *
            models[
                "user_item"
            ].shape[1]
        )
    )

    st.write(
        f"**Utenti:** {ratings['userId'].nunique()}  ·  "
        f"**Film:** {movies['movieId'].nunique()}  ·  "
        f"**Rating:** {len(ratings):,}"
    )

    st.write(
        f"**Sparsity CF:** {1-density:.2%}  ·  "
        f"**Min rating/item:** {MIN_RATINGS_PER_MOVIE}  ·  "
        f"**Min utenti comuni:** {MIN_COMMON_USERS}"
    )

    dataset_dist = (
        ratings[
            "rating"
        ]
        .value_counts()
        .sort_index()
        .rename_axis(
            "rating"
        )
        .reset_index(
            name="count"
        )
    )

    chart = (
        alt.Chart(
            dataset_dist
        )
        .mark_bar(
            cornerRadiusTopLeft=6,
            cornerRadiusTopRight=6,
        )
        .encode(
            x=alt.X(
                "rating:O",
                title="Rating",
            ),
            y=alt.Y(
                "count:Q",
                title="Numero di valutazioni",
            ),
            tooltip=[
                "rating:O",
                "count:Q",
            ],
        )
        .properties(
            height=320,
            title="Distribuzione complessiva dei rating MovieLens",
        )
    )

    st.altair_chart(
        chart,
        use_container_width=True,
    )
