# cspell:words Segoe Roboto imdb tmdb selectbox Streamlit sklearn hashlib pandas movieId userId colnames fillna cosine similarity concat pivot dataframe
import hashlib
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import numpy as np

# Streamlit cache compatibility shim for older versions
cache_data_decorator = getattr(st, "cache_data", None) or getattr(st, "cache", None)
if cache_data_decorator is None:
    def cache_data_decorator(func):
        return func

# Custom CSS for professional look
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    html, body, [data-testid="stAppViewContainer"] {
        background: #0f0f0f;
        font-family: 'Segoe UI', 'Roboto', sans-serif !important;
    }
    
    [data-testid="stHeader"] {
        background: transparent !important;
    }
    
    .main-header {
        font-size: 3em;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-bottom: 30px;
        letter-spacing: 1px;
        text-shadow: 0 2px 10px rgba(255, 255, 255, 0.1);
    }
    
    .sub-header {
        font-size: 1.8em;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 3px solid #ff6b6b;
    }
    
    .movie-card {
        background: #1a1a1a;
        padding: 20px;
        margin: 15px 0;
        border-radius: 12px;
        border-left: 5px solid #ff6b6b;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease;
        color: #e0e0e0;
    }
    
    .movie-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 15px rgba(255, 107, 107, 0.3);
        border-left-color: #ff8787;
    }
    
    [data-testid="stSidebar"] {
        background: #000000 !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: white !important;
    }
    
    body {
        font-family: 'Segoe UI', 'Roboto', sans-serif;
        color: #e0e0e0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #ff6b6b 0%, #ff5252 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 12px 30px;
        font-weight: 600;
        font-size: 1em;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.5);
        background: linear-gradient(135deg, #ff8787 0%, #ff7070 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
    }
    
    .stTextInput > div > div > input {
        background-color: #1a1a1a;
        border: 2px solid #333333;
        border-radius: 8px;
        padding: 12px 15px;
        font-size: 1em;
        transition: all 0.3s ease;
        color: #e0e0e0;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #ff6b6b;
        box-shadow: 0 0 0 3px rgba(255, 107, 107, 0.2);
        background-color: #262626;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #666666;
    }
    
    [data-testid="stForm"] {
        background: #1a1a1a;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    
    .stInfo, .stSuccess, .stError, .stWarning {
        border-radius: 8px;
        padding: 15px 20px;
        border-left: 5px solid;
        background-color: #262626 !important;
    }
    
    .stInfo {
        border-left-color: #4dabf7;
        color: #a5d8ff;
    }
    
    .stSuccess {
        border-left-color: #51cf66;
        color: #b2f2bb;
    }
    
    .stError {
        border-left-color: #ff8787;
        color: #ffa8a8;
    }
    
    .stWarning {
        border-left-color: #ffd43b;
        color: #ffe066;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
        font-weight: 600;
    }
    
    [data-testid="stExpander"] {
        border: 2px solid #333333;
        border-radius: 8px;
        overflow: hidden;
        background-color: #1a1a1a;
    }
    
    [data-testid="stExpander"] > div > button {
        padding: 16px;
        font-weight: 600;
        background: #262626;
        color: #ffffff;
    }
    
    [data-testid="stExpander"] > div > button:hover {
        background: #333333;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2.5em;
        font-weight: 700;
        color: #ff6b6b;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1.1em;
        color: #a0a0a0;
        font-weight: 500;
    }
    
    [data-testid="stMarkdownContainer"] {
        color: #e0e0e0;
    }
    
    p, span, label, div {
        color: #e0e0e0;
    }
    
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
MOVIES_FILE = BASE_DIR / "movies.csv"
RATINGS_FILE = BASE_DIR / "ratings.csv"
LINKS_FILE = BASE_DIR / "links.csv"
USERS_FILE = BASE_DIR / "users.csv"


@cache_data_decorator
def load_data():
    """Load and process all data with caching"""
    # Initialize user storage
    if not USERS_FILE.exists():
        pd.DataFrame(columns=["username", "password"]).to_csv(USERS_FILE, index=False)
    
    users = pd.read_csv(USERS_FILE)
    
    # Load datasets with safety checks
    try:
        movies = pd.read_csv(MOVIES_FILE).fillna("")
    except FileNotFoundError:
        movies = pd.DataFrame(columns=["movieId", "title", "genres"])
    except Exception as e:
        st.error(f"Error loading movies: {e}")
        movies = pd.DataFrame(columns=["movieId", "title", "genres"])
    
    try:
        ratings = pd.read_csv(RATINGS_FILE)
    except FileNotFoundError:
        ratings = pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])
    except Exception as e:
        st.error(f"Error loading ratings: {e}")
        ratings = pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])
    
    try:
        links = pd.read_csv(LINKS_FILE).fillna("")
    except FileNotFoundError:
        links = pd.DataFrame(columns=["movieId", "imdbId", "tmdbId"])
    except Exception as e:
        st.error(f"Error loading links: {e}")
        links = pd.DataFrame(columns=["movieId", "imdbId", "tmdbId"])
    
    # Prepare rating statistics (will be empty if no ratings)
    if not ratings.empty:
        rating_stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
        rating_stats = rating_stats.rename(columns={"mean": "avg_rating", "count": "rating_count"})
    else:
        rating_stats = pd.DataFrame(columns=["movieId", "avg_rating", "rating_count"])
    
    # Merge movies with links and rating stats
    if not movies.empty:
        if not links.empty and "movieId" in links.columns:
            movies = movies.merge(links, on="movieId", how="left")
        
        if not rating_stats.empty and "movieId" in rating_stats.columns:
            movies = movies.merge(rating_stats, on="movieId", how="left")
        
        # Ensure avg_rating exists
        if "avg_rating" not in movies.columns:
            movies["avg_rating"] = 0
        movies["avg_rating"] = pd.to_numeric(movies["avg_rating"], errors="coerce").fillna(0).round(2)
        
        # Ensure rating_count exists
        if "rating_count" not in movies.columns:
            movies["rating_count"] = 0
        movies["rating_count"] = pd.to_numeric(movies["rating_count"], errors="coerce").fillna(0).astype(int)
    else:
        movies = pd.DataFrame(columns=["movieId", "title", "genres", "avg_rating", "rating_count"])
    
    # Merge movies and ratings on movieId for collaborative filtering
    if not ratings.empty and not movies.empty and "movieId" in ratings.columns:
        movie_ratings = pd.merge(ratings, movies[["movieId", "title"]], on="movieId")
    else:
        movie_ratings = pd.DataFrame(columns=["userId", "movieId", "rating", "title"])
    
    # Create user-item matrix
    if not movie_ratings.empty and len(movie_ratings) > 0:
        try:
            user_item_matrix = movie_ratings.pivot_table(index="userId", columns="title", values="rating").fillna(0)
        except Exception as e:
            st.warning(f"Could not create user-item matrix: {e}")
            user_item_matrix = pd.DataFrame()
    else:
        user_item_matrix = pd.DataFrame()
    
    # Compute cosine similarity between items (movies) if possible
    item_similarity_df = pd.DataFrame()
    if not user_item_matrix.empty and user_item_matrix.shape[1] > 0:
        try:
            item_similarity = cosine_similarity(user_item_matrix.T)
            item_similarity_df = pd.DataFrame(
                item_similarity, 
                index=user_item_matrix.columns, 
                columns=user_item_matrix.columns
            )
        except Exception as e:
            st.warning(f"Could not compute item similarity: {e}")
    
    return users, movies, ratings, links, item_similarity_df


# Load all data
users, movies, ratings, links, item_similarity_df = load_data()


def hash_password(password: str) -> str:
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def clear_cache() -> None:
    if hasattr(st, "cache_data") and hasattr(st.cache_data, "clear"):
        st.cache_data.clear()
    elif hasattr(st, "legacy_caching") and hasattr(st.legacy_caching, "clear_cache"):
        st.legacy_caching.clear_cache()
    elif hasattr(st, "cache") and hasattr(st.cache, "clear"):
        st.cache.clear()


def save_user(username: str, password: str) -> None:
    """Save new user to CSV file"""
    try:
        users_df = pd.read_csv(USERS_FILE)
        hashed = hash_password(password)
        new_user = pd.DataFrame([{"username": username, "password": hashed}])
        users_df = pd.concat([users_df, new_user], ignore_index=True)
        users_df.to_csv(USERS_FILE, index=False)
        clear_cache()
    except Exception as e:
        st.error(f"Error saving user: {e}")


def authenticate(username: str, password: str) -> bool:
    """Check if user credentials are valid"""
    try:
        users_df = pd.read_csv(USERS_FILE)
        hashed = hash_password(password)
        matched = users_df[(users_df["username"] == username) & (users_df["password"] == hashed)]
        return not matched.empty
    except Exception:
        return False


def user_exists(username: str) -> bool:
    """Check if username already exists"""
    try:
        users_df = pd.read_csv(USERS_FILE)
        return not users_df[users_df["username"] == username].empty
    except Exception:
        return False


def recommend(movie_name: str):
    """Get recommendations for a movie"""
    if not movie_name or not movie_name.strip():
        return [], None, "Please enter a movie name."
    
    try:
        search_text = movie_name.strip().lower()
        
        if movies.empty or "title" not in movies.columns:
            return [], None, "No movies available in database."
        
        movies_lower = movies.copy()
        movies_lower["lower_title"] = movies_lower["title"].astype(str).str.lower()
        
        matches = movies_lower[movies_lower["lower_title"].str.contains(search_text, case=False, na=False, regex=False)]
        
        if matches.empty:
            return [], None, "Movie not found."
        
        exact_title = matches["title"].iloc[0]
        
        if item_similarity_df.empty or exact_title not in item_similarity_df.index:
            return [], None, "No ratings available for this movie."
        
        similar_scores = item_similarity_df[exact_title]
        sorted_similar = similar_scores.sort_values(ascending=False)
        
        recommended_titles = [title for title in sorted_similar.index if title != exact_title][:5]
        return recommended_titles, exact_title, None
    except Exception as e:
        return [], None, f"Error getting recommendations: {str(e)}"


def format_movie_links(row):
    links_text = []
    if row.get("imdbId"):
        imdb_id = str(row["imdbId"]).strip()
        if imdb_id:
            links_text.append(f"[IMDb](https://www.imdb.com/title/tt{imdb_id})")
    if row.get("tmdbId"):
        tmdb_id = str(row["tmdbId"]).strip()
        if tmdb_id:
            links_text.append(f"[TMDb](https://www.themoviedb.org/movie/{tmdb_id})")
    return " | ".join(links_text)


def format_rating(row):
    """Format rating display"""
    try:
        rating_count = 0
        avg_rating = 0
        
        if "rating_count" in row and pd.notna(row["rating_count"]):
            rating_count = int(row["rating_count"])
        
        if "avg_rating" in row and pd.notna(row["avg_rating"]):
            avg_rating = float(row["avg_rating"])
        
        if rating_count > 0:
            return f"{avg_rating:.2f} ⭐ ({rating_count} ratings)"
        return "No ratings yet"
    except Exception:
        return "No ratings yet"


def homepage_page():
    st.markdown('<div class="main-header">🎬 Movie Recommendation System</div>', unsafe_allow_html=True)
    
    # Hero section
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Discover Your Next Favorite Movie
        Get personalized movie recommendations powered by collaborative filtering. 
        Explore thousands of movies across various genres and find what suits your taste.
        
        **Key Features:**
        - 🎯 Personalized recommendations based on user ratings
        - 📊 Detailed movie information with ratings and links
        - 👥 Community-driven insights
        - 🔒 Secure user authentication
        """)
        st.success(f"Welcome back, {st.session_state.username}! Ready to explore movies?")
    
    with col2:
        pass
    
    # Stats section
    st.markdown("---")
    st.subheader("📈 Platform Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        with col1:
            st.metric("Movies", f"{len(movies):,}")
        with col2:
            st.metric("Ratings", f"{len(ratings):,}")
        with col3:
            if not movies.empty and "genres" in movies.columns:
                genres_count = len(movies["genres"].astype(str).str.split("|", expand=True).stack().unique())
            else:
                genres_count = 0
            st.metric("Genres", genres_count)
        with col4:
            st.metric("Users", len(users))
    except Exception as e:
        st.warning(f"Could not load statistics: {e}")
    
    # Testimonials or features
    st.markdown("---")
    st.subheader("💬 What Our Users Say")
    col1, col2 = st.columns(2)
    with col1:
        st.info('"This app helped me discover amazing movies I never knew existed!" - Movie Lover')
    with col2:
        st.info('"The recommendations are spot on. Highly recommend!" - Film Enthusiast')


def profile_page():
    st.markdown('<div class="sub-header">👤 My Profile</div>', unsafe_allow_html=True)
    if st.session_state.logged_in:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("https://via.placeholder.com/150", caption="Profile Picture", width=150)
        with col2:
            st.write(f"**Username:** {st.session_state.username}")
            st.write("**Member since:** Today")
            st.write("**Favorite genres:** Action, Comedy")
        st.write("---")
        st.subheader("Account Settings")
        if st.button("Change Password"):
            st.info("Password change feature coming soon!")
    else:
        st.error("Please login to view your profile.")


def subjects_page():
    st.markdown('<div class="sub-header">🎭 Movie Genres</div>', unsafe_allow_html=True)
    st.write("Explore different movie genres available in our database.")
    
    try:
        if movies.empty or "genres" not in movies.columns:
            st.warning("No genres data available.")
            return
        
        genres = movies["genres"].astype(str).str.split("|", expand=True).stack().unique()
        genres = [g.strip() for g in genres if g and g.strip()]
        
        if not genres:
            st.warning("No genres available.")
            return
        
        cols = st.columns(3)
        for i, genre in enumerate(sorted(genres)):
            with cols[i % 3]:
                st.markdown(f"**{genre}**")
                genre_movies = movies[movies["genres"].astype(str).str.contains(genre, case=False, na=False)]
                st.write(f"{len(genre_movies)} movies")
    except Exception as e:
        st.error(f"Error loading genres: {e}")


def get_movie_description(row):
    """Generate movie description"""
    try:
        genres_str = str(row.get("genres", "Unknown")).strip()
        genres = [g.strip() for g in genres_str.split("|") if g.strip()] if genres_str else ["Unknown"]
        primary_genre = genres[0] if genres else "Unknown"
        rating_text = format_rating(row)
        return f"This is an engaging {primary_genre.lower()} movie that has captivated audiences with its compelling storyline and excellent performances. Rated {rating_text}, it's a must-watch for fans of {', '.join(genres).lower()}."
    except Exception:
        return "A great movie worth watching."


def main_app_page():
    st.markdown('<div class="sub-header">🔍 Movie Recommendations</div>', unsafe_allow_html=True)
    st.write(f"Hello, **{st.session_state.username}**! Enter a movie you like to get personalized recommendations.")

    movie_input = st.text_input("Enter your favorite movie:", key="movie_input")

    if st.button("Get Recommendations"):
        recommendations, selected_title, error = recommend(movie_input)

        if error:
            st.error(error)
            return
        
        try:
            selected_movie_list = movies[movies["title"] == selected_title]
            if selected_movie_list.empty:
                st.error("Selected movie not found.")
                return
            
            selected_movie = selected_movie_list.iloc[0]
            st.subheader("🎥 Selected Movie")
            with st.expander(f"**{selected_movie['title']}**"):
                st.write(f"**Genres:** {selected_movie.get('genres', 'Unknown')}")
                st.write(f"**Rating:** {format_rating(selected_movie)}")
                link_text = format_movie_links(selected_movie)
                if link_text:
                    st.markdown(f"**Links:** {link_text}")
                st.write(f"**Description:** {get_movie_description(selected_movie)}")

            if recommendations:
                st.subheader("🎬 Recommended Movies")
                for rec in recommendations:
                    rec_movie_list = movies[movies["title"] == rec]
                    if not rec_movie_list.empty:
                        row = rec_movie_list.iloc[0]
                        with st.expander(f"**{row['title']}**"):
                            st.write(f"**Genres:** {row.get('genres', 'Unknown')}")
                            st.write(f"**Rating:** {format_rating(row)}")
                            rec_links = format_movie_links(row)
                            if rec_links:
                                st.markdown(f"**Links:** {rec_links}")
                            st.write(f"**Description:** {get_movie_description(row)}")
            else:
                st.write("No recommendations available.")
        except Exception as e:
            st.error(f"Error displaying recommendations: {str(e)}")


def login_page():
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="main-header">🎬 Welcome</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; margin-bottom: 30px; color: #a0a0a0;">
            <p>Sign in to your account to start discovering amazing movies</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            st.markdown("### Login")
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", placeholder="Enter your password", type="password")
            
            st.markdown("")
            submitted = st.form_submit_button("🔓 Login", use_container_width=True)
            
            if submitted:
                if not username or not password:
                    st.error("❌ Please enter both a username and password.")
                    return

                if authenticate(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.page = "Homepage"
                    st.success("✅ Logged in successfully. Redirecting...")
                else:
                    st.error("❌ Invalid username or password. Please try again.")
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #a0a0a0;">
            <p>Don't have an account? <b>Go to Register</b> from the menu to create one.</p>
        </div>
        """, unsafe_allow_html=True)


def register_page():
    # Center the register form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="main-header">🎬 Join Us</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; margin-bottom: 30px; color: #a0a0a0;">
            <p>Create an account to start your movie journey</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("register_form"):
            st.markdown("### Create Your Account")
            new_username = st.text_input("Username", placeholder="Choose a unique username")
            new_password = st.text_input("Password", placeholder="Create a strong password", type="password")
            confirm_password = st.text_input("Confirm Password", placeholder="Re-enter your password", type="password")
            
            st.markdown("")
            submitted = st.form_submit_button("✨ Register", use_container_width=True)
            
            if submitted:
                if not new_username or not new_password or not confirm_password:
                    st.error("❌ Please fill in all fields.")
                    return

                if new_password != confirm_password:
                    st.error("❌ Passwords do not match.")
                    return

                if user_exists(new_username):
                    st.error("❌ Username already exists. Please choose another.")
                    return

                save_user(new_username, new_password)
                st.session_state.logged_in = True
                st.session_state.username = new_username
                st.session_state.page = "Homepage"
                st.success("✅ Registration successful. Redirecting...")
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #a0a0a0;">
            <p>Already have an account? <b>Go to Login</b> from the menu to sign in.</p>
        </div>
        """, unsafe_allow_html=True)


if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["username"] = ""

if "page" not in st.session_state:
    st.session_state["page"] = "Homepage"

if not st.session_state.logged_in:
    st.session_state["page"] = "Login"
    menu_options = ["Login", "Register"]
else:
    menu_options = ["Homepage", "My Profile", "Subjects", "Recommendations", "Logout"]

page = st.sidebar.selectbox(
    "Menu",
    menu_options,
    index=menu_options.index(st.session_state.page) if st.session_state.page in menu_options else 0,
)

st.session_state.page = page

if page == "Homepage":
    homepage_page()
elif page == "Login":
    login_page()
elif page == "Register":
    register_page()
elif page == "My Profile":
    profile_page()
elif page == "Subjects":
    subjects_page()
elif page == "Recommendations":
    main_app_page()
elif page == "Logout":
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.page = "Homepage"
    st.success("You have been logged out.")
    homepage_page()