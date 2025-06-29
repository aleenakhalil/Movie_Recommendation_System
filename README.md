# Hybrid_Movie_Recommendation_System

## Project Overview  
This project implements a **Hybrid Movie Recommendation System** using the **MovieLens 1M dataset**. The goal is to recommend movies to users by combining **Content-Based Filtering** and **Collaborative Filtering** techniques. It also integrates the **TMDb API** to fetch and display movie posters. The app is built using **Streamlit** and provides an interactive UI for exploring movie suggestions across three modes.

---

## Dataset  
The dataset used is the **MovieLens 1M Dataset**, which includes:

- `ratings.dat`: User ratings for movies (`UserID::MovieID::Rating::Timestamp`)
- `movies.dat`: Movie metadata including titles and genres
- `users.dat`: Demographic info for users (gender, age, occupation)

Download available from: https://drive.google.com/file/d/1EYcJ_YJouhEph1LurQwyQLh0DM5H0COc/view 

---

## Key Features

### Data Preprocessing & Caching
- Loaded large `.dat` files efficiently using `pandas`
- Handled encoding issues with `latin-1`
- Applied `@st.cache_data` decorators for fast and optimized reloading

### Recommendation Logic

**1. Content-Based Filtering**
- Uses TF-IDF vectorization on movie genres
- Computes cosine similarity between movies
- Recommends similar movies based on genre profile

**2. Collaborative Filtering**
- Constructs a user-item rating matrix
- Applies Truncated SVD to reduce dimensions
- Finds similar users and recommends their favorite movies

**3. Hybrid Filtering**
- Combines both methods by averaging scores (weighted)
- Provides smarter recommendations using both user behavior and content similarity

### Poster Integration
- Fetches movie posters from **TMDb API**
- Cleans title queries for better poster matches

---

## UI Highlights

- Built entirely with **Streamlit**
- Light-themed UI with lavender-pink background (`#f8e8f4`)
- Three recommendation modes selectable via horizontal radio buttons
- Movie results are displayed in rows of 3 posters
- Mobile-friendly and responsive layout

---

## Tools and Libraries Used

- `streamlit` – Web UI
- `pandas`, `numpy` – Data handling
- `scikit-learn` – TF-IDF, SVD, cosine similarity
- `requests` – API calls to TMDb
- `TMDb API` – Movie poster retrieval

---

##  How to Run

1. Clone the repo or download the files
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Launch the app:
```bash
streamlit run app.py

