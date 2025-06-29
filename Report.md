# Hybrid Movie Recommendation System

## Project Overview  
This project implements a **Hybrid Movie Recommendation System** using the **MovieLens 1M dataset**. The goal is to recommend movies to users by combining **Content-Based Filtering** and **Collaborative Filtering** techniques. It also integrates the **TMDb API** to fetch and display movie posters. The app is built using **Streamlit** and provides an interactive UI for exploring movie suggestions across three modes.

---

## Objective  
- Build a hybrid recommendation system combining content and collaborative filtering  
- Personalize movie recommendations for users based on ratings and genre similarity  
- Integrate **TMDb API** for fetching movie posters  
- Create a simple and visually appealing **Streamlit** interface  

---

## Dataset Description  
The dataset used is the **MovieLens 1M Dataset**, which includes:
- `ratings.dat` – 1,000,209 user ratings for 3,900 movies  
- `movies.dat` – Movie titles and genres  
- `users.dat` – User demographics (age, gender, occupation)

> Dataset source: https://drive.google.com/file/d/1EYcJ_YJouhEph1LurQwyQLh0DM5H0COc/view
---

## Methodology

### 1. **Data Preprocessing**
- Loaded `.dat` files using pandas with proper separators and encoding  
- Used `@st.cache_data` to cache large datasets for fast reloading  
- Extracted clean movie titles to improve API querying

### 2. **Content-Based Filtering**
- Applied **TF-IDF Vectorizer** to movie genres  
- Computed cosine similarity between genre vectors  
- Recommended movies based on genre similarity to a selected movie

### 3. **Collaborative Filtering**
- Created user-item rating matrix  
- Used **Truncated SVD** for dimensionality reduction  
- Calculated cosine similarity between users  
- Recommended movies based on ratings from similar users

### 4. **Hybrid Recommendation**
- Combined content and collaborative scores using a weighted average (`alpha = 0.6`)  
- Returned top 5 movies sorted by hybrid scores

### 5. **Poster Retrieval**
- Fetched movie posters via the **TMDb API** using the movie title (excluding release year)  
- Displayed posters alongside movie titles

### 6. **User Interface**
- Built using **Streamlit**  
- Three recommendation modes: Content-Based, Collaborative Filtering, Hybrid  
- Posters displayed in rows of 3 with titles underneath  
- Background styled with a soft lavender-pink (`#f8e8f4`)

---

##  Sample Output  
- Personalized recommendations displayed with posters  
- Responsive layout (first 3 posters in row 1, next 2 in row 2)  
- Quick loading thanks to Streamlit caching

---

## Results  
- Personalized recommendations are generated in real-time for all three modes  
- Posters are fetched and displayed accurately for most titles  
- The hybrid model significantly improves relevance by combining strengths of both CBF and CF

---


