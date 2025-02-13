import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import joblib

class NetflixRecommender:
    def __init__(self):
        self.movies_df = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None
        
    def load_and_preprocess_data(self, filepath):
        
        print("Loading and preprocessing data... This might take a few minutes...")
        self.movies_df = pd.read_csv(filepath)
        
        self.movies_df['combined_features'] = self.movies_df.apply(self._combine_features, axis=1)
        
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.movies_df['combined_features'].fillna(''))
        
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        self.indices = pd.Series(self.movies_df.index, index=self.movies_df['title']).drop_duplicates()
        print("Data preprocessing completed!")

    def search_movies(self, search_term):
       
        search_term = search_term.lower()
        matches = self.movies_df[self.movies_df['title'].str.lower().str.contains(search_term, na=False)]
        
        if len(matches) == 0:
            return []
        
        matches = matches.sort_values('release_year', ascending=False)
        return matches[['title', 'type', 'release_year']].to_dict('records')
    
    def _combine_features(self, row):
      
        important_features = [
            row['type'],
            row['listed_in'],
            row['description'],
            row['director'] if pd.notna(row['director']) else '',
            row['cast'] if pd.notna(row['cast']) else ''
        ]
        return ' '.join(important_features)
    
    def get_recommendations(self, title, n_recommendations=5):
       
        if title not in self.indices:
            return None
            
        idx = self.indices[title]
        
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        sim_scores = sim_scores[1:n_recommendations+1]
        movie_indices = [i[0] for i in sim_scores]
        
        recommendations = self.movies_df.iloc[movie_indices][['title', 'type', 'listed_in', 'description']]
        recommendations['similarity_score'] = [i[1] for i in sim_scores]
        return recommendations
    
    def save_model(self, filepath):
       
        model_data = {
            'movies_df': self.movies_df,
            'tfidf_matrix': self.tfidf_matrix,
            'cosine_sim': self.cosine_sim,
            'indices': self.indices
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        
        print(f"Loading model from {filepath}...")
        model_data = joblib.load(filepath)
        self.movies_df = model_data['movies_df']
        self.tfidf_matrix = model_data['tfidf_matrix']
        self.cosine_sim = model_data['cosine_sim']
        self.indices = model_data['indices']
        print("Model loaded successfully!")

def main():
    recommender = NetflixRecommender()
    
    try:
        recommender.load_model('netflix_recommender_model.joblib')
    except:
        recommender.load_and_preprocess_data('netflix_titles.csv')
        recommender.save_model('netflix_recommender_model.joblib')
    
    while True:
        search_term = input("\nEnter a movie or show name to search (or 'quit' to exit): ")
        
        if search_term.lower() == 'quit':
            break
            
        results = recommender.search_movies(search_term)
        
        if results:
            print(f"\nFound {len(results)} matches:")
            for i, movie in enumerate(results, 1):
                print(f"{i}. {movie['title']} ({movie['type']}, {movie['release_year']})")
            
            while True:
                try:
                    selection = int(input("\nEnter the number of the movie you want recommendations for: ")) - 1
                    if 0 <= selection < len(results):
                        break
                    print("Please enter a valid number from the list.")
                except ValueError:
                    print("Please enter a valid number.")
            
            selected_movie = results[selection]['title']
            print(f"\nGetting recommendations based on: {selected_movie}")
            recommendations = recommender.get_recommendations(selected_movie)
            
            print("\nRecommended Movies/Shows:")
            for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                print(f"\n{i}. {row['title']} ({row['type']})")
                print(f"   Genres: {row['listed_in']}")
                print(f"   Description: {row['description']}")
                print(f"   Similarity Score: {row['similarity_score']:.2f}")
        else:
            print(f"No movies found matching '{search_term}'")

if __name__ == "__main__":
    main()