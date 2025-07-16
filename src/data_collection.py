"""
Music Industry Analytics - Data Collection Module

This module handles:
- Billboard Hot 100 web scraping
- Sample data generation for testing
- Data cleaning and preprocessing
- CSV export functionality
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime, timedelta
import json
import os
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class BillboardScraper:
    """Billboard Hot 100 web scraper with rate limiting and error handling."""
    
    def __init__(self, base_url: str = "https://www.billboard.com/charts/hot-100"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def get_chart_data(self, date: Optional[str] = None) -> pd.DataFrame:
        """
        Scrape Billboard Hot 100 data for a specific date.
        
        Args:
            date: Date string in YYYY-MM-DD format. If None, gets current chart.
            
        Returns:
            DataFrame with chart data
        """
        try:
            if date:
                url = f"{self.base_url}/{date}"
            else:
                url = self.base_url
            
            print(f"Scraping Billboard Hot 100 from: {url}")
            
            # Rate limiting
            time.sleep(random.uniform(1, 3))
            
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract chart data
            chart_data = []
            chart_items = soup.find_all('div', class_='o-chart-results-list-row-container')
            
            for item in chart_items:
                try:
                    # Extract rank
                    rank_elem = item.find('span', class_='c-label')
                    rank = int(rank_elem.text.strip()) if rank_elem else None
                    
                    # Extract song title
                    title_elem = item.find('h3', class_='c-title')
                    title = title_elem.text.strip() if title_elem else None
                    
                    # Extract artist
                    artist_elem = item.find('span', class_='c-label')
                    artist = artist_elem.text.strip() if artist_elem else None
                    
                    # Extract weeks on chart
                    weeks_elem = item.find('span', class_='c-label', string=lambda text: 'week' in text.lower() if text else False)
                    weeks = int(weeks_elem.text.split()[0]) if weeks_elem else 1
                    
                    if rank and title and artist:
                        chart_data.append({
                            'rank': rank,
                            'title': title,
                            'artist': artist,
                            'weeks_on_chart': weeks,
                            'date': date or datetime.now().strftime('%Y-%m-%d')
                        })
                        
                except Exception as e:
                    print(f"Error parsing chart item: {e}")
                    continue
            
            return pd.DataFrame(chart_data)
            
        except Exception as e:
            print(f"Error scraping Billboard data: {e}")
            return pd.DataFrame()
    
    def get_historical_data(self, start_date: str, end_date: str, interval_days: int = 7) -> pd.DataFrame:
        """
        Get historical chart data for a date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval_days: Days between chart snapshots
            
        Returns:
            DataFrame with historical chart data
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        all_data = []
        current = start
        
        while current <= end:
            date_str = current.strftime('%Y-%m-%d')
            print(f"Scraping data for {date_str}")
            
            chart_data = self.get_chart_data(date_str)
            if not chart_data.empty:
                all_data.append(chart_data)
            
            current += timedelta(days=interval_days)
            time.sleep(random.uniform(2, 5))  # Be respectful to the server
        
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


class SampleDataGenerator:
    """Generate sample data for testing and development."""
    
    def __init__(self):
        self.genres = [
            'Pop', 'Hip-Hop/Rap', 'Rock', 'R&B/Soul', 'Country', 
            'Electronic/Dance', 'Latin', 'Alternative', 'Indie', 'K-Pop'
        ]
        
        # Real song-artist combinations with correct genres
        self.song_artist_genre_data = [
            # Pop hits
            ('Anti-Hero', 'Taylor Swift', 'Pop'),
            ('Flowers', 'Miley Cyrus', 'Pop'),
            ('Vampire', 'Olivia Rodrigo', 'Pop'),
            ('As It Was', 'Harry Styles', 'Pop'),
            ('Hold Me Closer', 'Elton John & Britney Spears', 'Pop'),
            ('Late Night Talking', 'Harry Styles', 'Pop'),
            ('About Damn Time', 'Lizzo', 'Pop'),
            ('Break My Soul', 'Beyoncé', 'Pop'),
            ('Unholy', 'Sam Smith & Kim Petras', 'Pop'),
            ('Hold On', 'Justin Bieber', 'Pop'),
            
            # Hip-Hop/Rap hits
            ('HUMBLE.', 'Kendrick Lamar', 'Hip-Hop/Rap'),
            ('God\'s Plan', 'Drake', 'Hip-Hop/Rap'),
            ('Bad Guy', 'Billie Eilish', 'Hip-Hop/Rap'),
            ('The Box', 'Roddy Ricch', 'Hip-Hop/Rap'),
            ('Rockstar', 'DaBaby ft. Roddy Ricch', 'Hip-Hop/Rap'),
            ('Mood', '24kGoldn ft. Iann Dior', 'Hip-Hop/Rap'),
            ('Savage', 'Megan Thee Stallion', 'Hip-Hop/Rap'),
            ('WAP', 'Cardi B ft. Megan Thee Stallion', 'Hip-Hop/Rap'),
            ('Industry Baby', 'Lil Nas X ft. Jack Harlow', 'Hip-Hop/Rap'),
            ('In Da Club', '50 Cent', 'Hip-Hop/Rap'),
            
            # R&B/Soul hits
            ('Kill Bill', 'SZA', 'R&B/Soul'),
            ('Die For You', 'The Weeknd', 'R&B/Soul'),
            ('Shirt', 'SZA', 'R&B/Soul'),
            ('CUFF IT', 'Beyoncé', 'R&B/Soul'),
            ('Lift Me Up', 'Rihanna', 'R&B/Soul'),
            ('Calm Down', 'Rema & Selena Gomez', 'R&B/Soul'),
            ('Creepin\'', 'Metro Boomin, The Weeknd, 21 Savage', 'R&B/Soul'),
            ('Rich Flex', 'Drake & 21 Savage', 'R&B/Soul'),
            ('Good Days', 'SZA', 'R&B/Soul'),
            ('Essence', 'WizKid ft. Tems', 'R&B/Soul'),
            
            # Country hits
            ('Last Night', 'Morgan Wallen', 'Country'),
            ('You Proof', 'Morgan Wallen', 'Country'),
            ('Thought You Should Know', 'Morgan Wallen', 'Country'),
            ('The Kind of Love We Make', 'Luke Combs', 'Country'),
            ('Going, Going, Gone', 'Luke Combs', 'Country'),
            ('She Had Me At Heads Carolina', 'Cole Swindell', 'Country'),
            ('Buy Dirt', 'Jordan Davis ft. Luke Bryan', 'Country'),
            ('Til You Can\'t', 'Cody Johnson', 'Country'),
            ('If I Was a Cowboy', 'Miranda Lambert', 'Country'),
            ('Wild as Her', 'Corey Kent', 'Country'),
            
            # K-Pop hits
            ('Dynamite', 'BTS', 'K-Pop'),
            ('Butter', 'BTS', 'K-Pop'),
            ('Pink Venom', 'BLACKPINK', 'K-Pop'),
            ('Shut Down', 'BLACKPINK', 'K-Pop'),
            ('New Jeans', 'NewJeans', 'K-Pop'),
            ('Hype Boy', 'NewJeans', 'K-Pop'),
            ('Ditto', 'NewJeans', 'K-Pop'),
            ('OMG', 'NewJeans', 'K-Pop'),
            ('Super Shy', 'NewJeans', 'K-Pop'),
            ('ETA', 'NewJeans', 'K-Pop'),
            
            # Latin hits
            ('Tití Me Preguntó', 'Bad Bunny', 'Latin'),
            ('Me Porto Bonito', 'Bad Bunny & Chencho Corleone', 'Latin'),
            ('Ojitos Lindos', 'Bad Bunny & Bomba Estéreo', 'Latin'),
            ('Efecto', 'Bad Bunny', 'Latin'),
            ('Party', 'Bad Bunny & Rauw Alejandro', 'Latin'),
            ('Después de la Playa', 'Bad Bunny', 'Latin'),
            ('Moscow Mule', 'Bad Bunny', 'Latin'),
            ('La Corriente', 'Bad Bunny & Tony Dize', 'Latin'),
            ('El Apagón', 'Bad Bunny', 'Latin'),
            ('Andrea', 'Bad Bunny & Buscabulla', 'Latin'),
            
            # Rock hits (more recent/current)
            ('Running Up That Hill', 'Kate Bush', 'Rock'),
            ('Master of Puppets', 'Metallica', 'Rock'),
            ('Sweet Child O\' Mine', 'Guns N\' Roses', 'Rock'),
            ('Bohemian Rhapsody', 'Queen', 'Rock'),
            ('Hotel California', 'Eagles', 'Rock'),
            ('Smells Like Teen Spirit', 'Nirvana', 'Rock'),
            ('Wonderwall', 'Oasis', 'Rock'),
            ('Creep', 'Radiohead', 'Rock'),
            ('Zombie', 'The Cranberries', 'Rock'),
            ('Seven Nation Army', 'The White Stripes', 'Rock'),
            
            # Alternative hits
            ('Blinding Lights', 'The Weeknd', 'Alternative'),
            ('Circles', 'Post Malone', 'Alternative'),
            ('Sunflower', 'Post Malone & Swae Lee', 'Alternative'),
            ('Better Now', 'Post Malone', 'Alternative'),
            ('Psycho', 'Post Malone ft. Ty Dolla $ign', 'Alternative'),
            ('Congratulations', 'Post Malone ft. Quavo', 'Alternative'),
            ('I Fall Apart', 'Post Malone', 'Alternative'),
            ('White Iverson', 'Post Malone', 'Alternative'),
            ('Go Flex', 'Post Malone', 'Alternative'),
            ('Deja Vu', 'Post Malone', 'Alternative'),
            
            # Electronic/Dance hits
            ('Stay', 'The Kid LAROI & Justin Bieber', 'Electronic/Dance'),
            ('Cold Heart', 'Elton John & Dua Lipa', 'Electronic/Dance'),
            ('Don\'t Start Now', 'Dua Lipa', 'Electronic/Dance'),
            ('Levitating', 'Dua Lipa', 'Electronic/Dance'),
            ('Physical', 'Dua Lipa', 'Electronic/Dance'),
            ('Break My Heart', 'Dua Lipa', 'Electronic/Dance'),
            ('New Rules', 'Dua Lipa', 'Electronic/Dance'),
            ('IDGAF', 'Dua Lipa', 'Electronic/Dance'),
            ('One Kiss', 'Calvin Harris & Dua Lipa', 'Electronic/Dance'),
            ('We Found Love', 'Rihanna ft. Calvin Harris', 'Electronic/Dance')
        ]
    
    def generate_billboard_data(self, num_weeks: int = 52, num_songs: int = 100) -> pd.DataFrame:
        """
        Generate realistic Billboard Hot 100 data using actual song-artist combinations.
        
        Args:
            num_weeks: Number of weeks to generate data for
            num_songs: Number of songs per week
            
        Returns:
            DataFrame with realistic chart data
        """
        data = []
        start_date = datetime.now() - timedelta(weeks=num_weeks)
        
        # Create a pool of songs that will appear in charts
        chart_songs = []
        
        # Add realistic chart patterns - some songs are more popular than others
        for song, artist, genre in self.song_artist_genre_data:
            # Determine how often this song appears based on its "hit potential"
            if genre in ['Pop', 'Hip-Hop/Rap']:
                frequency = random.randint(15, 25)  # More frequent for popular genres
            elif genre in ['R&B/Soul', 'Country']:
                frequency = random.randint(10, 20)  # Medium frequency
            else:
                frequency = random.randint(5, 15)   # Less frequent for niche genres
            
            for _ in range(frequency):
                chart_songs.append((song, artist, genre))
        
        for week in range(num_weeks):
            current_date = start_date + timedelta(weeks=week)
            date_str = current_date.strftime('%Y-%m-%d')
            
            # Shuffle songs for this week to create realistic chart movement
            weekly_songs = random.sample(chart_songs, min(num_songs, len(chart_songs)))
            
            for rank in range(1, min(num_songs + 1, len(weekly_songs) + 1)):
                song, artist, genre = weekly_songs[rank - 1]
                
                # Realistic weeks on chart based on rank and genre
                if rank <= 10:
                    weeks_on_chart = random.randint(8, 25)  # Top hits stay longer
                elif rank <= 50:
                    weeks_on_chart = random.randint(3, 15)  # Mid-chart songs
                else:
                    weeks_on_chart = random.randint(1, 8)   # Lower chart songs
                
                data.append({
                    'rank': rank,
                    'title': song,
                    'artist': artist,
                    'genre': genre,
                    'weeks_on_chart': weeks_on_chart,
                    'date': date_str
                })
        
        return pd.DataFrame(data)
    
    def generate_spotify_features(self, num_songs: int = 1000) -> pd.DataFrame:
        """
        Generate realistic Spotify audio features data using actual songs.
        
        Args:
            num_songs: Number of songs to generate features for
            
        Returns:
            DataFrame with realistic audio features
        """
        data = []
        
        # Create multiple variations of each song to reach num_songs
        songs_needed = num_songs
        while len(data) < songs_needed:
            for song, artist, genre in self.song_artist_genre_data:
                if len(data) >= songs_needed:
                    break
                
                # Generate realistic audio features based on genre
                if genre == 'Pop':
                    danceability = random.uniform(0.6, 0.9)
                    energy = random.uniform(0.5, 0.8)
                    valence = random.uniform(0.4, 0.8)
                    tempo = random.uniform(100, 140)
                    acousticness = random.uniform(0.1, 0.4)
                    instrumentalness = random.uniform(0, 0.1)
                    liveness = random.uniform(0.05, 0.2)
                    speechiness = random.uniform(0.02, 0.08)
                elif genre == 'Hip-Hop/Rap':
                    danceability = random.uniform(0.7, 0.95)
                    energy = random.uniform(0.6, 0.9)
                    valence = random.uniform(0.3, 0.7)
                    tempo = random.uniform(80, 120)
                    acousticness = random.uniform(0, 0.2)
                    instrumentalness = random.uniform(0, 0.05)
                    liveness = random.uniform(0.05, 0.15)
                    speechiness = random.uniform(0.08, 0.25)
                elif genre == 'Rock':
                    danceability = random.uniform(0.3, 0.7)
                    energy = random.uniform(0.7, 0.95)
                    valence = random.uniform(0.3, 0.8)
                    tempo = random.uniform(120, 180)
                    acousticness = random.uniform(0.1, 0.6)
                    instrumentalness = random.uniform(0.1, 0.4)
                    liveness = random.uniform(0.1, 0.3)
                    speechiness = random.uniform(0.02, 0.06)
                elif genre == 'R&B/Soul':
                    danceability = random.uniform(0.4, 0.8)
                    energy = random.uniform(0.3, 0.7)
                    valence = random.uniform(0.2, 0.6)
                    tempo = random.uniform(60, 100)
                    acousticness = random.uniform(0.2, 0.5)
                    instrumentalness = random.uniform(0, 0.2)
                    liveness = random.uniform(0.05, 0.2)
                    speechiness = random.uniform(0.03, 0.1)
                elif genre == 'Country':
                    danceability = random.uniform(0.4, 0.7)
                    energy = random.uniform(0.4, 0.7)
                    valence = random.uniform(0.3, 0.7)
                    tempo = random.uniform(70, 110)
                    acousticness = random.uniform(0.3, 0.8)
                    instrumentalness = random.uniform(0.1, 0.3)
                    liveness = random.uniform(0.05, 0.2)
                    speechiness = random.uniform(0.02, 0.08)
                elif genre == 'K-Pop':
                    danceability = random.uniform(0.7, 0.9)
                    energy = random.uniform(0.6, 0.9)
                    valence = random.uniform(0.4, 0.8)
                    tempo = random.uniform(100, 140)
                    acousticness = random.uniform(0.1, 0.3)
                    instrumentalness = random.uniform(0, 0.1)
                    liveness = random.uniform(0.05, 0.2)
                    speechiness = random.uniform(0.02, 0.08)
                elif genre == 'Latin':
                    danceability = random.uniform(0.7, 0.95)
                    energy = random.uniform(0.6, 0.9)
                    valence = random.uniform(0.5, 0.8)
                    tempo = random.uniform(90, 130)
                    acousticness = random.uniform(0.1, 0.4)
                    instrumentalness = random.uniform(0, 0.2)
                    liveness = random.uniform(0.05, 0.2)
                    speechiness = random.uniform(0.03, 0.1)
                elif genre == 'Alternative':
                    danceability = random.uniform(0.4, 0.7)
                    energy = random.uniform(0.5, 0.8)
                    valence = random.uniform(0.3, 0.6)
                    tempo = random.uniform(80, 120)
                    acousticness = random.uniform(0.2, 0.6)
                    instrumentalness = random.uniform(0.1, 0.3)
                    liveness = random.uniform(0.05, 0.2)
                    speechiness = random.uniform(0.02, 0.08)
                elif genre == 'Electronic/Dance':
                    danceability = random.uniform(0.7, 0.95)
                    energy = random.uniform(0.7, 0.95)
                    valence = random.uniform(0.4, 0.8)
                    tempo = random.uniform(110, 150)
                    acousticness = random.uniform(0, 0.2)
                    instrumentalness = random.uniform(0.1, 0.4)
                    liveness = random.uniform(0.05, 0.2)
                    speechiness = random.uniform(0.02, 0.06)
                else:  # Default for other genres
                    danceability = random.uniform(0.4, 0.8)
                    energy = random.uniform(0.4, 0.8)
                    valence = random.uniform(0.3, 0.7)
                    tempo = random.uniform(80, 140)
                    acousticness = random.uniform(0.1, 0.5)
                    instrumentalness = random.uniform(0, 0.3)
                    liveness = random.uniform(0.05, 0.2)
                    speechiness = random.uniform(0.02, 0.1)
                
                # Generate popularity based on genre and features
                base_popularity = 50
                if genre in ['Pop', 'Hip-Hop/Rap']:
                    base_popularity = 70
                elif genre in ['R&B/Soul', 'Country']:
                    base_popularity = 60
                elif genre in ['K-Pop', 'Latin']:
                    base_popularity = 65
                else:
                    base_popularity = 45
                
                # Add variation based on audio features
                popularity = min(100, max(20, base_popularity + 
                                        (danceability - 0.5) * 40 +
                                        (energy - 0.5) * 30 +
                                        random.randint(-10, 10)))
                
                data.append({
                    'track_name': song,
                    'artist_name': artist,
                    'genre': genre,
                    'danceability': danceability,
                    'energy': energy,
                    'valence': valence,
                    'tempo': tempo,
                    'acousticness': acousticness,
                    'instrumentalness': instrumentalness,
                    'liveness': liveness,
                    'speechiness': speechiness,
                    'popularity': int(popularity)
                })
        
        return pd.DataFrame(data)


class DataProcessor:
    """Handle data cleaning, preprocessing, and export."""
    
    def __init__(self):
        self.output_dir = '../data'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def clean_billboard_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess Billboard data.
        
        Args:
            df: Raw Billboard data
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Clean text fields
        df['title'] = df['title'].str.strip()
        df['artist'] = df['artist'].str.strip()
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Add week number
        df['week_number'] = df['date'].dt.isocalendar().week
        df['year'] = df['date'].dt.year
        
        # Add genre if not present (for sample data)
        if 'genre' not in df.columns:
            df['genre'] = 'Unknown'
        
        return df
    
    def clean_spotify_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess Spotify data.
        
        Args:
            df: Raw Spotify data
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Clean text fields
        df['track_name'] = df['track_name'].str.strip()
        df['artist_name'] = df['artist_name'].str.strip()
        df['genre'] = df['genre'].str.strip()
        
        # Ensure numeric columns are numeric
        numeric_cols = ['danceability', 'energy', 'valence', 'tempo', 'acousticness', 
                       'instrumentalness', 'liveness', 'speechiness', 'popularity']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['track_name', 'artist_name', 'genre'])
        
        return df
    
    def export_data(self, df: pd.DataFrame, filename: str, format: str = 'csv'):
        """
        Export data to file.
        
        Args:
            df: DataFrame to export
            filename: Output filename
            format: Export format ('csv', 'json', 'excel')
        """
        filepath = os.path.join(self.output_dir, filename)
        
        if format == 'csv':
            df.to_csv(filepath, index=False)
        elif format == 'json':
            df.to_json(filepath, orient='records', indent=2)
        elif format == 'excel':
            df.to_excel(filepath, index=False)
        
        print(f"Data exported to: {filepath}")
    
    def load_data(self, filename: str, format: str = 'csv') -> pd.DataFrame:
        """
        Load data from file.
        
        Args:
            filename: Input filename
            format: File format ('csv', 'json', 'excel')
            
        Returns:
            Loaded DataFrame
        """
        filepath = os.path.join(self.output_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return pd.DataFrame()
        
        try:
            if format == 'csv':
                return pd.read_csv(filepath)
            elif format == 'json':
                return pd.read_json(filepath)
            elif format == 'excel':
                return pd.read_excel(filepath)
            else:
                print(f"Unsupported format: {format}")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()


def main():
    """Main function to demonstrate data collection."""
    print("Music Industry Analytics - Data Collection")
    print("=" * 50)
    
    # Initialize components
    scraper = BillboardScraper()
    generator = SampleDataGenerator()
    processor = DataProcessor()
    
    # Generate sample data for testing
    print("\n1. Generating sample Billboard data...")
    billboard_data = generator.generate_billboard_data(num_weeks=26, num_songs=100)
    billboard_data = processor.clean_billboard_data(billboard_data)
    processor.export_data(billboard_data, 'sample_billboard_data.csv')
    
    print("\n2. Generating sample Spotify features data...")
    spotify_data = generator.generate_spotify_features(num_songs=500)
    spotify_data = processor.clean_spotify_data(spotify_data)
    processor.export_data(spotify_data, 'sample_spotify_features.csv')
    
    print("\n3. Data collection complete!")
    print(f"Billboard data shape: {billboard_data.shape}")
    print(f"Spotify data shape: {spotify_data.shape}")
    
    # Show sample of the data
    print("\nSample Billboard data:")
    print(billboard_data.head())
    
    print("\nSample Spotify features:")
    print(spotify_data.head())


if __name__ == "__main__":
    main() 