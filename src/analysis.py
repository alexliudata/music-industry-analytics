"""
Music Industry Analytics - Analysis Module

This module handles:
- Genre trend analysis over time
- Artist performance analysis
- Audio features correlation
- Business insights generation
- Visualization creation
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from typing import List, Dict


class GenreTrendAnalyzer:
    """Analyze genre trends and performance over time."""
    
    def __init__(self, billboard_data: pd.DataFrame, spotify_data: pd.DataFrame):
        self.billboard_data = billboard_data
        self.spotify_data = spotify_data
        self.genre_trends = None
        self.genre_insights = {}
    
    def analyze_genre_trends(self) -> pd.DataFrame:
        """
        Analyze genre performance trends over time.
        
        Returns:
            DataFrame with genre trend analysis
        """
        if self.billboard_data.empty:
            return pd.DataFrame()
        
        # Ensure date is datetime
        self.billboard_data['date'] = pd.to_datetime(self.billboard_data['date'])
        
        # Group by genre and date to get weekly performance
        genre_weekly = self.billboard_data.groupby(['genre', 'date']).agg({
            'rank': ['count', 'mean', 'min'],
            'weeks_on_chart': 'mean'
        }).reset_index()
        
        # Flatten column names
        genre_weekly.columns = ['genre', 'date', 'song_count', 'avg_rank', 'best_rank', 'avg_weeks']
        
        # Calculate genre momentum (change in performance over time)
        genre_weekly['date'] = pd.to_datetime(genre_weekly['date'])
        genre_weekly = genre_weekly.sort_values(['genre', 'date'])
        
        # Calculate rolling averages and momentum
        genre_weekly['rolling_avg_rank'] = genre_weekly.groupby('genre')['avg_rank'].rolling(4).mean().reset_index(0, drop=True)
        genre_weekly['momentum'] = genre_weekly.groupby('genre')['rolling_avg_rank'].diff()
        
        # Calculate genre market share
        total_songs = genre_weekly.groupby('date')['song_count'].sum().reset_index()
        genre_weekly = genre_weekly.merge(total_songs, on='date', suffixes=('', '_total'))
        genre_weekly['market_share'] = genre_weekly['song_count'] / genre_weekly['song_count_total']
        
        self.genre_trends = genre_weekly
        return genre_weekly
    
    def identify_emerging_genres(self, threshold: float = 0.1) -> List[str]:
        """
        Identify genres with positive momentum.
        
        Args:
            threshold: Minimum momentum threshold
            
        Returns:
            List of emerging genres
        """
        if self.genre_trends is None:
            self.analyze_genre_trends()
        
        # Get recent momentum for each genre
        recent_data = self.genre_trends.groupby('genre').tail(4)
        genre_momentum = recent_data.groupby('genre')['momentum'].mean()
        
        # Identify emerging genres (negative momentum means improving rank)
        emerging = genre_momentum[genre_momentum < -threshold].index.tolist()
        
        return emerging
    
    def identify_declining_genres(self, threshold: float = 0.1) -> List[str]:
        """
        Identify genres with negative momentum.
        
        Args:
            threshold: Minimum momentum threshold
            
        Returns:
            List of declining genres
        """
        if self.genre_trends is None:
            self.analyze_genre_trends()
        
        # Get recent momentum for each genre
        recent_data = self.genre_trends.groupby('genre').tail(4)
        genre_momentum = recent_data.groupby('genre')['momentum'].mean()
        
        # Identify declining genres (positive momentum means worsening rank)
        declining = genre_momentum[genre_momentum > threshold].index.tolist()
        
        return declining
    
    def get_genre_insights(self) -> Dict:
        """
        Generate comprehensive genre insights.
        
        Returns:
            Dictionary with genre insights
        """
        if self.genre_trends is None:
            self.analyze_genre_trends()
        
        insights = {}
        
        # Top performing genres
        recent_data = self.genre_trends.groupby('genre').tail(4)
        top_genres = recent_data.groupby('genre')['avg_rank'].mean().nsmallest(5)
        insights['top_genres'] = top_genres.to_dict()
        
        # Emerging genres
        insights['emerging_genres'] = self.identify_emerging_genres()
        
        # Declining genres
        insights['declining_genres'] = self.identify_declining_genres()
        
        # Market share leaders
        market_leaders = recent_data.groupby('genre')['market_share'].mean().nlargest(5)
        insights['market_leaders'] = market_leaders.to_dict()
        
        # Genre stability (lowest variance in rank)
        genre_stability = recent_data.groupby('genre')['avg_rank'].std().nsmallest(5)
        insights['stable_genres'] = genre_stability.to_dict()
        
        return insights


class ArtistAnalyzer:
    """Analyze artist performance and trends."""
    
    def __init__(self, billboard_data: pd.DataFrame):
        self.billboard_data = billboard_data
        self.artist_performance = None
    
    def analyze_artist_performance(self) -> pd.DataFrame:
        """
        Analyze individual artist performance.
        
        Returns:
            DataFrame with artist performance metrics
        """
        if self.billboard_data.empty:
            return pd.DataFrame()
        
        # Calculate artist performance metrics
        artist_metrics = self.billboard_data.groupby('artist').agg({
            'rank': ['count', 'mean', 'min'],
            'weeks_on_chart': 'mean',
            'genre': 'nunique'
        }).reset_index()
        
        # Flatten column names
        artist_metrics.columns = ['artist', 'total_songs', 'avg_rank', 'best_rank', 'avg_weeks', 'genre_diversity']
        
        # Calculate artist consistency (lower std = more consistent)
        artist_consistency = self.billboard_data.groupby('artist')['rank'].std().reset_index()
        artist_consistency.columns = ['artist', 'rank_consistency']
        
        artist_metrics = artist_metrics.merge(artist_consistency, on='artist')
        
        # Calculate artist momentum (recent performance vs overall)
        # Ensure date is datetime
        self.billboard_data['date'] = pd.to_datetime(self.billboard_data['date'])
        recent_data = self.billboard_data[self.billboard_data['date'] >= 
                                        (self.billboard_data['date'].max() - timedelta(weeks=8))]
        
        recent_performance = recent_data.groupby('artist')['rank'].mean().reset_index()
        recent_performance.columns = ['artist', 'recent_avg_rank']
        
        artist_metrics = artist_metrics.merge(recent_performance, on='artist', how='left')
        artist_metrics['momentum'] = artist_metrics['avg_rank'] - artist_metrics['recent_avg_rank']
        
        self.artist_performance = artist_metrics
        return artist_metrics
    
    def get_top_artists(self, metric: str = 'total_songs', top_n: int = 10) -> pd.DataFrame:
        """
        Get top performing artists by metric.
        
        Args:
            metric: Performance metric to rank by
            top_n: Number of top artists to return
            
        Returns:
            DataFrame with top artists
        """
        if self.artist_performance is None:
            self.analyze_artist_performance()
        
        if metric == 'best_rank':
            return self.artist_performance.nsmallest(top_n, metric)
        else:
            return self.artist_performance.nlargest(top_n, metric)
    
    def get_rising_artists(self, threshold: float = 5.0) -> pd.DataFrame:
        """
        Get artists with positive momentum.
        
        Args:
            threshold: Minimum momentum threshold
            
        Returns:
            DataFrame with rising artists
        """
        if self.artist_performance is None:
            self.analyze_artist_performance()
        
        rising = self.artist_performance[self.artist_performance['momentum'] > threshold]
        return rising.sort_values('momentum', ascending=False)


class AudioFeaturesAnalyzer:
    """Analyze Spotify audio features and correlations."""
    
    def __init__(self, spotify_data: pd.DataFrame):
        self.spotify_data = spotify_data
        self.feature_correlations = None
    
    def analyze_audio_features(self) -> pd.DataFrame:
        """
        Analyze audio features by genre.
        
        Returns:
            DataFrame with genre audio feature profiles
        """
        if self.spotify_data.empty:
            return pd.DataFrame()
        
        # Calculate average audio features by genre
        audio_features = ['danceability', 'energy', 'valence', 'tempo', 'acousticness', 
                         'instrumentalness', 'liveness', 'speechiness']
        
        genre_features = self.spotify_data.groupby('genre')[audio_features].mean()
        
        # Add popularity by genre
        genre_popularity = self.spotify_data.groupby('genre')['popularity'].mean()
        genre_features['popularity'] = genre_popularity
        
        return genre_features
    
    def calculate_feature_correlations(self) -> pd.DataFrame:
        """
        Calculate correlations between audio features and popularity.
        
        Returns:
            DataFrame with feature correlations
        """
        if self.spotify_data.empty:
            return pd.DataFrame()
        
        # Select numeric columns
        numeric_cols = ['danceability', 'energy', 'valence', 'tempo', 'acousticness', 
                       'instrumentalness', 'liveness', 'speechiness', 'popularity']
        
        available_cols = [col for col in numeric_cols if col in self.spotify_data.columns]
        
        if len(available_cols) < 2:
            return pd.DataFrame()
        
        # Calculate correlation matrix
        correlation_matrix = self.spotify_data[available_cols].corr()
        
        self.feature_correlations = correlation_matrix
        return correlation_matrix
    
    def identify_optimal_features(self) -> Dict:
        """
        Identify optimal audio features for popularity.
        
        Returns:
            Dictionary with optimal feature ranges
        """
        if self.spotify_data.empty:
            return {}
        
        # Analyze features for high-popularity songs
        high_pop = self.spotify_data[self.spotify_data['popularity'] >= 80]
        
        if high_pop.empty:
            return {}
        
        optimal_features = {}
        audio_features = ['danceability', 'energy', 'valence', 'tempo', 'acousticness', 
                         'instrumentalness', 'liveness', 'speechiness']
        
        for feature in audio_features:
            if feature in high_pop.columns:
                optimal_features[feature] = {
                    'mean': high_pop[feature].mean(),
                    'std': high_pop[feature].std(),
                    'min': high_pop[feature].min(),
                    'max': high_pop[feature].max()
                }
        
        return optimal_features
    
    def cluster_genres_by_features(self, n_clusters: int = 5) -> Dict:
        """
        Cluster genres based on audio features.
        
        Args:
            n_clusters: Number of clusters
            
        Returns:
            Dictionary with genre clusters
        """
        if self.spotify_data.empty:
            return {}
        
        # Prepare data for clustering
        genre_features = self.analyze_audio_features()
        
        if genre_features.empty:
            return {}
        
        # Select features for clustering
        feature_cols = ['danceability', 'energy', 'valence', 'tempo', 'acousticness', 
                       'instrumentalness', 'liveness', 'speechiness']
        available_features = [col for col in feature_cols if col in genre_features.columns]
        
        if len(available_features) < 2:
            return {}
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(genre_features[available_features])
        
        # Perform clustering
        kmeans = KMeans(n_clusters=min(n_clusters, len(genre_features)), random_state=42)
        clusters = kmeans.fit_predict(scaled_features)
        
        # Create cluster assignments
        cluster_assignments = pd.DataFrame({
            'genre': genre_features.index,
            'cluster': clusters
        })
        
        return {
            'cluster_assignments': cluster_assignments.to_dict('records'),
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'feature_names': available_features
        }


class BusinessInsightsGenerator:
    """Generate business insights and recommendations."""
    
    def __init__(self, genre_analyzer: GenreTrendAnalyzer, 
                 artist_analyzer: ArtistAnalyzer, 
                 audio_analyzer: AudioFeaturesAnalyzer):
        self.genre_analyzer = genre_analyzer
        self.artist_analyzer = artist_analyzer
        self.audio_analyzer = audio_analyzer
    
    def generate_market_insights(self) -> Dict:
        """
        Generate comprehensive market insights.
        
        Returns:
            Dictionary with market insights
        """
        insights = {}
        
        # Genre insights
        genre_insights = self.genre_analyzer.get_genre_insights()
        insights['genre_insights'] = genre_insights
        
        # Artist insights
        top_artists = self.artist_analyzer.get_top_artists('total_songs', 10)
        rising_artists = self.artist_analyzer.get_rising_artists()
        
        insights['artist_insights'] = {
            'top_artists': top_artists.to_dict('records'),
            'rising_artists': rising_artists.to_dict('records')
        }
        
        # Audio feature insights
        optimal_features = self.audio_analyzer.identify_optimal_features()
        insights['audio_insights'] = optimal_features
        
        return insights
    
    def generate_strategic_recommendations(self) -> List[Dict]:
        """
        Generate strategic recommendations for artists and labels.
        
        Returns:
            List of strategic recommendations
        """
        recommendations = []
        
        # Genre-based recommendations
        emerging_genres = self.genre_analyzer.identify_emerging_genres()
        declining_genres = self.genre_analyzer.identify_declining_genres()
        
        if emerging_genres:
            recommendations.append({
                'type': 'genre_opportunity',
                'title': 'Emerging Genre Opportunities',
                'description': f'Genres showing positive momentum: {", ".join(emerging_genres)}',
                'action': 'Consider developing artists in these genres or collaborating with existing artists',
                'priority': 'high'
            })
        
        if declining_genres:
            recommendations.append({
                'type': 'genre_warning',
                'title': 'Declining Genre Trends',
                'description': f'Genres showing negative momentum: {", ".join(declining_genres)}',
                'action': 'Reconsider heavy investment in these genres or pivot to emerging alternatives',
                'priority': 'medium'
            })
        
        # Artist-based recommendations
        rising_artists = self.artist_analyzer.get_rising_artists()
        if not rising_artists.empty:
            top_rising = rising_artists.head(5)['artist'].tolist()
            recommendations.append({
                'type': 'artist_opportunity',
                'title': 'Rising Artist Opportunities',
                'description': f'Artists with strong momentum: {", ".join(top_rising)}',
                'action': 'Consider collaboration opportunities or signing these artists',
                'priority': 'high'
            })
        
        # Audio feature recommendations
        optimal_features = self.audio_analyzer.identify_optimal_features()
        if optimal_features:
            recommendations.append({
                'type': 'audio_optimization',
                'title': 'Audio Feature Optimization',
                'description': 'Optimal audio features for high popularity identified',
                'action': 'Use these feature ranges as guidelines for production decisions',
                'priority': 'medium'
            })
        
        return recommendations
    
    def generate_risk_assessment(self) -> Dict:
        """
        Generate risk assessment for the music industry.
        
        Returns:
            Dictionary with risk assessment
        """
        risks = {}
        
        # Market concentration risk
        genre_insights = self.genre_analyzer.get_genre_insights()
        market_leaders = genre_insights.get('market_leaders', {})
        
        if market_leaders:
            top_genre_share = max(market_leaders.values())
            if top_genre_share > 0.4:  # More than 40% market share
                risks['market_concentration'] = {
                    'level': 'high',
                    'description': f'Top genre has {top_genre_share:.1%} market share',
                    'mitigation': 'Diversify portfolio across multiple genres'
                }
        
        # Genre decline risk
        declining_genres = self.genre_analyzer.identify_declining_genres()
        if len(declining_genres) > 3:
            risks['genre_decline'] = {
                'level': 'medium',
                'description': f'{len(declining_genres)} genres showing decline',
                'mitigation': 'Monitor trends and consider genre diversification'
            }
        
        return risks


class VisualizationGenerator:
    """Generate interactive visualizations for the dashboard."""
    
    def __init__(self, genre_analyzer: GenreTrendAnalyzer, 
                 artist_analyzer: ArtistAnalyzer, 
                 audio_analyzer: AudioFeaturesAnalyzer):
        self.genre_analyzer = genre_analyzer
        self.artist_analyzer = artist_analyzer
        self.audio_analyzer = audio_analyzer
    
    def create_genre_trend_chart(self) -> go.Figure:
        """
        Create interactive genre trend chart.
        
        Returns:
            Plotly figure with genre trends
        """
        if self.genre_analyzer.genre_trends is None:
            self.genre_analyzer.analyze_genre_trends()
        
        fig = go.Figure()
        
        for genre in self.genre_analyzer.genre_trends['genre'].unique():
            genre_data = self.genre_analyzer.genre_trends[
                self.genre_analyzer.genre_trends['genre'] == genre
            ]
            
            fig.add_trace(go.Scatter(
                x=genre_data['date'],
                y=genre_data['rolling_avg_rank'],
                mode='lines+markers',
                name=genre,
                hovertemplate=f'<b>{genre}</b><br>' +
                             'Date: %{x}<br>' +
                             'Avg Rank: %{y:.1f}<br>' +
                             '<extra></extra>'
            ))
        
        fig.update_layout(
            title='Genre Performance Trends Over Time',
            xaxis_title='Date',
            yaxis_title='Average Rank (Lower is Better)',
            yaxis_autorange='reversed',
            hovermode='x unified',
            height=500
            autosize=True,  # Allow auto-sizing for full width
            margin=dict(l=50, r=50, t=80, b=50),  # Better margins
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_market_share_chart(self) -> go.Figure:
        """
        Create market share pie chart.
        
        Returns:
            Plotly figure with market share
        """
        if self.genre_analyzer.genre_trends is None:
            self.genre_analyzer.analyze_genre_trends()
        
        # Get recent market share data
        recent_data = self.genre_analyzer.genre_trends.groupby('genre').tail(4)
        market_share = recent_data.groupby('genre')['market_share'].mean().sort_values(ascending=False)
        
        fig = go.Figure(data=[go.Pie(
            labels=market_share.index,
            values=market_share.values,
            hole=0.3,
            hovertemplate='<b>%{label}</b><br>' +
                         'Market Share: %{percent}<br>' +
                         '<extra></extra>'
        )])
        
        fig.update_layout(
            title='Current Market Share by Genre',
            height=400
        )
        
        return fig
    
    def create_artist_performance_chart(self) -> go.Figure:
        """
        Create artist performance scatter plot.
        
        Returns:
            Plotly figure with artist performance
        """
        if self.artist_analyzer.artist_performance is None:
            self.artist_analyzer.analyze_artist_performance()
        
        fig = go.Figure()
        
        # Enhanced hover template with more details
        hover_template = (
            '<b>%{text}</b><br>' +
            'Total Billboard Chart Appearances: %{x}<br>' +
            'Average Billboard Chart Rank: %{y:.1f}<br>' +
            'Momentum Score: %{marker.color:.1f}<br>' +
            'Average Weeks on Chart: %{marker.size:.1f}<br>' +
            '<extra></extra>'
        )
        
        fig.add_trace(go.Scatter(
            x=self.artist_analyzer.artist_performance['total_songs'],
            y=self.artist_analyzer.artist_performance['avg_rank'],
            mode='markers',
            text=self.artist_analyzer.artist_performance['artist'],
            hovertemplate=hover_template,
            marker=dict(
                size=self.artist_analyzer.artist_performance['avg_weeks'],
                color=self.artist_analyzer.artist_performance['momentum'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title='Momentum Score')
            )
        ))
        
        fig.update_layout(
            title='Artist Performance Analysis',
            xaxis_title='Total Billboard Chart Appearances',
            yaxis_title='Average Billboard Chart Rank (Lower = Better)',
            yaxis_autorange='reversed',
            height=500
        )
        
        return fig
    
    def create_audio_features_heatmap(self) -> go.Figure:
        """
        Create audio features correlation heatmap.
        
        Returns:
            Plotly figure with audio features heatmap
        """
        correlation_matrix = self.audio_analyzer.calculate_feature_correlations()
        
        if correlation_matrix.empty:
            # Create empty figure if no data
            fig = go.Figure()
            fig.add_annotation(
                text="No audio features data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            hovertemplate='<b>%{y} vs %{x}</b><br>' +
                         'Correlation: %{z:.3f}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title='Audio Features Correlation Matrix',
            height=500
        )
        
        return fig


def main():
    """Main function to demonstrate analysis capabilities."""
    print("Music Industry Analytics - Analysis Module")
    print("=" * 50)
    
    # Load sample data
    from data_collection import DataProcessor
    
    processor = DataProcessor()
    billboard_data = processor.load_data('sample_billboard_data.csv')
    spotify_data = processor.load_data('sample_spotify_features.csv')
    
    if billboard_data.empty or spotify_data.empty:
        print("No data found. Please run data collection first.")
        return
    
    # Initialize analyzers
    genre_analyzer = GenreTrendAnalyzer(billboard_data, spotify_data)
    artist_analyzer = ArtistAnalyzer(billboard_data)
    audio_analyzer = AudioFeaturesAnalyzer(spotify_data)
    
    # Perform analysis
    print("\n1. Analyzing genre trends...")
    genre_trends = genre_analyzer.analyze_genre_trends()
    print(f"Analyzed {len(genre_trends)} genre trend data points")
    
    print("\n2. Analyzing artist performance...")
    artist_performance = artist_analyzer.analyze_artist_performance()
    print(f"Analyzed {len(artist_performance)} artists")
    
    print("\n3. Analyzing audio features...")
    audio_features = audio_analyzer.analyze_audio_features()
    print(f"Analyzed {len(audio_features)} genre audio profiles")
    
    # Generate insights
    print("\n4. Generating business insights...")
    insights_generator = BusinessInsightsGenerator(genre_analyzer, artist_analyzer, audio_analyzer)
    market_insights = insights_generator.generate_market_insights()
    recommendations = insights_generator.generate_strategic_recommendations()
    
    # Display key insights
    print("\n=== KEY INSIGHTS ===")
    print(f"Emerging genres: {market_insights['genre_insights'].get('emerging_genres', [])}")
    print(f"Declining genres: {market_insights['genre_insights'].get('declining_genres', [])}")
    print(f"Number of recommendations: {len(recommendations)}")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main() 