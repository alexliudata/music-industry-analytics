"""
Music Industry Analytics - Streamlit Dashboard

Interactive dashboard for analyzing music industry trends,
artist performance, and generating business insights.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_collection import DataProcessor, SampleDataGenerator
from analysis import (
    GenreTrendAnalyzer, 
    ArtistAnalyzer, 
    AudioFeaturesAnalyzer,
    BusinessInsightsGenerator,
    VisualizationGenerator
)

# Page configuration
st.set_page_config(
    page_title="Music Industry Analytics",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffc107;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #28a745;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache data for the dashboard."""
    processor = DataProcessor()
    
    # Try to load the new realistic data first
    billboard_data = processor.load_data('billboard_hot_100.csv')
    spotify_data = processor.load_data('spotify_features.csv')
    
    # If new data doesn't exist, generate it
    if billboard_data.empty or spotify_data.empty:
        st.info("Generating realistic music industry data...")
        generator = SampleDataGenerator()
        
        billboard_data = generator.generate_billboard_data(num_weeks=52, num_songs=100)
        billboard_data = processor.clean_billboard_data(billboard_data)
        processor.export_data(billboard_data, 'billboard_hot_100.csv')
        
        spotify_data = generator.generate_spotify_features(num_songs=1000)
        spotify_data = processor.clean_spotify_data(spotify_data)
        processor.export_data(spotify_data, 'spotify_features.csv')
    
    return billboard_data, spotify_data

@st.cache_data
def initialize_analyzers(billboard_data, spotify_data):
    """Initialize and cache analyzers."""
    genre_analyzer = GenreTrendAnalyzer(billboard_data, spotify_data)
    artist_analyzer = ArtistAnalyzer(billboard_data)
    audio_analyzer = AudioFeaturesAnalyzer(spotify_data)
    
    return genre_analyzer, artist_analyzer, audio_analyzer

def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<h1 class="main-header">🎵 Music Industry Analytics</h1>', unsafe_allow_html=True)
    st.markdown("### Strategic Insights for Artists and Labels")
    
    # Load data
    with st.spinner("Loading data..."):
        billboard_data, spotify_data = load_data()
    
    # Initialize analyzers
    genre_analyzer, artist_analyzer, audio_analyzer = initialize_analyzers(billboard_data, spotify_data)
    
    # Sidebar
    st.sidebar.title("📊 Dashboard Controls")
    
    # Date range selector
    st.sidebar.subheader("📅 Date Range")
    if not billboard_data.empty:
        # Convert string dates to datetime objects
        billboard_data['date'] = pd.to_datetime(billboard_data['date'])
        min_date = billboard_data['date'].min().date()
        max_date = billboard_data['date'].max().date()
        
        date_range = st.sidebar.date_input(
            "Select date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    
    # Genre filter
    st.sidebar.subheader("🎼 Genre Filter")
    if not billboard_data.empty:
        available_genres = sorted(billboard_data['genre'].unique())
        selected_genres = st.sidebar.multiselect(
            "Select genres to analyze",
            options=available_genres,
            default=available_genres[:5] if len(available_genres) >= 5 else available_genres
        )
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", 
        "🎼 Genre Analysis", 
        "👤 Artist Analysis", 
        "🎵 Audio Features", 
        "💡 Business Insights"
    ])
    
    with tab1:
        show_overview_tab(billboard_data, spotify_data, genre_analyzer, artist_analyzer)
    
    with tab2:
        show_genre_analysis_tab(billboard_data, genre_analyzer)
    
    with tab3:
        show_artist_analysis_tab(billboard_data, artist_analyzer)
    
    with tab4:
        show_audio_features_tab(spotify_data, audio_analyzer)
    
    with tab5:
        show_business_insights_tab(genre_analyzer, artist_analyzer, audio_analyzer)

def show_overview_tab(billboard_data, spotify_data, genre_analyzer, artist_analyzer):
    """Display overview dashboard."""
    
    st.header("📊 Market Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Songs Analyzed",
            value=f"{len(billboard_data):,}",
            delta=f"+{len(billboard_data) - 1000:,}" if len(billboard_data) > 1000 else None
        )
    
    with col2:
        st.metric(
            label="Unique Artists",
            value=f"{billboard_data['artist'].nunique():,}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Genres Tracked",
            value=f"{billboard_data['genre'].nunique():,}",
            delta=None
        )
    
    with col4:
        st.metric(
            label="Time Period",
            value=f"{(pd.to_datetime(billboard_data['date'].max()) - pd.to_datetime(billboard_data['date'].min())).days} days",
            delta=None
        )
    
    # Genre performance overview
    st.subheader("🎼 Genre Performance Overview")
    
    # Analyze genre trends
    genre_trends = genre_analyzer.analyze_genre_trends()
    
    if not genre_trends.empty:
        # Create genre trend chart
        viz_generator = VisualizationGenerator(genre_analyzer, artist_analyzer, None)
        trend_fig = viz_generator.create_genre_trend_chart()
        st.plotly_chart(trend_fig, use_container_width=True)
        
        # Market share chart
        market_fig = viz_generator.create_market_share_chart()
        st.plotly_chart(market_fig, use_container_width=True)
    
    # Recent top performers
    st.subheader("🏆 Recent Top Performers")
    
    if not billboard_data.empty:
        recent_data = billboard_data[pd.to_datetime(billboard_data['date']) >= 
                                   (pd.to_datetime(billboard_data['date'].max()) - timedelta(weeks=4))]
        
        top_songs = recent_data[recent_data['rank'] <= 10][['rank', 'title', 'artist', 'genre']].head(10)
        st.dataframe(top_songs, use_container_width=True)

def show_genre_analysis_tab(billboard_data, genre_analyzer):
    """Display genre analysis."""
    
    st.header("🎼 Genre Trend Analysis")
    
    # Analyze genre trends
    genre_trends = genre_analyzer.analyze_genre_trends()
    
    if not genre_trends.empty:
        # Emerging vs declining genres
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚀 Emerging Genres")
            emerging_genres = genre_analyzer.identify_emerging_genres()
            
            if emerging_genres:
                for genre in emerging_genres:
                    st.markdown(f'<div class="success-box">✅ <strong>{genre}</strong> - Showing positive momentum</div>', 
                               unsafe_allow_html=True)
            else:
                st.info("No emerging genres identified in the current data.")
        
        with col2:
            st.subheader("📉 Declining Genres")
            declining_genres = genre_analyzer.identify_declining_genres()
            
            if declining_genres:
                for genre in declining_genres:
                    st.markdown(f'<div class="warning-box">⚠️ <strong>{genre}</strong> - Showing negative momentum</div>', 
                               unsafe_allow_html=True)
            else:
                st.info("No declining genres identified in the current data.")
        
        # Genre insights
        st.subheader("📊 Genre Insights")
        genre_insights = genre_analyzer.get_genre_insights()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Performing Genres (by average rank):**")
            for genre, rank in list(genre_insights.get('top_genres', {}).items())[:5]:
                st.write(f"• {genre}: {rank:.1f}")
        
        with col2:
            st.write("**Market Share Leaders:**")
            for genre, share in list(genre_insights.get('market_leaders', {}).items())[:5]:
                st.write(f"• {genre}: {share:.1%}")
        
        # Genre trend visualization
        st.subheader("📈 Genre Performance Trends")
        viz_generator = VisualizationGenerator(genre_analyzer, None, None)
        trend_fig = viz_generator.create_genre_trend_chart()
        st.plotly_chart(trend_fig, use_container_width=True)

def show_artist_analysis_tab(billboard_data, artist_analyzer):
    """Display artist analysis."""
    
    st.header("👤 Artist Performance Analysis")
    
    # Analyze artist performance
    artist_performance = artist_analyzer.analyze_artist_performance()
    
    if not artist_performance.empty:
        # Top artists by different metrics
        st.subheader("🏆 Top Artists")
        
        metric_options = ['total_songs', 'avg_rank', 'best_rank', 'avg_weeks', 'momentum']
        metric_labels = ['Total Songs', 'Average Rank', 'Best Rank', 'Avg Weeks', 'Momentum']
        
        selected_metric = st.selectbox(
            "Rank artists by:",
            options=metric_options,
            format_func=lambda x: metric_labels[metric_options.index(x)]
        )
        
        top_artists = artist_analyzer.get_top_artists(selected_metric, 10)
        st.dataframe(top_artists[['artist', selected_metric, 'genre_diversity']], use_container_width=True)
        
        # Rising artists
        st.subheader("📈 Rising Artists")
        rising_artists = artist_analyzer.get_rising_artists()
        
        if not rising_artists.empty:
            st.write("Artists with strong positive momentum:")
            for _, artist in rising_artists.head(5).iterrows():
                st.markdown(f'<div class="success-box">🚀 <strong>{artist["artist"]}</strong> - Momentum: {artist["momentum"]:.1f}</div>', 
                           unsafe_allow_html=True)
        else:
            st.info("No rising artists identified in the current data.")
        
        # Artist performance visualization
        st.subheader("📊 Artist Performance Scatter Plot")
        viz_generator = VisualizationGenerator(None, artist_analyzer, None)
        artist_fig = viz_generator.create_artist_performance_chart()
        st.plotly_chart(artist_fig, use_container_width=True)

def show_audio_features_tab(spotify_data, audio_analyzer):
    """Display audio features analysis."""
    
    st.header("🎵 Audio Features Analysis")
    
    if not spotify_data.empty:
        # Audio features by genre
        st.subheader("🎼 Audio Features by Genre")
        genre_features = audio_analyzer.analyze_audio_features()
        
        if not genre_features.empty:
            st.dataframe(genre_features.round(3), use_container_width=True)
            
            # Feature correlation heatmap
            st.subheader("🔥 Audio Features Correlation")
            viz_generator = VisualizationGenerator(None, None, audio_analyzer)
            heatmap_fig = viz_generator.create_audio_features_heatmap()
            st.plotly_chart(heatmap_fig, use_container_width=True)
        
        # Optimal features for popularity
        st.subheader("⭐ Optimal Features for High Popularity")
        optimal_features = audio_analyzer.identify_optimal_features()
        
        if optimal_features:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Feature Ranges for High Popularity Songs:**")
                for feature, stats in optimal_features.items():
                    st.write(f"• **{feature}**: {stats['mean']:.3f} ± {stats['std']:.3f}")
            
            with col2:
                st.write("**Feature Analysis:**")
                st.write("• **Danceability**: Higher values indicate more danceable tracks")
                st.write("• **Energy**: Higher values indicate more energetic tracks")
                st.write("• **Valence**: Higher values indicate more positive/happy tracks")
                st.write("• **Tempo**: Speed of the track in BPM")
        
        # Genre clustering
        st.subheader("🎯 Genre Clustering by Audio Features")
        clusters = audio_analyzer.cluster_genres_by_features()
        
        if clusters:
            st.write("**Genre Clusters:**")
            for assignment in clusters['cluster_assignments']:
                st.write(f"• **{assignment['genre']}**: Cluster {assignment['cluster']}")
        else:
            st.info("Insufficient data for genre clustering.")

def show_business_insights_tab(genre_analyzer, artist_analyzer, audio_analyzer):
    """Display business insights and recommendations."""
    
    st.header("💡 Business Insights & Recommendations")
    
    # Initialize insights generator
    insights_generator = BusinessInsightsGenerator(genre_analyzer, artist_analyzer, audio_analyzer)
    
    # Generate insights
    market_insights = insights_generator.generate_market_insights()
    recommendations = insights_generator.generate_strategic_recommendations()
    risks = insights_generator.generate_risk_assessment()
    
    # Market insights summary
    st.subheader("📊 Market Insights Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Emerging Genres",
            value=len(market_insights['genre_insights'].get('emerging_genres', [])),
            delta=None
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Declining Genres",
            value=len(market_insights['genre_insights'].get('declining_genres', [])),
            delta=None
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Strategic recommendations
    st.subheader("🎯 Strategic Recommendations")
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            priority_color = {
                'high': 'success-box',
                'medium': 'insight-box',
                'low': 'warning-box'
            }.get(rec['priority'], 'insight-box')
            
            st.markdown(f'''
            <div class="{priority_color}">
                <h4>{i}. {rec['title']}</h4>
                <p><strong>Description:</strong> {rec['description']}</p>
                <p><strong>Recommended Action:</strong> {rec['action']}</p>
                <p><strong>Priority:</strong> {rec['priority'].title()}</p>
            </div>
            ''', unsafe_allow_html=True)
    else:
        st.info("No specific recommendations available with current data.")
    
    # Risk assessment
    st.subheader("⚠️ Risk Assessment")
    
    if risks:
        for risk_type, risk_info in risks.items():
            risk_color = {
                'high': 'warning-box',
                'medium': 'insight-box',
                'low': 'success-box'
            }.get(risk_info['level'], 'insight-box')
            
            st.markdown(f'''
            <div class="{risk_color}">
                <h4>{risk_type.replace('_', ' ').title()}</h4>
                <p><strong>Level:</strong> {risk_info['level'].title()}</p>
                <p><strong>Description:</strong> {risk_info['description']}</p>
                <p><strong>Mitigation:</strong> {risk_info['mitigation']}</p>
            </div>
            ''', unsafe_allow_html=True)
    else:
        st.success("No significant risks identified in the current market analysis.")
    
    # Action items
    st.subheader("📋 Action Items")
    
    st.markdown("""
    **For Artists:**
    - Monitor emerging genre trends and consider genre experimentation
    - Analyze audio features of successful tracks in your target genre
    - Collaborate with rising artists in complementary genres
    
    **For Labels:**
    - Diversify artist roster across emerging genres
    - Invest in artists showing strong momentum
    - Monitor market concentration risks
    - Use audio feature analysis for A&R decisions
    
    **For Industry Professionals:**
    - Track genre momentum indicators regularly
    - Analyze cross-genre collaboration opportunities
    - Monitor market share distribution for concentration risks
    """)

if __name__ == "__main__":
    main() 