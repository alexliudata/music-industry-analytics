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
    .tooltip {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.25rem;
        padding: 0.5rem;
        font-size: 0.875rem;
        color: #6c757d;
        margin-top: 0.25rem;
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

def filter_data_by_date_range(data, date_range):
    """Filter data by selected date range."""
    if len(date_range) == 2 and data is not None and not data.empty:
        start_date, end_date = date_range
        data['date'] = pd.to_datetime(data['date'])
        filtered_data = data[(data['date'].dt.date >= start_date) & 
                           (data['date'].dt.date <= end_date)]
        return filtered_data
    return data

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
    
    # Export functionality
    st.sidebar.subheader("📤 Export Data")
    if st.sidebar.button("Export Analysis Report", help="Download current analysis as CSV"):
        # Create a comprehensive report
        if not billboard_data.empty:
            # Summary metrics
            summary_data = {
                'Metric': ['Total Songs', 'Unique Artists', 'Genres Analyzed', 'Date Range'],
                'Value': [
                    len(billboard_data),
                    billboard_data['artist'].nunique(),
                    billboard_data['genre'].nunique(),
                    f"{billboard_data['date'].min().strftime('%Y-%m-%d')} to {billboard_data['date'].max().strftime('%Y-%m-%d')}"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            
            # Artist performance data
            artist_performance = artist_analyzer.analyze_artist_performance()
            if not artist_performance.empty:
                artist_df = artist_performance[['artist', 'total_songs', 'avg_rank', 'best_rank', 'momentum', 'genre_diversity']].copy()
                artist_df.columns = ['Artist', 'Total Chart Appearances', 'Average Rank', 'Best Rank', 'Momentum Score', 'Genre Diversity']
            else:
                artist_df = pd.DataFrame()
            
            # Combine data for export
            if not artist_df.empty:
                # Create a comprehensive report
                report_data = {
                    'Section': ['Summary'] * len(summary_df) + ['Artist Performance'] * len(artist_df),
                    'Metric': list(summary_df['Metric']) + list(artist_df.columns),
                    'Value': list(summary_df['Value']) + [artist_df.iloc[i].tolist() for i in range(len(artist_df))]
                }
                report_df = pd.DataFrame(report_data)
                
                st.sidebar.download_button(
                    label="Download Full Report",
                    data=report_df.to_csv(index=False),
                    file_name="music_analytics_report.csv",
                    mime="text/csv"
                )
            else:
                st.sidebar.download_button(
                    label="Download Summary",
                    data=summary_df.to_csv(index=False),
                    file_name="music_analytics_summary.csv",
                    mime="text/csv"
                )
    
    # Global date range selector (syncs across all tabs)
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
            max_value=max_date,
            help="This filter applies to all dashboard tabs"
        )
        
        # Filter data by date range
        filtered_billboard = filter_data_by_date_range(billboard_data, date_range)
        filtered_spotify = spotify_data  # Spotify data doesn't have dates, so keep as is
    
    # Genre filter
    st.sidebar.subheader("🎼 Genre Filter")
    if not billboard_data.empty:
        available_genres = sorted(billboard_data['genre'].unique())
        selected_genres = st.sidebar.multiselect(
            "Select genres to analyze",
            options=available_genres,
            default=available_genres[:5] if len(available_genres) >= 5 else available_genres,
            help="Filter analysis to specific genres"
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
        show_overview_tab(filtered_billboard, filtered_spotify, genre_analyzer, artist_analyzer, audio_analyzer)
    
    with tab2:
        show_genre_analysis_tab(filtered_billboard, genre_analyzer, audio_analyzer, artist_analyzer)
    
    with tab3:
        show_artist_analysis_tab(filtered_billboard, artist_analyzer, genre_analyzer, audio_analyzer)
    
    with tab4:
        show_audio_features_tab(filtered_spotify, audio_analyzer, genre_analyzer, artist_analyzer)
    
    with tab5:
        show_business_insights_tab(genre_analyzer, artist_analyzer, audio_analyzer)

def show_overview_tab(billboard_data, spotify_data, genre_analyzer, artist_analyzer, audio_analyzer):
    """Display overview dashboard."""
    
    st.header("📊 Market Overview")
    
    # Key metrics with tooltips
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Songs Analyzed",
            value=f"{len(billboard_data):,}",
            delta=f"+{len(billboard_data) - 1000:,}" if len(billboard_data) > 1000 else None
        )
        st.markdown('<div class="tooltip">📊 Total number of chart entries in the selected date range</div>', unsafe_allow_html=True)
    
    with col2:
        st.metric(
            label="Unique Artists",
            value=f"{billboard_data['artist'].nunique():,}",
            delta=None
        )
        st.markdown('<div class="tooltip">👤 Number of distinct artists with chart entries</div>', unsafe_allow_html=True)
    
    with col3:
        st.metric(
            label="Genres Tracked",
            value=f"{billboard_data['genre'].nunique():,}",
            delta=None
        )
        st.markdown('<div class="tooltip">🎼 Number of different music genres represented</div>', unsafe_allow_html=True)
    
    with col4:
        if not billboard_data.empty:
            time_period = (pd.to_datetime(billboard_data['date'].max()) - pd.to_datetime(billboard_data['date'].min())).days
            st.metric(
                label="Time Period",
                value=f"{time_period} days",
                delta=None
            )
            st.markdown('<div class="tooltip">📅 Duration of the selected analysis period</div>', unsafe_allow_html=True)
    
    # Genre performance overview
    st.subheader("🎼 Genre Performance Overview")
    
    # Analyze genre trends
    genre_trends = genre_analyzer.analyze_genre_trends()
    
    if not genre_trends.empty:
        # Create genre trend chart with clarification
        st.markdown('<div class="tooltip">📈 Average chart position across all songs in the genre, lower is better</div>', unsafe_allow_html=True)
        
        viz_generator = VisualizationGenerator(genre_analyzer, artist_analyzer, audio_analyzer)
        trend_fig = viz_generator.create_genre_trend_chart()
        st.plotly_chart(trend_fig, use_container_width=True)
        
        # Market share chart with tooltip
        st.markdown('<div class="tooltip">📊 Percentage of total chart entries by genre (hover for song counts)</div>', unsafe_allow_html=True)
        
        market_fig = viz_generator.create_market_share_chart()
        st.plotly_chart(market_fig, use_container_width=True)
    
    # Recent top performers with improved table
    st.subheader("🏆 Recent Top Performers")
    
    if not billboard_data.empty:
        # Get the latest date for context
        latest_date = pd.to_datetime(billboard_data['date'].max()).strftime('%B %d, %Y')
        st.markdown(f'<div class="tooltip">📅 As of latest chart update: {latest_date}</div>', unsafe_allow_html=True)
        
        # Get recent data (last 4 weeks)
        recent_data = billboard_data[pd.to_datetime(billboard_data['date']) >= 
                                   (pd.to_datetime(billboard_data['date'].max()) - timedelta(weeks=4))]
        
        # Get top 10 songs and sort by rank
        top_songs = recent_data[recent_data['rank'] <= 10][['rank', 'title', 'artist', 'genre']].head(10)
        top_songs = top_songs.sort_values('rank')  # Sort by rank (1-10)
        
        # Display as a clean table
        st.dataframe(
            top_songs,
            use_container_width=True,
            column_config={
                "rank": st.column_config.NumberColumn("Rank", help="Chart position (1-10)"),
                "title": st.column_config.TextColumn("Song Title"),
                "artist": st.column_config.TextColumn("Artist"),
                "genre": st.column_config.TextColumn("Genre")
            }
        )

def show_genre_analysis_tab(billboard_data, genre_analyzer, audio_analyzer, artist_analyzer):
    """Display genre analysis."""
    
    st.header("🎼 Genre Trend Analysis")
    
    # Emerging vs declining genres with explanation
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚀 Emerging Genres")
        st.markdown('<div class="tooltip">📈 Genres showing positive momentum (improving average rank over time)</div>', unsafe_allow_html=True)
        
        emerging_genres = genre_analyzer.identify_emerging_genres()
        
        if emerging_genres:
            for genre in emerging_genres:
                # Get momentum value for display
                genre_trends = genre_analyzer.analyze_genre_trends()
                if not genre_trends.empty:
                    recent_momentum = genre_trends[genre_trends['genre'] == genre]['momentum'].tail(4).mean()
                    momentum_text = f"↑{abs(recent_momentum):.1f} momentum" if recent_momentum < 0 else f"↓{recent_momentum:.1f} momentum"
                    st.markdown(f'<div class="success-box">✅ <strong>{genre}</strong> - {momentum_text}</div>', 
                               unsafe_allow_html=True)
        else:
            st.info("No emerging genres identified in the current data.")
    
    with col2:
        st.subheader("📉 Declining Genres")
        st.markdown('<div class="tooltip">📉 Genres showing negative momentum (worsening average rank over time)</div>', unsafe_allow_html=True)
        
        declining_genres = genre_analyzer.identify_declining_genres()
        
        if declining_genres:
            for genre in declining_genres:
                # Get momentum value for display
                genre_trends = genre_analyzer.analyze_genre_trends()
                if not genre_trends.empty:
                    recent_momentum = genre_trends[genre_trends['genre'] == genre]['momentum'].tail(4).mean()
                    momentum_text = f"↓{recent_momentum:.1f} momentum" if recent_momentum > 0 else f"↑{abs(recent_momentum):.1f} momentum"
                    st.markdown(f'<div class="warning-box">⚠️ <strong>{genre}</strong> - {momentum_text}</div>', 
                               unsafe_allow_html=True)
        else:
            st.info("No declining genres identified in the current data.")
    
    # Genre trend visualization
    st.subheader("📈 Genre Performance Trends")
    viz_generator = VisualizationGenerator(genre_analyzer, artist_analyzer, audio_analyzer)
    trend_fig = viz_generator.create_genre_trend_chart()
    st.plotly_chart(trend_fig, use_container_width=True)
    
    # Genre insights with clarified metrics
    st.subheader("📊 Genre Insights")
    genre_insights = genre_analyzer.get_genre_insights()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Top Performing Genres (by average rank):**")
        st.markdown('<div class="tooltip">🏆 Lower average rank = better performance (rank 1 is best)</div>', unsafe_allow_html=True)
        for genre, rank in list(genre_insights.get('top_genres', {}).items())[:5]:
            st.write(f"• {genre}: {rank:.1f} (avg rank)")
    
    with col2:
        st.write("**Market Share Leaders:**")
        st.markdown('<div class="tooltip">📊 Percentage of total chart entries by genre</div>', unsafe_allow_html=True)
        for genre, share in list(genre_insights.get('market_leaders', {}).items())[:5]:
            st.write(f"• {genre}: {share:.1%}")

def show_artist_analysis_tab(billboard_data, artist_analyzer, genre_analyzer, audio_analyzer):
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
        
        # Clarify genre diversity metric
        st.markdown('<div class="tooltip">🎭 Genre Diversity: Number of different genres an artist has charted in (1 = single genre, higher = more diverse)</div>', unsafe_allow_html=True)
        
        st.dataframe(top_artists[['artist', selected_metric, 'genre_diversity']], use_container_width=True)
        
        # Rising artists with context line
        st.subheader("📈 Rising Artists")
        st.markdown('<div class="tooltip">🚀 Artists demonstrating above-average performance growth in the past 6 months based on synthetic chart data</div>', unsafe_allow_html=True)
        
        rising_artists = artist_analyzer.get_rising_artists()
        
        if not rising_artists.empty:
            st.write("Artists with strong positive momentum:")
            for _, artist in rising_artists.head(5).iterrows():
                momentum_text = f"↑{artist['momentum']:.1f} rank improvement" if artist['momentum'] > 0 else f"↓{abs(artist['momentum']):.1f} rank decline"
                st.markdown(f'<div class="success-box">🚀 <strong>{artist["artist"]}</strong> - {momentum_text}</div>', 
                           unsafe_allow_html=True)
        else:
            st.info("No rising artists identified in the current data.")
        
        # Artist performance visualization with improved labels
        st.subheader("📊 Artist Performance Scatter Plot")
        st.markdown('<div class="tooltip">💡 Hover over points to see artist details, momentum score, and performance metrics</div>', unsafe_allow_html=True)
        
        viz_generator = VisualizationGenerator(genre_analyzer, artist_analyzer, audio_analyzer)
        artist_fig = viz_generator.create_artist_performance_chart()
        st.plotly_chart(artist_fig, use_container_width=True)

def show_audio_features_tab(spotify_data, audio_analyzer, genre_analyzer, artist_analyzer):
    """Display audio features analysis."""
    
    st.header("🎵 Audio Features Analysis")
    
    if not spotify_data.empty:
        # Audio features by genre
        st.subheader("🎼 Audio Features by Genre")
        genre_features = audio_analyzer.analyze_audio_features()
        
        if not genre_features.empty:
            st.dataframe(genre_features.round(3), use_container_width=True)
            
            # Feature correlation heatmap with legend
            st.subheader("🔥 Audio Features Correlation")
            st.markdown('<div class="tooltip">🔴 Red = negative correlation, 🔵 Blue = positive correlation, ⚪ White = no correlation</div>', unsafe_allow_html=True)
            
            viz_generator = VisualizationGenerator(genre_analyzer, artist_analyzer, audio_analyzer)
            heatmap_fig = viz_generator.create_audio_features_heatmap()
            st.plotly_chart(heatmap_fig, use_container_width=True)
        
        # Optimal features for popularity with explanation
        st.subheader("⭐ Optimal Features for High Popularity")
        st.markdown('<div class="tooltip">📊 Based on mean ± std of top 10% most popular songs</div>', unsafe_allow_html=True)
        
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
    
    # Market insights summary with momentum definition
    st.subheader("📊 Market Insights Summary")
    st.markdown('<div class="tooltip">📈 Based on Billboard chart performance analysis over time</div>', unsafe_allow_html=True)
    
    # Add momentum definition
    st.markdown('<div class="tooltip">📊 **Momentum** = change in average Billboard rank over time; positive = improving, negative = declining</div>', unsafe_allow_html=True)
    
    # Add realistic data quality note
    st.info("💡 **Data Note**: Analysis based on simulated data with realistic industry patterns. Real-world implementation would require Billboard API access and live data feeds.")
    
    # Compact badges for emerging/declining genres
    emerging_count = len(market_insights['genre_insights'].get('emerging_genres', []))
    declining_count = len(market_insights['genre_insights'].get('declining_genres', []))
    
    st.markdown(f'''
    <div style="display: flex; gap: 2rem; align-items: center; margin-bottom: 1.5rem;">
        <span style="background-color: #28a745; color: white; padding: 0.5rem 1.2rem; border-radius: 1rem; font-weight: bold; font-size: 1.1rem;">Emerging Genres: {emerging_count}</span>
        <span style="background-color: #ffc107; color: #333; padding: 0.5rem 1.2rem; border-radius: 1rem; font-weight: bold; font-size: 1.1rem;">Declining Genres: {declining_count}</span>
    </div>
    ''', unsafe_allow_html=True)
    
    # Strategic recommendations with backing metrics
    st.subheader("🎯 Strategic Recommendations")
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            priority_color = {
                'high': 'success-box',
                'medium': 'insight-box',
                'low': 'warning-box'
            }.get(rec['priority'], 'insight-box')
            
            # Add backing metrics for emerging genres recommendation
            if 'emerging' in rec['title'].lower() and 'genre' in rec['title'].lower():
                emerging_genres = market_insights['genre_insights'].get('emerging_genres', [])
                if emerging_genres:
                    genre_trends = genre_analyzer.analyze_genre_trends()
                    metrics_text = ""
                    # Only include genres that actually have data
                    valid_genres = []
                    for genre in emerging_genres:
                        if not genre_trends.empty and genre in genre_trends['genre'].values:
                            recent_momentum = genre_trends[genre_trends['genre'] == genre]['momentum'].tail(4).mean()
                            market_share = genre_trends[genre_trends['genre'] == genre]['market_share'].tail(4).mean()
                            metrics_text += f"<br>• <strong>{genre}</strong>: ↑{abs(recent_momentum):.1f} momentum, {market_share:.1%} market share"
                            valid_genres.append(genre)
                    
                    # Update description to only mention genres with supporting data
                    if valid_genres:
                        rec['description'] = rec['description'].replace(
                            f'Genres showing positive momentum: {", ".join(emerging_genres)}',
                            f'Genres showing positive momentum: {", ".join(valid_genres)}'
                        )
                        rec['description'] += f"<br><br><strong>Supporting Data:</strong>{metrics_text}"
            
            # Add backing metrics for rising artists recommendation with synthetic data note
            elif 'rising' in rec['title'].lower() and 'artist' in rec['title'].lower():
                rising_artists = artist_analyzer.get_rising_artists()
                if not rising_artists.empty:
                    metrics_text = "<br><br><strong>Top Rising Artists:</strong>"
                    metrics_text += "<br><em>Note: Based on simulated data for illustrative purposes</em>"
                    for _, artist in rising_artists.head(3).iterrows():
                        momentum_text = f"↑{artist['momentum']:.1f} rank improvement" if artist['momentum'] > 0 else f"↓{abs(artist['momentum']):.1f} rank decline"
                        metrics_text += f"<br>• <strong>{artist['artist']}</strong>: {momentum_text}"
                    
                    rec['description'] += metrics_text
            
            # Vary recommendation language to avoid repetition
            if 'developing artists' in rec['action']:
                rec['action'] = rec['action'].replace('Consider developing artists in these genres or collaborating with existing artists', 
                                                     'Explore signing new talent or building joint projects in these trending genres')
            elif 'reducing exposure' in rec['action']:
                rec['action'] = rec['action'].replace('Consider reducing exposure to declining genres', 
                                                     'Reduce overexposure in fading genres while reallocating marketing spend to growth areas')
            
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