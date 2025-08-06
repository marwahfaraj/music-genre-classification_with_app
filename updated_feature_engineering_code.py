# Updated Feature Engineering Code with Golden Slide Aesthetic
# Copy this code into your 02_Feature_Engineering.ipynb notebook

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import mode
import warnings
import os
from matplotlib.colors import LinearSegmentedColormap

# Install kaleido for plotly static image export if needed
try:
    import kaleido
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kaleido"])
    import kaleido

warnings.filterwarnings("ignore", category=FutureWarning)

# ═══════════════════════════════════════════════════════════════
# 🎨 GOLDEN THEME SETUP - Matching the slide aesthetic
# ═══════════════════════════════════════════════════════════════

# Set up elegant golden color scheme matching the slide aesthetic
plt.style.use('dark_background')
golden_palette = ['#D4AF37', '#DAA520', '#B8860B', '#CD853F', '#DEB887', '#F4A460']

print("📚 Libraries imported successfully!")
print("🎨 Golden color scheme activated!")

# ═══════════════════════════════════════════════════════════════
# 📊 UPDATED VISUALIZATION CODE
# ═══════════════════════════════════════════════════════════════

# 1. GENRE DISTRIBUTION VISUALIZATION
def create_genre_distribution_plot(data):
    """Create genre distribution bar chart with golden aesthetic"""
    plt.figure(figsize=(12, 6), facecolor='#2C1810')
    genre_counts = data['playlist_genre'].value_counts()
    ax = sns.barplot(x=genre_counts.index, y=genre_counts.values, palette=golden_palette)
    ax.set_facecolor('#2C1810')
    plt.title('Song Distribution by Genre', fontsize=18, color='#D4AF37', fontweight='bold', pad=20)
    plt.xlabel('Genre', fontsize=14, color='#DEB887', fontweight='semibold')
    plt.ylabel('Count', fontsize=14, color='#DEB887', fontweight='semibold')

    # Style the plot to match slide aesthetic
    ax.tick_params(colors='#DEB887', labelsize=11)
    ax.spines['bottom'].set_color('#D4AF37')
    ax.spines['left'].set_color('#D4AF37')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, color='#D4AF37')

    # Add percentage labels
    total = len(data)
    for p in ax.patches:
        percentage = f'{100 * p.get_height()/total:.1f}%'
        ax.annotate(percentage, (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10),
                    textcoords='offset points', fontsize=11, color='#F5F5DC', fontweight='bold')

    # Save the plot
    os.makedirs('/workspace/images/eda', exist_ok=True)
    plt.savefig('/workspace/images/eda/genre_distribution.png', 
                facecolor='#2C1810', edgecolor='none', dpi=300, bbox_inches='tight')
    plt.show()
    print("💾 Plot saved to: /workspace/images/eda/genre_distribution.png")

# 2. CORRELATION MATRIX VISUALIZATION
def create_correlation_matrix(data):
    """Create correlation matrix heatmap with golden theme"""
    plt.figure(figsize=(14, 10), facecolor='#2C1810')
    audio_features = ['danceability', 'energy', 'loudness',
                     'acousticness', 'valence', 'tempo',
                     'speechiness', 'instrumentalness']
    corr_matrix = data[audio_features].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Create custom colormap with golden tones
    colors = ['#2C1810', '#8B4513', '#CD853F', '#D4AF37', '#F5DEB3']
    golden_cmap = LinearSegmentedColormap.from_list('golden', colors, N=256)

    ax = plt.gca()
    ax.set_facecolor('#2C1810')
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
                cmap=golden_cmap, linewidths=0.5, linecolor='#D4AF37',
                annot_kws={'color': '#2C1810', 'fontweight': 'bold'},
                cbar_kws={'shrink': 0.8})
    plt.title("Audio Feature Correlation Matrix", fontsize=18, color='#D4AF37', fontweight='bold', pad=20)

    # Style axes
    ax.tick_params(colors='#DEB887', labelsize=11)
    plt.xticks(rotation=45, ha='right', color='#DEB887')
    plt.yticks(rotation=0, color='#DEB887')

    # Save the plot
    plt.savefig('/workspace/images/eda/correlation_matrix.png', 
                facecolor='#2C1810', edgecolor='none', dpi=300, bbox_inches='tight')
    plt.show()
    print("💾 Plot saved to: /workspace/images/eda/correlation_matrix.png")

# 3. FEATURE DISTRIBUTION BOXPLOTS
def create_feature_boxplots(data):
    """Create feature distribution boxplots with golden theme"""
    audio_features = ['danceability', 'energy', 'loudness',
                     'acousticness', 'valence', 'tempo',
                     'speechiness', 'instrumentalness']
    
    plt.figure(figsize=(15, 10), facecolor='#2C1810')
    for i, feature in enumerate(audio_features[:6], 1):
        plt.subplot(2, 3, i)
        ax = plt.gca()
        ax.set_facecolor('#2C1810')
        
        sns.boxplot(x='playlist_genre', y=feature, data=data, palette=golden_palette)
        plt.title(f'{feature.capitalize()} by Genre', fontsize=14, color='#D4AF37', fontweight='bold', pad=10)
        plt.xticks(rotation=45, color='#DEB887', fontsize=10)
        plt.yticks(color='#DEB887', fontsize=10)
        plt.xlabel('Genre', color='#DEB887', fontsize=11, fontweight='semibold')
        plt.ylabel(feature.capitalize(), color='#DEB887', fontsize=11, fontweight='semibold')
        
        # Style the subplot
        ax.tick_params(colors='#DEB887')
        for spine in ax.spines.values():
            spine.set_color('#D4AF37')
            spine.set_linewidth(0.8)
        ax.grid(True, alpha=0.2, color='#D4AF37')

    plt.tight_layout()
    # Save the plot
    plt.savefig('/workspace/images/eda/feature_distributions.png', 
                facecolor='#2C1810', edgecolor='none', dpi=300, bbox_inches='tight')
    plt.show()
    print("💾 Plot saved to: /workspace/images/eda/feature_distributions.png")

# 4. INTERACTIVE RADAR CHART
def create_radar_chart(data):
    """Create interactive radar chart with golden theme"""
    audio_features = ['danceability', 'energy', 'loudness',
                     'acousticness', 'valence', 'tempo',
                     'speechiness', 'instrumentalness']
    
    genre_means = data.groupby('playlist_genre')[audio_features].mean().reset_index()

    # Define golden color palette for radar chart
    radar_colors = ['#D4AF37', '#DAA520', '#B8860B', '#CD853F', '#DEB887', '#F4A460']

    fig = go.Figure()
    for i, genre in enumerate(genre_means['playlist_genre']):
        fig.add_trace(go.Scatterpolar(
            r=genre_means[genre_means['playlist_genre'] == genre][audio_features].values[0],
            theta=audio_features,
            fill='toself',
            name=genre,
            line=dict(color=radar_colors[i % len(radar_colors)], width=3),
            fillcolor=radar_colors[i % len(radar_colors)],
            opacity=0.6
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, 
                range=[0, 1],
                tickfont=dict(color='#DEB887', size=12),
                gridcolor='#D4AF37',
                linecolor='#D4AF37'
            ),
            angularaxis=dict(
                tickfont=dict(color='#DEB887', size=12),
                gridcolor='#D4AF37',
                linecolor='#D4AF37'
            ),
            bgcolor='#2C1810'
        ),
        showlegend=True,
        title=dict(
            text='Audio Feature Profiles by Genre',
            font=dict(color='#D4AF37', size=18, family='Arial Black'),
            x=0.5
        ),
        height=600,
        paper_bgcolor='#2C1810',
        plot_bgcolor='#2C1810',
        font=dict(color='#DEB887'),
        legend=dict(
            font=dict(color='#DEB887', size=12),
            bgcolor='rgba(44, 24, 16, 0.8)',
            bordercolor='#D4AF37',
            borderwidth=1
        )
    )

    # Save the interactive plot as HTML and static image
    fig.write_html('/workspace/images/eda/radar_chart.html')
    fig.write_image('/workspace/images/eda/radar_chart.png', width=800, height=600)
    fig.show()
    print("💾 Interactive plot saved to: /workspace/images/eda/radar_chart.html")
    print("💾 Static plot saved to: /workspace/images/eda/radar_chart.png")

# ═══════════════════════════════════════════════════════════════
# 🔄 USAGE INSTRUCTIONS
# ═══════════════════════════════════════════════════════════════

"""
USAGE INSTRUCTIONS:

1. Replace your existing plotting cells in the notebook with the function calls:
   
   # Instead of the old genre distribution code, use:
   create_genre_distribution_plot(data)
   
   # Instead of the old correlation matrix code, use:
   create_correlation_matrix(data)
   
   # Instead of the old boxplot code, use:
   create_feature_boxplots(data)
   
   # Instead of the old radar chart code, use:
   create_radar_chart(data)

2. All plots will automatically be saved to /workspace/images/eda/ folder

3. The color scheme matches your presentation slide with:
   - Dark brown background (#2C1810)
   - Golden palette for data visualization
   - Elegant typography and styling

4. Make sure your data loading path is correct for your environment:
   - Colab: '/content/spotify_songs.csv'
   - Local: './data/spotify_songs.csv'
   - Current: '/workspace/data/spotify_songs.csv'
"""

print("🎵 Updated Feature Engineering Code Ready!")
print("📝 Copy the functions above into your notebook cells")
print("💫 Your visualizations will match the slide aesthetic perfectly!")