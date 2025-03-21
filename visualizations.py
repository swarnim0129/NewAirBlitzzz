import plotly.express as px
import plotly.graph_objects as go
import folium
import pandas as pd
import numpy as np
from data_processor import AirQualityDataProcessor
import plotly.figure_factory as ff
from scipy import stats
from folium.plugins import HeatMap
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class AirQualityVisualizer:
    def __init__(self):
        self.data_processor = AirQualityDataProcessor()
        self.df = self.data_processor.df
        
    def create_3d_boxplot(self, pollutant):
        """Create a 3D boxplot for pollutant levels"""
        fig = go.Figure()
        
        for quality in self.df['air_quality'].unique():
            mask = self.df['air_quality'] == quality
            fig.add_trace(go.Box(
                y=self.df[mask][pollutant],
                name=quality,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8,
                marker=dict(
                    color=self._get_quality_color(quality),
                    size=4,
                    line=dict(color='white', width=1)
                )
            ))
        
        fig.update_layout(
            title=f'3D Boxplot of {pollutant} by Air Quality',
            yaxis_title=f'{pollutant} Concentration',
            showlegend=True,
            template='plotly_dark',
            height=600
        )
        
        return fig
    
    def create_force_directed_graph(self):
        """Create a force-directed graph showing pollutant relationships"""
        corr_matrix = self.df[['pm25', 'pm10', 'no2', 'so2', 'co']].corr()
        
        nodes = []
        edges = []
        
        # Create nodes first
        for pollutant in corr_matrix.columns:
            nodes.append(dict(
                id=pollutant,
                label=pollutant,
                color=self._get_pollutant_color(pollutant),
                size=30
            ))
        
        # Create edges
        for i, pollutant in enumerate(corr_matrix.columns):
            for j, target in enumerate(corr_matrix.columns):
                if i < j:  # Avoid duplicate edges
                    correlation = corr_matrix.iloc[i, j]
                    if abs(correlation) > 0.3:  # Only show significant correlations
                        edges.append(dict(
                            source=i,  # Use index directly
                            target=j,  # Use index directly
                            value=abs(correlation),
                            color='red' if correlation > 0 else 'blue'
                        ))
        
        fig = go.Figure(
            data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="white", width=0.5),
                    label=[node['label'] for node in nodes],
                    color=[node['color'] for node in nodes]
                ),
                link=dict(
                    source=[edge['source'] for edge in edges],
                    target=[edge['target'] for edge in edges],
                    value=[edge['value'] for edge in edges],
                    color=[edge['color'] for edge in edges]
                )
            )]
        )
        
        fig.update_layout(
            title="Pollutant Relationship Network",
            font_size=12,
            template='plotly_dark',
            height=600
        )
        
        return fig
    
    def create_animated_line_chart(self, pollutant):
        """Create an animated line chart for pollutant trends"""
        fig = px.line(
            self.df,
            x='timestamp',
            y=pollutant,
            color='air_quality',
            animation_frame='timestamp',
            template='plotly_dark',
            title=f'Animated {pollutant} Trends by Air Quality'
        )
        
        fig.update_layout(
            height=500,
            showlegend=True,
            xaxis_title="Time",
            yaxis_title=f"{pollutant} Concentration"
        )
        
        return fig
    
    def create_heatmap(self):
        """Create a heatmap of pollution levels"""
        # Create a simple correlation heatmap instead of geographic heatmap
        corr_matrix = self.df[['pm25', 'pm10', 'no2', 'so2', 'co']].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title='Pollutant Correlation Heatmap',
            height=500
        )
        
        return fig
    
    def create_3d_bar_chart(self):
        """Create a 3D bar chart of pollutant levels"""
        pollutants = ['pm25', 'pm10', 'no2', 'so2', 'co']
        qualities = self.df['air_quality'].unique()
        
        fig = go.Figure(data=[
            go.Bar(
                name=quality,
                x=pollutants,
                y=[self.df[self.df['air_quality'] == quality][p].mean() 
                   for p in pollutants],
                marker_color=self._get_quality_color(quality)
            ) for quality in qualities
        ])
        
        fig.update_layout(
            title='3D Bar Chart of Average Pollutant Levels by Air Quality',
            barmode='group',
            template='plotly_dark',
            height=600,
            scene=dict(
                xaxis_title='Pollutants',
                yaxis_title='Concentration',
                zaxis_title='Air Quality'
            )
        )
        
        return fig
    
    def _get_quality_color(self, quality):
        """Get color for air quality category"""
        colors = {
            'Good': '#00ff00',
            'Moderate': '#ffff00',
            'Poor': '#ff9900',
            'Hazardous': '#ff0000'
        }
        return colors.get(quality, '#808080')
    
    def _get_pollutant_color(self, pollutant):
        """Get color for pollutant type"""
        colors = {
            'pm25': '#ff0000',
            'pm10': '#ff6600',
            'no2': '#ffcc00',
            'so2': '#00cc00',
            'co': '#0000ff'
        }
        return colors.get(pollutant, '#808080')
    
    def create_3d_surface_plot(self, pollutant, param1, param2):
        """Create a 3D surface plot showing pollutant levels across two parameters"""
        fig = go.Figure(data=[go.Surface(z=self.df[pollutant].values.reshape(10, 10),
                                       x=self.df[param1].values.reshape(10, 10),
                                       y=self.df[param2].values.reshape(10, 10))])
        
        fig.update_layout(
            title=f'3D Surface Plot: {pollutant.upper()} vs {param1} and {param2}',
            scene=dict(
                xaxis_title=param1,
                yaxis_title=param2,
                zaxis_title=pollutant.upper()
            ),
            autosize=True
        )
        return fig
    
    def create_3d_scatter_plot(self, pollutant, param1, param2):
        """Create a 3D scatter plot showing pollutant levels across two parameters"""
        fig = go.Figure(data=[go.Scatter3d(
            x=self.df[param1],
            y=self.df[param2],
            z=self.df[pollutant],
            mode='markers',
            marker=dict(
                size=8,
                color=self.df[pollutant],
                colorscale='Viridis',
                opacity=0.8
            )
        )])
        
        fig.update_layout(
            title=f'3D Scatter Plot: {pollutant.upper()} vs {param1} and {param2}',
            scene=dict(
                xaxis_title=param1,
                yaxis_title=param2,
                zaxis_title=pollutant.upper()
            ),
            autosize=True
        )
        return fig
    
    def create_3d_bar_plot(self, pollutant):
        """Create a 3D bar plot showing pollutant levels across time"""
        fig = go.Figure(data=[go.Bar3d(
            x=self.df.index,
            y=[pollutant] * len(self.df),
            z=self.df[pollutant],
            marker=dict(
                color=self.df[pollutant],
                colorscale='Viridis',
                opacity=0.8
            )
        )])
        
        fig.update_layout(
            title=f'3D Bar Plot: {pollutant.upper()} Levels Over Time',
            scene=dict(
                xaxis_title='Time',
                yaxis_title='Pollutant',
                zaxis_title='Level'
            ),
            autosize=True
        )
        return fig
    
    def create_3d_density_plot(self, pollutant, param1, param2):
        """Create a 3D density plot showing pollutant distribution"""
        # Create a 3D histogram
        fig = go.Figure(data=[go.Histogram2d(
            x=self.df[param1],
            y=self.df[param2],
            z=self.df[pollutant],
            colorscale='Viridis',
            histnorm='probability'
        )])
        
        fig.update_layout(
            title=f'3D Density Plot: {pollutant.upper()} Distribution',
            xaxis_title=param1,
            yaxis_title=param2,
            autosize=True
        )
        return fig
    
    def create_3d_vector_field(self, pollutant, param1, param2):
        """Create a 3D vector field plot showing pollutant gradients"""
        # Calculate gradients
        dx = np.gradient(self.df[pollutant].values.reshape(10, 10), axis=1)
        dy = np.gradient(self.df[pollutant].values.reshape(10, 10), axis=0)
        
        fig = go.Figure(data=[go.Cone(
            x=self.df[param1].values.reshape(10, 10),
            y=self.df[param2].values.reshape(10, 10),
            z=self.df[pollutant].values.reshape(10, 10),
            u=dx,
            v=dy,
            w=np.zeros_like(dx),
            colorscale='Viridis',
            sizeref=0.3,
            sizemode='absolute'
        )])
        
        fig.update_layout(
            title=f'3D Vector Field: {pollutant.upper()} Gradients',
            scene=dict(
                xaxis_title=param1,
                yaxis_title=param2,
                zaxis_title=pollutant.upper()
            ),
            autosize=True
        )
        return fig
    
    def create_3d_network_graph(self):
        """Create a 3D network graph showing pollutant correlations"""
        # Calculate correlation matrix
        corr_matrix = self.df[['pm25', 'pm10', 'no2', 'so2', 'co']].corr()
        
        # Create nodes
        nodes = [dict(name=pollutant, x=0, y=0, z=0) for pollutant in ['pm25', 'pm10', 'no2', 'so2', 'co']]
        
        # Create edges
        edges = []
        for i in range(len(['pm25', 'pm10', 'no2', 'so2', 'co'])):
            for j in range(i+1, len(['pm25', 'pm10', 'no2', 'so2', 'co'])):
                if abs(corr_matrix.iloc[i,j]) > 0.5:  # Only show strong correlations
                    edges.append(dict(
                        source=i,
                        target=j,
                        value=abs(corr_matrix.iloc[i,j])
                    ))
        
        fig = go.Figure(data=[go.Scatter3d(
            x=[node['x'] for node in nodes],
            y=[node['y'] for node in nodes],
            z=[node['z'] for node in nodes],
            mode='markers+text',
            marker=dict(size=10),
            text=[node['name'] for node in nodes],
            textposition="top center",
        )])
        
        # Add edges
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=[nodes[edge['source']]['x'], nodes[edge['target']]['x']],
                y=[nodes[edge['source']]['y'], nodes[edge['target']]['y']],
                z=[nodes[edge['source']]['z'], nodes[edge['target']]['z']],
                mode='lines',
                line=dict(width=edge['value']*2)
            ))
        
        fig.update_layout(
            title='3D Network Graph: Pollutant Correlations',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            autosize=True
        )
        return fig
    
    def create_3d_contour_plot(self, pollutant, param1, param2):
        """Create a 3D contour plot showing pollutant levels"""
        fig = go.Figure(data=[go.Surface(
            z=self.df[pollutant].values.reshape(10, 10),
            x=self.df[param1].values.reshape(10, 10),
            y=self.df[param2].values.reshape(10, 10),
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
            )
        )])
        
        fig.update_layout(
            title=f'3D Contour Plot: {pollutant.upper()} Levels',
            scene=dict(
                xaxis_title=param1,
                yaxis_title=param2,
                zaxis_title=pollutant.upper()
            ),
            autosize=True
        )
        return fig
    
    def create_3d_ternary_plot(self, pollutant1, pollutant2, pollutant3):
        """Create a ternary plot showing three pollutant relationships"""
        fig = go.Figure(data=[go.Scatterternary(
            a=self.df[pollutant1],
            b=self.df[pollutant2],
            c=self.df[pollutant3],
            mode='markers',
            marker=dict(
                size=8,
                color=self.df[pollutant1],
                colorscale='Viridis',
                opacity=0.8
            )
        )])
        
        fig.update_layout(
            title=f'Ternary Plot: {pollutant1.upper()}, {pollutant2.upper()}, {pollutant3.upper()}',
            ternary=dict(
                aaxis_title=pollutant1.upper(),
                baxis_title=pollutant2.upper(),
                caxis_title=pollutant3.upper()
            ),
            autosize=True
        )
        return fig 