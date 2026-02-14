import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


# Page configuration
st.set_page_config(
    page_title="Cyber Robots AI - Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #16213e 0%, #0f3460 100%);
        border-right: 2px solid #4a9eff;
    }
    
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        color: #66b3ff !important;
        text-shadow: 0 0 8px rgba(102, 179, 255, 0.3);
    }
    
    p, div, span, label {
        font-family: 'Rajdhani', sans-serif;
        color: #e8e8e8;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #66b3ff;
        font-family: 'Orbitron', sans-serif;
    }
    
    .stButton button {
        background: linear-gradient(90deg, #4a9eff 0%, #357abd 100%);
        color: #ffffff;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-family: 'Orbitron', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(74, 158, 255, 0.2);
    }
    
    .stButton button:hover {
        background: linear-gradient(90deg, #66b3ff 0%, #4a9eff 100%);
        box-shadow: 0 6px 20px rgba(74, 158, 255, 0.4);
        transform: translateY(-2px);
    }
    
    [data-testid="stDataFrame"] {
        background: rgba(22, 33, 62, 0.6);
        border: 1px solid #4a9eff;
        border-radius: 8px;
    }
    
    .stAlert {
        background: rgba(74, 158, 255, 0.1);
        border-left: 4px solid #4a9eff;
        border-radius: 4px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(22, 33, 62, 0.4);
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #66b3ff;
        border-radius: 4px;
        font-family: 'Orbitron', sans-serif;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #4a9eff 0%, #357abd 100%);
        color: #ffffff;
    }
    
    [data-testid="stFileUploader"] {
        background: rgba(22, 33, 62, 0.4);
        border: 2px dashed #4a9eff;
        border-radius: 8px;
        padding: 2rem;
    }
    
    .stSuccess {
        background: rgba(76, 175, 80, 0.15);
        border-left: 4px solid #4caf50;
        color: #a5d6a7;
    }
    
    .stError {
        background: rgba(244, 67, 54, 0.15);
        border-left: 4px solid #f44336;
        color: #ef9a9a;
    }
    
    .stWarning {
        background: rgba(255, 152, 0, 0.15);
        border-left: 4px solid #ff9800;
        color: #ffcc80;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(22, 33, 62, 0.6) 0%, rgba(15, 52, 96, 0.6) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #4a9eff;
        box-shadow: 0 4px 15px rgba(74, 158, 255, 0.15);
        margin: 1rem 0;
    }
    
    .logo-container {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
        border-bottom: 2px solid #4a9eff;
    }
    
    .brand-name {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.5rem;
        font-weight: 900;
        color: #ffffff;
        text-shadow: 0 0 20px rgba(102, 179, 255, 0.5);
        letter-spacing: 3px;
        margin-top: 1rem;
    }
    
    .brand-tagline {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.2rem;
        color: #66b3ff;
        letter-spacing: 2px;
        margin-top: 0.5rem;
    }
    
    .chart-description {
        text-align: center;
        color: #b0b0b0;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        font-style: italic;
    }
    
    .status-card {
        background: linear-gradient(135deg, rgba(74, 158, 255, 0.1) 0%, rgba(53, 122, 189, 0.1) 100%);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #4a9eff;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load and train models

BASE_DIR = os.path.dirname(__file__)
file_path = os.path.join(BASE_DIR, "machinery_data.csv")

data = pd.read_csv(file_path)
data.ffill(inplace=True)

features = ['sensor_1', 'sensor_2', 'sensor_3', 'operational_hours']
target_rul = 'RUL'
target_maintenance = 'maintenance'
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(data[features], data[target_rul], test_size=0.2, random_state=42)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(data[features], data[target_maintenance], test_size=0.2, random_state=42)

reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train_reg, y_train_reg)
clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X_train_clf, y_train_clf)
kmeans = KMeans(n_clusters=2, random_state=42)
data['cluster'] = kmeans.fit_predict(data[features])

def predict_maintenance(features):
    features_df = pd.DataFrame([features], columns=['sensor_1', 'sensor_2', 'sensor_3', 'operational_hours'])
    rul_pred = reg_model.predict(features_df)
    maint_pred = clf_model.predict(features_df)
    cluster_pred = kmeans.predict(features_df)
    return {
        'RUL Prediction': rul_pred[0],
        'Maintenance Prediction': 'Needs Maintenance' if maint_pred[0] == 1 else 'Normal',
        'Anomaly Detection': 'Anomaly' if cluster_pred[0] == 1 else 'Normal'
    }

def predict_batch(uploaded_data):
    uploaded_data_normalized = uploaded_data.copy()
    uploaded_data_normalized[features] = scaler.transform(uploaded_data[features])
    
    rul_predictions = reg_model.predict(uploaded_data_normalized[features])
    maint_predictions = clf_model.predict(uploaded_data_normalized[features])
    cluster_predictions = kmeans.predict(uploaded_data_normalized[features])
    
    results = uploaded_data.copy()
    results['Predicted_RUL'] = rul_predictions
    results['Maintenance_Status'] = ['Needs Maintenance' if x == 1 else 'Normal' for x in maint_predictions]
    results['Anomaly_Status'] = ['Anomaly' if x == 1 else 'Normal' for x in cluster_predictions]
    
    return results

# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="logo-container">
        <div class="brand-name">CYBER ROBOTS</div>
        <div class="brand-tagline">PREDICTIVE AI</div>
    </div>
    """, unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Historical Data", "Input Data", "Results", "Visualizations"],
        icons=["house-fill", "database-fill", "cloud-upload-fill", "graph-up", "bar-chart-line-fill"],
        menu_icon="robot",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#66b3ff", "font-size": "20px"}, 
            "nav-link": {
                "font-family": "'Orbitron', sans-serif",
                "font-size": "14px",
                "text-align": "left",
                "margin": "5px",
                "color": "#e8e8e8",
                "border-radius": "8px",
            },
            "nav-link-selected": {
                "background": "linear-gradient(90deg, #4a9eff 0%, #357abd 100%)",
                "color": "#ffffff",
                "font-weight": "700",
            },
        }
    )

# Main content
if selected == "Home":
    st.markdown('<h1 style="text-align: center;">CYBER ROBOTS AI</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #66b3ff;">Industrial Predictive Maintenance System</h3>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Determine which data to show
    if 'batch_results' in st.session_state:
        current_data = st.session_state['batch_input']
        results_data = st.session_state['batch_results']
        data_source = "Uploaded Data"
    else:
        current_data = data
        results_data = data
        data_source = "Historical Data"
    
    # General Status Section
    st.markdown(f"### System Status - {data_source}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_records = len(current_data)
        st.metric("Total Records", total_records)
    
    with col2:
        if 'Maintenance_Status' in results_data.columns:
            needs_maint = (results_data['Maintenance_Status'] == 'Needs Maintenance').sum()
            maint_rate = (needs_maint / total_records * 100)
        else:
            maint_rate = (results_data[target_maintenance].sum() / total_records * 100)
        st.metric("Maintenance Rate", f"{maint_rate:.1f}%")
    
    with col3:
        if 'Predicted_RUL' in results_data.columns:
            avg_rul = results_data['Predicted_RUL'].mean()
        else:
            avg_rul = results_data['RUL'].mean()
        st.metric("Avg RUL", f"{avg_rul:.0f} hrs")
    
    with col4:
        if 'Anomaly_Status' in results_data.columns:
            anomalies = (results_data['Anomaly_Status'] == 'Anomaly').sum()
            anomaly_rate = (anomalies / total_records * 100)
        else:
            anomaly_rate = 0
        st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
    
    st.markdown("---")
    
    # Interactive Visualizations
    st.markdown("### Real-Time Analytics Dashboard")
    
    # Row 1: 3D Scatter Plot & Gauge Chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 3D Sensor Space")
        fig = go.Figure(data=[go.Scatter3d(
            x=current_data['sensor_1'],
            y=current_data['sensor_2'],
            z=current_data['sensor_3'],
            mode='markers',
            marker=dict(
                size=4,
                color=current_data['operational_hours'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Op. Hours"),
                line=dict(width=0.5, color='white')
            )
        )])
        fig.update_layout(
            scene=dict(
                xaxis_title='Sensor 1',
                yaxis_title='Sensor 2',
                zaxis_title='Sensor 3',
                bgcolor='#1a1a2e'
            ),
            paper_bgcolor='#16213e',
            font=dict(color='white'),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<p class="chart-description">Interactive 3D visualization of sensor readings colored by operational hours. Rotate and zoom to explore data patterns.</p>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### System Health Gauge")
        health_score = 100 - maint_rate
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=health_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Health Score", 'font': {'color': 'white'}},
            delta={'reference': 80, 'increasing': {'color': "#4caf50"}},
            gauge={
                'axis': {'range': [None, 100], 'tickcolor': 'white'},
                'bar': {'color': "#4a9eff"},
                'bgcolor': "#1a1a2e",
                'borderwidth': 2,
                'bordercolor': "white",
                'steps': [
                    {'range': [0, 50], 'color': '#f44336'},
                    {'range': [50, 75], 'color': '#ff9800'},
                    {'range': [75, 100], 'color': '#4caf50'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(
            paper_bgcolor='#16213e',
            font={'color': 'white'},
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<p class="chart-description">Real-time system health indicator based on maintenance requirements. Higher scores indicate better equipment condition.</p>', unsafe_allow_html=True)
    
    # Row 2: Distribution Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Maintenance Status Distribution")
        if 'Maintenance_Status' in results_data.columns:
            status_counts = results_data['Maintenance_Status'].value_counts()
        else:
            status_counts = pd.Series({
                'Normal': (results_data[target_maintenance] == 0).sum(),
                'Needs Maintenance': (results_data[target_maintenance] == 1).sum()
            })
        
        fig = go.Figure(data=[go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            hole=.4,
            marker_colors=['#4caf50', '#f44336']
        )])
        fig.update_layout(
            paper_bgcolor='#16213e',
            font=dict(color='white'),
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<p class="chart-description">Distribution of equipment requiring maintenance vs. normal operation. Lower maintenance needs indicate better predictive maintenance.</p>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### RUL Distribution")
        if 'Predicted_RUL' in results_data.columns:
            rul_data = results_data['Predicted_RUL']
        else:
            rul_data = results_data['RUL']
        
        fig = go.Figure(data=[go.Histogram(
            x=rul_data,
            nbinsx=30,
            marker_color='#4a9eff',
            opacity=0.75
        )])
        fig.update_layout(
            xaxis_title="Remaining Useful Life (hours)",
            yaxis_title="Frequency",
            paper_bgcolor='#16213e',
            plot_bgcolor='#1a1a2e',
            font=dict(color='white'),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<p class="chart-description">Histogram showing the distribution of predicted remaining useful life across all equipment. Plan maintenance for low RUL values.</p>', unsafe_allow_html=True)
    
    # Row 3: Time Series
    st.markdown("#### Sensor Trends Over Operational Time")
    fig = make_subplots(rows=1, cols=1)
    
    colors_sensors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    for i, (sensor, color) in enumerate(zip(['sensor_1', 'sensor_2', 'sensor_3'], colors_sensors)):
        fig.add_trace(go.Scatter(
            x=current_data['operational_hours'],
            y=current_data[sensor],
            mode='markers',
            name=f'Sensor {i+1}',
            marker=dict(color=color, size=5, opacity=0.6)
        ))
    
    fig.update_layout(
        xaxis_title="Operational Hours",
        yaxis_title="Sensor Values",
        paper_bgcolor='#16213e',
        plot_bgcolor='#1a1a2e',
        font=dict(color='white'),
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<p class="chart-description">Scatter plot showing sensor readings across operational hours. Each sensor uses a distinct color for easy differentiation.</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Capabilities Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="text-align: center;">Real-Time Analysis</h3>
            <p style="text-align: center;">Process sensor data instantly with AI-powered predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="text-align: center;">High Accuracy</h3>
            <p style="text-align: center;">Advanced ML models for precise maintenance forecasting</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="text-align: center;">Smart Analytics</h3>
            <p style="text-align: center;">Comprehensive visualizations and insights</p>
        </div>
        """, unsafe_allow_html=True)

elif selected == "Historical Data":
    st.markdown('<h1>Historical Sensor Data</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(data))
    with col2:
        st.metric("Sensors", "3")
    with col3:
        maintenance_rate = (data[target_maintenance].sum() / len(data) * 100)
        st.metric("Maintenance Rate", f"{maintenance_rate:.1f}%")
    with col4:
        avg_rul = data[target_rul].mean()
        st.metric("Avg RUL", f"{avg_rul:.0f} hrs")
    
    st.markdown("### Data Preview")
    st.dataframe(data.head(20), use_container_width=True)
    
    st.download_button(
        label="Download Sample Format",
        data=data[['sensor_1', 'sensor_2', 'sensor_3', 'operational_hours']].head(10).to_csv(index=False),
        file_name='sample_sensor_data.csv',
        mime='text/csv'
    )

elif selected == "Input Data":
    st.markdown('<h1>Input Sensor Data</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["Upload CSV", "Generate Random", "Manual Input"])
    
    with tab1:
        st.markdown("### Upload Sensor Data")
        st.info("Required columns: sensor_1, sensor_2, sensor_3, operational_hours")
        
        uploaded_file = st.file_uploader("Drop your CSV file here", type=['csv'])
        
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                required_columns = ['sensor_1', 'sensor_2', 'sensor_3', 'operational_hours']
                missing_columns = [col for col in required_columns if col not in uploaded_df.columns]
                
                if missing_columns:
                    st.error(f"Missing columns: {', '.join(missing_columns)}")
                else:
                    st.success(f"Successfully loaded {len(uploaded_df)} records!")
                    st.dataframe(uploaded_df.head(), use_container_width=True)
                    
                    if st.button("Process Data", key="process_csv"):
                        with st.spinner("Processing..."):
                            results_df = predict_batch(uploaded_df)
                            st.session_state['batch_results'] = results_df
                            st.session_state['batch_input'] = uploaded_df
                            st.success("Analysis Complete! Check the Home page for updated dashboard.")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                needs_maintenance = (results_df['Maintenance_Status'] == 'Needs Maintenance').sum()
                                st.metric("Needs Maintenance", f"{needs_maintenance}/{len(results_df)}")
                            with col2:
                                anomalies = (results_df['Anomaly_Status'] == 'Anomaly').sum()
                                st.metric("Anomalies", f"{anomalies}/{len(results_df)}")
                            with col3:
                                avg_rul = results_df['Predicted_RUL'].mean()
                                st.metric("Avg RUL", f"{avg_rul:.1f} hrs")
                            
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Results",
                                data=csv,
                                file_name='prediction_results.csv',
                                mime='text/csv'
                            )
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with tab2:
        st.markdown("### Generate Test Data")
        
        if st.button('Generate Random Values', use_container_width=True):
            sensor_1 = np.random.uniform(data['sensor_1'].min(), data['sensor_1'].max())
            sensor_2 = np.random.uniform(data['sensor_2'].min(), data['sensor_2'].max())
            sensor_3 = np.random.uniform(data['sensor_3'].min(), data['sensor_3'].max())
            operational_hours = np.random.uniform(data['operational_hours'].min(), data['operational_hours'].max())
            st.session_state['generated_values'] = [sensor_1, sensor_2, sensor_3, operational_hours]
            st.success("Random values generated!")

        if 'generated_values' in st.session_state and st.session_state['generated_values'] is not None:
            vals = st.session_state['generated_values']
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sensor 1", f"{vals[0]:.3f}")
                st.metric("Sensor 2", f"{vals[1]:.3f}")
            with col2:
                st.metric("Sensor 3", f"{vals[2]:.3f}")
                st.metric("Op. Hours", f"{vals[3]:.1f}")

            if st.button('Use These Values', use_container_width=True):
                st.session_state['input_features'] = st.session_state['generated_values']
                st.success("Values saved! Check Results page.")
    
    with tab3:
        st.markdown("### Manual Input")
        sensor_1 = st.slider('Sensor 1', float(data['sensor_1'].min()), float(data['sensor_1'].max()), float(data['sensor_1'].mean()))
        sensor_2 = st.slider('Sensor 2', float(data['sensor_2'].min()), float(data['sensor_2'].max()), float(data['sensor_2'].mean()))
        sensor_3 = st.slider('Sensor 3', float(data['sensor_3'].min()), float(data['sensor_3'].max()), float(data['sensor_3'].mean()))
        operational_hours = st.slider('Operational Hours', int(data['operational_hours'].min()), int(data['operational_hours'].max()), int(data['operational_hours'].mean()))

        if st.button('Submit', use_container_width=True):
            st.session_state['input_features'] = [sensor_1, sensor_2, sensor_3, operational_hours]
            st.success("Data submitted! Navigate to Results.")

elif selected == "Results":
    st.markdown('<h1>AI Predictions</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    if 'batch_results' in st.session_state:
        results_df = st.session_state['batch_results']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            needs_maintenance = (results_df['Maintenance_Status'] == 'Needs Maintenance').sum()
            st.metric("Maintenance Required", f"{needs_maintenance}/{len(results_df)}")
        with col2:
            anomalies = (results_df['Anomaly_Status'] == 'Anomaly').sum()
            st.metric("Anomalies Detected", f"{anomalies}/{len(results_df)}")
        with col3:
            avg_rul = results_df['Predicted_RUL'].mean()
            st.metric("Average RUL", f"{avg_rul:.1f} hrs")
        
        st.markdown("### Full Results")
        st.dataframe(results_df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Maintenance Status")
            fig, ax = plt.subplots(figsize=(8, 6))
            fig.patch.set_facecolor('#16213e')
            ax.set_facecolor('#16213e')
            maintenance_counts = results_df['Maintenance_Status'].value_counts()
            colors = ['#4caf50', '#f44336']
            wedges, texts, autotexts = ax.pie(maintenance_counts.values, labels=maintenance_counts.index, 
                                               autopct='%1.1f%%', colors=colors, textprops={'color': 'white', 'fontsize': 12})
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("#### Anomaly Detection")
            fig, ax = plt.subplots(figsize=(8, 6))
            fig.patch.set_facecolor('#16213e')
            ax.set_facecolor('#16213e')
            anomaly_counts = results_df['Anomaly_Status'].value_counts()
            wedges, texts, autotexts = ax.pie(anomaly_counts.values, labels=anomaly_counts.index, 
                                               autopct='%1.1f%%', colors=colors, textprops={'color': 'white', 'fontsize': 12})
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            st.pyplot(fig)
            plt.close()
        
    elif 'input_features' in st.session_state:
        prediction = predict_maintenance(st.session_state['input_features'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Remaining Useful Life", f"{prediction['RUL Prediction']:.2f} hrs")
        with col2:
            status = prediction['Maintenance Prediction']
            st.metric("Maintenance Status", status)
            if status == 'Needs Maintenance':
                st.error('Maintenance Required!')
        with col3:
            anomaly = prediction['Anomaly Detection']
            st.metric("Anomaly Status", anomaly)
            if anomaly == 'Anomaly':
                st.warning('Anomaly Detected!')
    else:
        st.warning("No data to analyze. Please submit data in the Input Data section.")

elif selected == "Visualizations":
    st.markdown('<h1>Data Visualizations</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    if 'batch_input' in st.session_state:
        viz_data = st.session_state['batch_input']
        st.info("Showing visualizations for uploaded data")
    else:
        viz_data = data
        st.info("Showing visualizations for historical data")
    
    # 1. Histogram
    st.markdown("### Histogram of Sensor Readings")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('#16213e')
    
    colors = ['#4a9eff', '#66b3ff', '#80c1ff']
    
    for i, (ax, sensor, color) in enumerate(zip(axs, ['sensor_1', 'sensor_2', 'sensor_3'], colors)):
        ax.set_facecolor('#1a1a2e')
        sns.histplot(viz_data[sensor], bins=30, ax=ax, kde=True, color=color, alpha=0.7)
        ax.set_title(f'Sensor {i+1}', color='white', fontsize=14, fontweight='bold')
        ax.set_xlabel('Value', color='white')
        ax.set_ylabel('Frequency', color='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2, color='white')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.markdown('<p class="chart-description">Distribution of sensor readings showing frequency and density. KDE curves indicate the probability density of values.</p>', unsafe_allow_html=True)
    
    # 2. Scatter Plot with different colors
    st.markdown("### Scatter Plot of Sensor Readings vs Operational Hours")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('#16213e')
    
    scatter_colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']  # Red, Teal, Blue
    
    for i, (ax, sensor, color) in enumerate(zip(axs, ['sensor_1', 'sensor_2', 'sensor_3'], scatter_colors)):
        ax.set_facecolor('#1a1a2e')
        ax.scatter(viz_data['operational_hours'], viz_data[sensor], alpha=0.6, c=color, s=30, edgecolors='white', linewidth=0.5)
        ax.set_title(f'Operational Hours vs Sensor {i+1}', color='white', fontsize=14, fontweight='bold')
        ax.set_xlabel('Operational Hours', color='white')
        ax.set_ylabel(f'Sensor {i+1}', color='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2, color='white')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.markdown('<p class="chart-description">Correlation between sensor readings and operational hours. Each sensor uses a distinct color (red, teal, blue) for clear identification.</p>', unsafe_allow_html=True)
    
    # 3. Line Chart
    if 'RUL' in viz_data.columns:
        st.markdown("### Line Chart of RUL Over Time")
        fig, ax = plt.subplots(figsize=(18, 6))
        fig.patch.set_facecolor('#16213e')
        ax.set_facecolor('#1a1a2e')
        
        ax.plot(viz_data['operational_hours'], viz_data['RUL'], marker='o', linestyle='-', 
                color='#4a9eff', linewidth=2, markersize=4, alpha=0.8)
        ax.set_title('RUL Over Operational Hours', color='white', fontsize=16, fontweight='bold')
        ax.set_xlabel('Operational Hours', color='white', fontsize=12)
        ax.set_ylabel('RUL (Remaining Useful Life)', color='white', fontsize=12)
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2, color='white')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('<p class="chart-description">Trend of remaining useful life across operational hours. Declining RUL indicates approaching maintenance needs.</p>', unsafe_allow_html=True)
    
    # 4. Correlation Heatmap
    st.markdown("### Sensor Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#16213e')
    
    correlation = viz_data[['sensor_1', 'sensor_2', 'sensor_3', 'operational_hours']].corr()
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                square=True, ax=ax, cbar_kws={'label': 'Correlation'},
                linewidths=1, linecolor='white')
    ax.set_title('Sensor Correlation Matrix', color='white', fontsize=16, fontweight='bold', pad=20)
    plt.setp(ax.get_xticklabels(), color='white')
    plt.setp(ax.get_yticklabels(), color='white')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.markdown('<p class="chart-description">Correlation matrix showing relationships between sensors. Values range from -1 (negative correlation) to +1 (positive correlation).</p>', unsafe_allow_html=True)
    
    # 5. Box Plots
    st.markdown("### Sensor Value Distribution (Box Plots)")
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#16213e')
    ax.set_facecolor('#1a1a2e')
    
    sensor_data = [viz_data['sensor_1'], viz_data['sensor_2'], viz_data['sensor_3']]
    bp = ax.boxplot(sensor_data, labels=['Sensor 1', 'Sensor 2', 'Sensor 3'],
                    patch_artist=True, notch=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='white')
    
    ax.set_title('Sensor Value Distribution', color='white', fontsize=16, fontweight='bold')
    ax.set_ylabel('Sensor Values', color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2, color='white', axis='y')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<p class="chart-description">Box plots showing statistical distribution of sensor values including median, quartiles, and outliers.</p>', unsafe_allow_html=True)

