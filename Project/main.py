import subprocess
import sys
import os

def auto_install():
    flag_file = ".installed.flag"
    if not os.path.exists(flag_file):
        print("[INFO] Installing packages from instllation.txt ...")
        requirements_path = os.path.join(os.path.dirname(__file__), "installation.txt")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", "-r", requirements_path])
        with open(flag_file, "w") as f:
            f.write("done")
        print("[INFO] Restarting script to apply package updates...\n")
        os.execv(sys.executable, [sys.executable]+ sys.argv)

#auto_install()

import streamlit as st
import pandas as pd
import pydeck as pdk
import os
import plotly.express as px
from dotenv import load_dotenv
from utils.file_loader import load_transaction_csv
from train_models.anomaly_detector import run_anomaly_detection
from train_models.fraud_classifier import run_fraud_classification
from GenAI.location_summary_generator import generate_region_summary
from GenAI.fraud_mail_generator import generate_advisory_email
from utils.aggregator import group_fraud_summary
from utils.geo_mapper import map_locations_to_coordinates
from GenAI.location_summary_generator import generate_region_summary
from GenAI.fraud_mail_generator import generate_advisory_email
from GenAI.device_summary_generator import generate_device_summary
from utils.aggregator import group_fraud_summary

#Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="FraudSight AI", layout="wide")
st.title("Fraud Action Intelligence Dashboard")

with st.sidebar:
    st.header("Upload Transaction Data")
    region_dir = "data/regions"
    region_file = st.selectbox("Choose Region File", os.listdir(region_dir))
    st.session_state["selected_region"]= region_file.split('_',1)[0]

tabs = st.tabs(["ML Pipeline", "Action Advisor", "Dashboard"])

# ML Pipeline
with tabs[0]:
    st.header("ML Analysis for Region: " + region_file.replace(".csv", ""))
    with st.container(height=800):
        df = load_transaction_csv(os.path.join(region_dir, region_file))
        anomaly_df = run_anomaly_detection(df)
        st.subheader("Detected Anomalies")
        st.dataframe(anomaly_df, use_container_width=True)

        classified_df = run_fraud_classification(anomaly_df)
        st.subheader("Classified Frauds")
        st.dataframe(classified_df, use_container_width=True)
    
    # Bar chart data preparation
    st.subheader("Region-wise Fraud Summary (GenAI)")
    fraud_counts = classified_df["fraud_type"].value_counts().sort_index()
    fraud_counts_df = fraud_counts.reset_index()
    fraud_counts_df.columns = ["fraud_type", "count"]
    # Dynamic label generation: Each fraud_type code is split and capitalized
    fraud_counts_df["fraud_type_label"] = fraud_counts_df["fraud_type"].apply(
        lambda code: ' '.join([w.capitalize() for w in code.replace('_', ' ').split()])
    )

    fig = px.bar(
        fraud_counts_df,
        x="fraud_type_label",
        y="count",
        color_discrete_sequence=["#006EDB"],
        text="count",
        height=400,
    )
    fig.update_traces(
        marker_line_width=0,
        textposition="outside"
    )
    fig.update_layout(
        xaxis=dict(title=None, tickangle=90),  # vertical text labels for categories
        yaxis=dict(title=None, gridcolor="#F0F2F6"),
        showlegend=False,
        plot_bgcolor="#fff",
        margin=dict(l=40, r=40, b=40, t=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # GenAI Summary 
    summary = generate_region_summary(region_file, classified_df, api_key)
    st.session_state.summary = summary
    #st.text_area("Region summary (from GenAI)", summary, height=800)

    col1, col2 = st.columns(2)
    with col1:
            st.subheader("Top Branches with High Fraud Count")
            fraud_counts = group_fraud_summary(classified_df)
            st.dataframe(fraud_counts)
    with col2:
            st.subheader("Region-wise Fraud Summary (GenAI)")
            st.markdown(st.session_state.summary)
    

# Action Advisor
with tabs[1]:
    st.header("Fraud Action Advisor")
    if st.button(f"Generate Advisory Email"):
        st.success("Email generated successfully")
    locations = classified_df["location"].unique() if "location" in classified_df else []
    if len(locations) > 0:
        loc = st.selectbox("Select Branch Location", locations)
        email = generate_advisory_email(loc, classified_df, api_key)
        #st.markdown(email)
        st.text_area("Advisory Email Draft", email, height=300)
        if st.button(f"Send Email Alert to {loc}"):
            from utils.send_mail import send_advisory_email
            try:
                send_advisory_email(loc, email)
                st.success(f"Advisory email sent to branch {loc} successfully!")
            except Exception as e:
                st.error(f"Error sending email: {e}")
           
            
    else:
        st.warning("No location data available in file")

# Dashboard
# tab 2 - Dashboard
with tabs[2]:
    st.header("Consolidated Fraud Dashboard")
    region=st.session_state["selected_region"]
    st.info(f"Showing fraud insights for region: **{region}**")
    # --- 1. Top 5 High-Risk Locations (NEW PLOTLY CHART) ---
    st.subheader("Top 5 High-Risk Locations")
    
    # Prepare the data:
    # We only want fraud records
    fraud_df = classified_df[classified_df['predicted_fraud'] == 1]
    
    if not fraud_df.empty:
        # Get the value counts for locations, select the top 5
        location_counts = fraud_df['location'].value_counts().head(5)
        
        # Convert to DataFrame for Plotly
        top_5_df = location_counts.reset_index()
        top_5_df.columns = ['Location', 'Fraud Count']

        # Create the Plotly horizontal bar chart
        fig_top_5 = px.bar(
            top_5_df,
            x='Location',           # Values on X-axis
            y='Fraud Count',              # Categories on Y-axis
            #orientation='h',           # Make it horizontal
            text='Fraud Count',        # Show count on bars
            color_discrete_sequence=["#006EDB"] # Set bar color
        )

        # Style the figure to match your screenshot
        fig_top_5.update_traces(
            textposition="outside",
            marker_line_width=0
        )
        fig_top_5.update_layout(
            plot_bgcolor="#FFFFFF",
            # Sorts Y-axis to show the highest count at the top
            yaxis=dict(title=None, categoryorder='total ascending'), 
            xaxis=dict(title=None, showgrid=True, gridcolor="#F0F2F6"),
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
            height=300
        )
        
        st.plotly_chart(fig_top_5, use_container_width=True)
    
    else:
        st.info("No fraudulent activities detected to display top locations.")

    # --- 2. Fraud Map  ---
    st.subheader("Fraud Map")
    geo_df = map_locations_to_coordinates(classified_df)

    # compute total frauds per state
    state_total_frauds=geo_df.groupby("location").size().reset_index(name="total_frauds")
    geo_df=geo_df.merge(state_total_frauds, on="location", how="left")

    unique_states = geo_df.groupby("location").first().reset_index()

    #Prepare tooltip html for pydeck (state name, total frauds, fraud types)
    def prepare_tooltip(state):
        rows = geo_df[geo_df["location"] == state]
        counts = rows["fraud_type"].value_counts()
        lines = [f"- {k}: {v}" for k,v in counts.items()]
        return "<br>".join(lines)
    
    unique_states["tooltip"] = unique_states["location"].apply(
        lambda x : f"<b>{x}</b>: {unique_states[unique_states['location'] == x]['total_frauds'].values[0]} frauds<br>" + prepare_tooltip(x)
    )

    #PyDeck ScatterPlotLayer for orangedots with hoverinfo

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=unique_states,
        get_position='[longitude, latitude]',
        get_fill_color = [232, 119, 34], # Orange color
        get_radius=40000,
        radius_mix_pixels=10,
        radius_max_pixels=40,
        pickable=True,
        auto_higlight=True
    )

    view_state = pdk.ViewState(latitude=22.0, longitude=79.0, zoom=4)

    deck_map = pdk.Deck(
        # map_style='open-street-map',
        map_style='mapbox://styles/mapbox/light-v10',
        initial_view_state=view_state,
        layers=[layer],
        tooltip={"html":"{tooltip}", "style":{"color": "white"}},
    )

    # show interactive pydeck chart (orange dots with tooltips)
    st.pydeck_chart(deck_map)
    
    #st.map(map_df)
    
    
    st.divider()
    
    # --- 3. Device Fraud Footprint & Analyzer (From your full main.py) ---
    st.subheader("Device Fraud Footprint & Attack Pattern Analyzer")

    col1, col2 = st.columns([1, 1]) # Create two columns

    with col1:
        st.markdown("##### Fraud Distribution by Device")
        # Prepare data for pie chart
        # (This assumes fraud_df was defined above, which it now is)
        if not fraud_df.empty:
            device_counts = fraud_df['device_type'].value_counts()
            device_df = device_counts.reset_index()
            device_df.columns = ['device_type', 'count']
            
            # Create Plotly Pie Chart
            fig_pie = px.pie(
                device_df, 
                names='device_type', 
                values='count', 
                title="Fraud by Device Type"
            )
            fig_pie.update_layout(margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No fraudulent activities detected to analyze device footprint.")

    with col2:
        st.markdown("##### Gemini-Powered Device Risk Summary")
        device_summary = generate_device_summary(classified_df, api_key)
        st.markdown(device_summary)
    
    st.divider()

    # --- 4. Aggregate Fraud Counts (Original Code) ---
    st.subheader("Fraud Counts by Type and Branch")
    agg = group_fraud_summary(classified_df)
    # This original chart shows fraud type *and* branch
    st.bar_chart(agg)
