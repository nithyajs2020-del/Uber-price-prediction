import streamlit as st
import pandas as pd
import numpy as np
import pickle
import folium
from streamlit_folium import st_folium
import datetime # Import datetime module

# Set page configuration for a wider layout
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# Custom CSS for black background, white text, and enhanced styling
st.markdown("""
<style>
/* Main app container */
.stApp {
    background-color: black;
    color: white;
    font-family: 'Arial', sans-serif;
}
/* Headers */
h1, h2, h3, h4, h5, h6, .stMarkdown {
    color: white;
}
/* Center the main title */
h1 {
    text-align: center;
}
/* Sidebar specific styling */
.st-emotion-cache-vk3357.e1nx6e3q1 {
    background-color: black; /* Sidebar background */
    color: white; /* Sidebar text */
}
.st-emotion-cache-vk3357.e1nx6e3q1 p {
    color: white; /* Ensure text in sidebar also white */
}
/* Input widgets */
.stSlider > label {
    color: orange; /* Slider label */
}
.stSlider [data-baseweb="slider"] div:nth-child(1) {
    background-color: #555; /* Slider track */
}
.stSlider [data-baseweb="slider"] div:nth-child(2) {
    background-color: #FFA500; /* Slider fill - Orange */
}
.stSlider [data-baseweb="slider"] div:nth-child(3) {
    background-color: #FFA500; /* Slider thumb - Orange */
}
/* Passenger Count specific styling (to ensure white text) */
.stSlider div[data-testid="stTickBarMinMax"] p {
    color: orange; /* Min/Max values */
}
.stSlider div[data-testid="stThumbValue"] p {
    color: orange; /* Current thumb value */
}
.stDateInput label, .stTimeInput label {
    color: white; /* Date/Time input labels */
}
.stDateInput input, .stTimeInput input {
    background-color: #333; /* Date/Time input background */
    color: white; /* Date/Time input text */
    border: 1px solid #555;
}
.stSelectbox label {
    color: white; /* Selectbox label */
}
.stSelectbox [data-baseweb="select"] > div {
    background-color: #333; /* Selectbox background */
    color: white; /* Selectbox text */
    border: 1px solid #555;
}
.stSelectbox [data-baseweb="select"] > div > div:first-child {
    color: white; /* Selected value in selectbox */
}
.stButton > button {
    background-color: #007bff; /* Predict button background */
    color: white; /* Predict button text */
    border-radius: 5px;
    border: none;
    padding: 10px 20px;
    font-size: 16px;
}
.stButton > button:hover {
    background-color: #0056b3;
}
.stAlert {
    background-color: #28a745; /* Success message background */
    color: white; /* Success message text */
}
</style>
""", unsafe_allow_html=True)

# Load the trained model
with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the scalers
with open('x_scaler.pkl', 'rb') as file:
    x_scaler = pickle.load(file)
with open('y_scaler.pkl', 'rb') as file:
    y_scaler = pickle.load(file)

# Load the one-hot encoded column names
with open('one_hot_encoded_columns.pkl', 'rb') as file:
    one_hot_encoded_columns = pickle.load(file)

# Haversine distance function (from notebook)
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance

# Streamlit app layout
st.title('🚗 Uber Fare Prediction App')

st.write("Enter the details of your trip to predict the fare amount.")

# Initialize session state for coordinates and mode
if 'pickup_coords' not in st.session_state:
    st.session_state['pickup_coords'] = [40.738354, -73.999817] # Default NYC area
if 'dropoff_coords' not in st.session_state:
    st.session_state['dropoff_coords'] = [40.723217, -73.999512] # Default NYC area
if 'selection_mode' not in st.session_state:
    st.session_state['selection_mode'] = 'pickup' # 'pickup' or 'dropoff'

# Create two columns for map and inputs
col1, col2 = st.columns([2, 1]) # Map takes 2/3 width, inputs 1/3

with col1:
    st.subheader("Select Pickup and Dropoff Locations")
    # Create a Folium map
    m = folium.Map(location=[(st.session_state['pickup_coords'][0] + st.session_state['dropoff_coords'][0])/2,
                             (st.session_state['pickup_coords'][1] + st.session_state['dropoff_coords'][1])/2],
                   zoom_start=12) # Reverted zoom for NYC area

    # Add markers for pickup and dropoff
    folium.Marker(
        location=st.session_state['pickup_coords'],
        popup="Pickup",
        icon=folium.Icon(color="green")
    ).add_to(m)

    folium.Marker(
        location=st.session_state['dropoff_coords'],
        popup="Dropoff",
        icon=folium.Icon(color="red")
    ).add_to(m)

    # Display the map and capture clicks
    map_output = st_folium(m, width=700, height=500, key="folium_map") # Changed width to 700

    # Handle map clicks
    if map_output and "last_clicked" in map_output and map_output["last_clicked"] is not None:
        lat, lon = map_output["last_clicked"]["lat"], map_output["last_clicked"]["lng"]
        if st.session_state['selection_mode'] == 'pickup':
            st.session_state['pickup_coords'] = [lat, lon]
            st.session_state['selection_mode'] = 'dropoff' # Switch to dropoff mode
        else: # selection_mode == 'dropoff'
            st.session_state['dropoff_coords'] = [lat, lon]
            st.session_state['selection_mode'] = 'pickup' # Switch back to pickup mode

    st.write(f"Current selection mode: **{st.session_state['selection_mode'].capitalize()}**")
    st.write(f"Pickup: Lat {st.session_state['pickup_coords'][0]:.6f}, Lon {st.session_state['pickup_coords'][1]:.6f}")
    st.write(f"Dropoff: Lat {st.session_state['dropoff_coords'][0]:.6f}, Lon {st.session_state['dropoff_coords'][1]:.6f}")

pickup_latitude = st.session_state['pickup_coords'][0]
pickup_longitude = st.session_state['pickup_coords'][1]
dropoff_latitude = st.session_state['dropoff_coords'][0]
dropoff_longitude = st.session_state['dropoff_coords'][1]

with col2:
    st.subheader("Trip Details")
    passenger_count = st.selectbox('Passenger Count', list(range(1, 7)), index=0) # Changed to selectbox

    # Single date input
    trip_date = st.date_input('Date', datetime.date(2015, 5, 7), min_value=datetime.date(2009, 1, 1), max_value=datetime.date(2026, 12, 31)) # Updated max_value
    year = trip_date.year
    month = trip_date.month
    day = trip_date.day
    day_of_week = trip_date.weekday() # Monday=0, Sunday=6

    # Time input widget
    trip_time = st.time_input('Time', datetime.time(9, 0)) # Default to 9 AM
    hour = trip_time.hour

    st.write(f"Selected Time: {trip_time.strftime('%I:%M %p')}") # Display time in 12-hour format

    # Preprocess inputs for prediction
    if st.button('Predict Fare'):
        # Calculate distance
        distance = haversine_distance(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude)

        # Determine is_weekend
        is_weekend = 1 if day_of_week >= 5 else 0

        # Determine type_of_vehicle for one-hot encoding
        uber_go = 0
        uber_xl = 0
        if passenger_count > 4:
            uber_xl = 1
        else:
            uber_go = 1

        # Create a DataFrame for the new data (in the same order as x_train)
        new_data = pd.DataFrame({
            'distance': [distance],
            'is_weekend': [is_weekend],
            'Day_of_week': [day_of_week],
            'Hour': [hour],
            'Month': [month],
            'Year': [year],
            'Uber Go': [uber_go],
            'Uber XL': [uber_xl]
        })

        # Scale the new data using the loaded x_scaler
        new_data_scaled = x_scaler.transform(new_data.values)

        # Make prediction
        prediction_scaled = model.predict(new_data_scaled)

        # Inverse transform the prediction to get actual fare amount
        predicted_fare = y_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))

        st.success(f"Predicted Fare Amount: ${predicted_fare[0][0]:.2f}")

st.markdown("### Note on Inputs:")
st.markdown("- **Map Selection**: Click on the map to set pickup and dropoff locations alternatively. The current selection mode is displayed.")
st.markdown("- **Day of Week**: 0=Monday, 6=Sunday.")
st.markdown("- **Passenger Count**: Determines 'Uber Go' or 'Uber XL' (XL for >4 passengers).")
