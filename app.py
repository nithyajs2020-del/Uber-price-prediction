import streamlit as st
import pandas as pd
import numpy as np
import pickle

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
st.title('Uber Fare Prediction App')

st.write("Enter the details of your trip to predict the fare amount.")

# User Inputs
pickup_latitude = st.number_input('Pickup Latitude', value=40.738354, format="%.6f")
pickup_longitude = st.number_input('Pickup Longitude', value=-73.999817, format="%.6f")
dropoff_latitude = st.number_input('Dropoff Latitude', value=40.723217, format="%.6f")
dropoff_longitude = st.number_input('Dropoff Longitude', value=-73.999512, format="%.6f")
passenger_count = st.slider('Passenger Count', 1, 6, 1)
year = st.selectbox('Year', list(range(2009, 2016)), index=5)
month = st.selectbox('Month', list(range(1, 13)), index=4)
day = st.selectbox('Day', list(range(1, 32)), index=6)
hour = st.selectbox('Hour', list(range(0, 24)), index=19)
day_of_week = st.selectbox('Day of Week', list(range(0, 7)), format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x], index=3)

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
st.markdown("- **Pickup/Dropoff Latitude/Longitude**: Decimal degrees.")
st.markdown("- **Day of Week**: 0=Monday, 6=Sunday.")
st.markdown("- **Passenger Count**: Determines 'Uber Go' or 'Uber XL' (XL for >4 passengers).")
