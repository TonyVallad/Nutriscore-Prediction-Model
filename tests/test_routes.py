import pytest
from flask import json
from app import create_app

@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# Test the /loading-dataframe-status route
def test_loading_dataframe_status(client):
    response = client.get('/loading-dataframe-status')
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, dict)  # Expecting a dictionary response
    assert 'loading_dataframe_status' in data  # Expected key in response

# Test the / (home) route
def test_home_route(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'<html' in response.data  # Check if response contains HTML content

# Test the /predict route with valid data
def test_predict_route_valid_data(client):
    response = client.post('/predict', json={
        'feature1': 2.0,  # Replace with actual features required for prediction
        'feature2': 1.5,
        # Add additional features as needed
    })
    assert response.status_code == 200
    data = response.get_json()
    assert 'score' in data  # Assuming a 'score' field in the JSON response

# Test the /predict route with missing or invalid data
def test_predict_route_invalid_data(client):
    response = client.post('/predict', json={
        'feature1': 'invalid',  # Sending a string instead of a number
        # Missing required features
    })
    assert response.status_code == 400  # Expecting a 400 Bad Request
    data = response.get_json()
    assert 'error' in data  # Assuming an error message is returned
