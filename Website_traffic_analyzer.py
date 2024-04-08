import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Function to load and preprocess data
def load_data(file):
    data = pd.read_csv(file)
    # Add preprocessing steps if needed
    return data

# Function to train model and make predictions
def train_and_predict(data):
    X = data[['page_loads', 'first_visits', 'returning_visits', 'days_f']]
    y = data['unique_visits']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return y_test, predictions

# Main function
def main():
    st.title("CSV Upload and Model Prediction")

    # File uploader
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write("Data uploaded successfully!")
        
        # Perform model predictions
        y_test, predictions = train_and_predict(data)
        
        # Display predictions
        st.write("Actual vs. Predicted values:")
        df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
        st.write(df)
        line_chart = px.line(data, x='Day', y=['page_loads', 'unique_visits', 'first_visits', 'returning_visits'],
                     labels={'value': 'Visits'}, title='Page Loads & Visitors over Time')
        st.plotly_chart(line_chart)

        # Plot histogram
        histogram = px.histogram(data, x='unique_visits', color='Day', title='Unique Visits for Each Day')
        st.plotly_chart(histogram)

        # Plot density heatmap
        density_heatmap = px.density_heatmap(data, x='Day', y=['page_loads', 'unique_visits', 'first_visits', 'returning_visits'],
                                            marginal_x="histogram", marginal_y="histogram",
                                            title='Correlation for Each Data Point')
        st.plotly_chart(density_heatmap)

        # Plot actual vs. predicted values
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df)
        st.pyplot(plt)

if __name__ == "__main__":
    main()
