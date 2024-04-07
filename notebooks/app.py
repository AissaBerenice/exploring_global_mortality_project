import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

#IMPORTED DATA
df = pd.read_csv("C:/Users/Aissa/apps/iteso/project-deathcausesprediction/data/raw/deaths_data.csv")

#APP
st.title('Exploring Global Mortality: Visualization and Forecasting of Death Causes by Country')
st.header("Data Analytics Project")

#DROPDOWN MENU Country
selected_country = st.selectbox('Select a country:', df['Entity'].unique())

#DROPDOWN MENU Filtered Country
filtered_country = df[df['Entity'] == selected_country]

#DROPDOWN MENU Death reason
filtered_columns = [col for col in df.columns if col not in ['Entity', 'Code','Year']]
selected_column = st.selectbox('Select a column:', filtered_columns)

#NEW Dataframe filtered country and filtered_columnn
df_new = filtered_country[['Year', selected_column]]

#FEATURES AND TARGET
X = df_new[["Year"]]
y = df_new[selected_column]

# LINEAR REGRESSION NODEL
model = LinearRegression()

#TRAINED MODEL
model.fit(X, y)

#PREDICTION of next 10 years
future_years = np.arange(2020, 2030).reshape(-1, 1)
predicted_values = model.predict(future_years)

#JOINED PREDICTIONS DF
predictions_df = pd.DataFrame({
    "Year": future_years.flatten(),
    "Prediction": predicted_values
})

#JOINED DF
join = [df_new,predictions_df]
joined_df = pd.concat(join)

#PLOTTING WITH STREAMLIT
st.title(f"Linear Regression Predictions for {selected_column}")
st.write("Actual Data (1990-2019) vs. Predicted Values (2020-2029)")

st.line_chart(joined_df, x="Year", y=[selected_column, 'Prediction'], color=["#ffaa0088",'#0000FF'])