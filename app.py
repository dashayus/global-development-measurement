# app.py

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.io as pio
import os


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', message='No selected file')

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Read the uploaded CSV file
        df = pd.read_csv(file_path)
        # Renaming columns
        df.rename({'Birth Rate':'Birth_Rate',
           'Business Tax Rate':'Business_Tax_Rate',
           'CO2 Emissions':'CO2_Emissions',
           'Days to Start Business':'Days_to_Start_Business',
           'Ease of Business':'Ease_of_Business',
           'Energy Usage':'Energy_Usage',
           'Health Exp % GDP':'Health_Exp_GDP',
           'Health Exp/Capita':'Health_Exp_per_Capita',
           'Hours to do Tax':'Hours_to_do_Tax',
           'Infant Mortality Rate':'Infant_Mortality_Rate',
           'Internet Usage':'Internet_Usage',
           'Lending Interest':'Lending_Interest',
           'Life Expectancy Female':'Life_Expectancy_Female',
           'Life Expectancy Male':'Life_Expectancy_Male',
           'Mobile Phone Usage':'Mobile_Phone_Usage',
           'Number of Records':'Number_of_Records',
           'Population 0-14':'Population_0_to_14',
           'Population 15-64':'Population_15_to_64',
           'Population 65+':'Population_above_65',
           'Population Total':'Population_Total',
           'Population Urban':'Population_Urban',
           'Tourism Inbound':'Tourism_Inbound',
           'Tourism Outbound':'Tourism_Outbound'}, axis=1, inplace=True)
        # Remove '%' and '$' symbols from specific columns
        df['Business_Tax_Rate'] = df['Business_Tax_Rate'].str.rstrip('%').astype('float') / 100
        df['GDP'] = df['GDP'].replace('[\$,]', '', regex=True).astype('float')
        df['Health_Exp_per_Capita'] = df['Health_Exp_per_Capita'].replace('[\$,]', '', regex=True).astype('float')
        df['Tourism_Inbound'] = df['Tourism_Inbound'].replace('[\$,]', '', regex=True).astype('float')
        df['Tourism_Outbound'] = df['Tourism_Outbound'].replace('[\$,]', '', regex=True).astype('float')
        # Drop columns not required
        df = df.drop(['Ease_of_Business', 'Number_of_Records'], axis=1)
        # Transfrom Country column using label encoder
        label_encoder = LabelEncoder()
        df['Country']= label_encoder.fit_transform(df['Country'])
        # Handling missing values using KNN Imputer
        imputer = KNNImputer(n_neighbors=7)
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        # Standardize the data 
        scaler = StandardScaler()
        scaled_df = scaler.fit_transform(df)
        # We will perform PCA on first 13 components
        pca1 = PCA(n_components= 13)
        pca_scaled = pca1.fit_transform(scaled_df)
        # Creating a dataframe of the components
        pca_scaled_df = pd.DataFrame(data = pca_scaled, columns = ['PC1', 'PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13'])
        # Assuming 'X_train' is your training data
        clusters_new = KMeans(3, random_state=42)
        predict_Kmeans_ag=clusters_new.fit_predict(pca_scaled_df)
        
        # # # Dataframe with labels
        pca_scaled_df['KMeans_Labels'] = predict_Kmeans_ag

        # Datapoints present in each cluster
        Predictions=pca_scaled_df['KMeans_Labels'].value_counts()

        
        
        silhouette_score_average_K_mean = silhouette_score(pca_scaled_df, predict_Kmeans_ag)

       
       
       # Generate a scatter plot using Plotly
        fig = px.scatter(pca_scaled_df, x='PC1', y='PC2',  color=predict_Kmeans_ag ,labels={'PC1': 'X', 'PC2': 'Y'}, title='Cluster Plot')

        # Convert the plot to HTML
        plot_html = pio.to_html(fig, full_html=False)

        

        # Display the contents of the CSV file
        return render_template('result.html',silhouette_score=silhouette_score_average_K_mean,Predictions=Predictions,plot_html=plot_html)



if __name__ == '__main__':
    app.run(debug=True)
