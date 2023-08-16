# Improve your ratings on Airbnb

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Run Instructions](#instructions)
5. [Licensing, Authors, and Acknowledgements](#licensing)


## Installation <a name="installation"></a>

Beyond the Anaconda distribution of Python to run the code here its is necessary to install the library `plotly`.

The code should run with no issues using Python version 3.11.2 and plotly version 5.16.0.

## Project Motivation<a name="motivation"></a>

In this project I performed a statistical analysis to determine which are the more important features of a listing that are associated to high review scores. Then I created a web application that, according to the results of the analysis, allows potential hosts to test if a new post could be catchy after enlisting and to predict the expected review ratings for their accommodation. The application uses a regression model trained on the Airbnb activity record in the Seattle area over a year span.

The tasks involved are the following:
1.	Download and preprocess the Seattle Airbnb open data.
2.	Determine the most influential features in the dataset to predict the review rating scores by studying the correlation coefficients of a linear model.
3.	Train a regressor that can predict the expected rating scores for a new listing.
4.	Develop a front-end component to collect users’ inputs and display predictions based on the trained regressor.
5.	Make the application run on a browser.

Further details are available in the `Report` file.

## File Descriptions <a name="files"></a>

In the main directory of the project you can find the following folders:
- data
- EDA
- models
- app
- notebooks
- CRISP-DM process

1. In the `data` folder you can find:
    - two .csv files with the original data from Kaggle
    - a python script that cleans data and stores in database
    - the AirbnbRatings database with the cleaned data
    
2. In the folder `EDA` you can find  3 notebooks available here to showcase work related to the questions:
    - How can you improve your ratings as AirBnB host?
    - Do ratings depend on the location of the listings?
    - How much ratings depend on prices and availability?
    - Which features of each listing affects the most its ratings?
    - Are ratings realated to hosts' behaviours and experience?
    - Are ratings related to booking policies?

    Each of the notebooks is exploratory in searching through the data pertaining to the questions showcased by the notebook title. 
    Markdown cells were used to assist in walking through the thought process for individual steps. 

3. In the `models` folder you can find:
    - a python script that trains a regression model and saves its parametrers in the `regressor.plk` files
    - three .plk files storing the parameters for 3 trained models

4. In the `app` folder you can find a python script and two html templates that are necessary to run the app and contain all instuctions for visualization and routing

5. In the `notebooks` folder you can find two notebooks performing an ETL and a ML pipeline that clean the data, store in database, train the regressor and save the model. 
    The code of these notebooks correspond to the code of the `process_data.py` and `train_regressor.py` scripts. 
    The trained models are stored in the folder `models` as `airbnb_pipeline.pkl` and `airbnb_gridsearch.pkl`.
    Markdown cells were used to assist in walking through the thought process for individual steps. 

## Run Instructions <a name="instructions"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/listings.csv data/AirbnbRatings.db`
    - To run ML pipeline that trains regressor and saves
        `python models/train_regressor.py data/AirbnbRatings.db models/regressor.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:3000/

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Airbnb for the data. You can find the Licensing for the data and other descriptive information at the Kaggle link available [here](https://www.kaggle.com/datasets/airbnb/seattle). Otherwise, feel free to use the code here as you would like!


