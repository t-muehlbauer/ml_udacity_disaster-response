# Disaster Response Pipeline Project
# UDACITY Nanodegree - Data Scientist

## Description
### The Project
In the second project of this nanodegree, we will develop a model to classify messages sent during disasters. There are 36 predefined categories such as 'Child alone', 'Aid related', or 'Medical help'. This classification is crucial for directing these messages to the appropriate disaster relief agencies. The project entails data preparation and the construction of the actual model. The data is sourced from Appen (formerly Figure 8) and comprises authentic messages.

The end result will be a web application that takes a message as input and provides the corresponding classification as output:
![alt text](udacity_webapp_dr.png.png "Screenshot of the Apllication")


### The Files
- app: This folder contains the templates and run.py for the web application.
- data: This folder contains the database, the csv's and process_data.py for the data preparation and transfer.
- model: This folder contains the actual machine learning model and train_classifier.py for training the model.

Notice: The .ipynb files are just for preparing the data and are not necessary to run this project.

## Getting Started:
1. Clone the project. It has to run with Python 3 with the following libraries: numpy, pandas, sqlalchemy, re, NLTK, pickle, Sklearn, plotly and flask.

2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Go to `app` directory: `cd app`

4. Run your web app: `python run.py`

5. Click the `PREVIEW` button in the IDE or open the to open http://0.0.0.0:3000/

## Licensing, Authors, Acknowledgements
Thanks to Appen (formally Figure 8) for providing the data. Feel free to use the content while citing me,Udacity and/or Appen.