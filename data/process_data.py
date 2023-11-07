# import libraries
import pandas as pd
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    """
    Load and merge messages and categories datasets.

    Args:
        messages_filepath (str): The file path of the messages dataset.
        categories_filepath (str): The file path of the categories dataset.

    Returns:
        pandas.DataFrame: A merged DataFrame containing messages and corresponding categories.
    """
    # Load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # Load categories dataset
    categories = pd.read_csv(categories_filepath) 
    
    # Merge datasets on 'id' column
    df = messages.merge(categories, left_on='id', right_on='id', how='inner')
    
    return df

    


def clean_data(df):
    """
    Clean and preprocess the input DataFrame containing message categories.

    Args:
        df (pandas.DataFrame): Input DataFrame containing a column 'categories' with semicolon-separated category strings.

    Returns:
        pandas.DataFrame: Cleaned DataFrame with individual category columns and numeric values.
    """
    # Create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(pat=";", n=-1, expand=True)
    
    # Select the first row of the categories dataframe
    row = categories.iloc[0]

    # Use this row to extract a list of new column names for categories.
    # One way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])

    # Rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: int(x[-1]))
        # Convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # replace the "wrong" values from '2' to '1'
    categories.replace(2, 1, inplace=True)
       
    # Drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)
    
    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    return df

        
        

def save_data(df, database_filename):
    """
    Save a DataFrame to an SQLite database.

    Args:
        df (pandas.DataFrame): The DataFrame to be saved.
        database_filename (str): The desired filename for the SQLite database.

    Returns:
        None
    """
    engine = create_engine('sqlite:///'+database_filename)
    table_name = database_filename.replace(".db","") + "_Table"
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')



def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()