# import libraries
import pandas as pd
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    """ 
    Load Messages and Category CSV Files. Files are merged based upon a common ID key.
  
    Parameters: 
    messages_filepath (string): Relative or Full Path to the Messages CSV File
    categories_filepath (string): Relative or Full Path to the Categories CSV File
  
    Returns: 
    df: Dataframe of combined Messages and Categories data set.
  
    """
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, on='id')
    
    return df


def clean_data(df):
    """ 
    Parses the category columns to update headers with proper names, and updates 
    from the original format. Finally, it drops duplicates in the frame.
  
    Parameters: 
    df (df): Combined Messages & Categories Dataframe
  
    Returns: 
    df: Dataframe of cleaned data, with proper header names and data types.
  
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    # use this row to extract a list of new column names for categories.
    row = categories.iloc[[1]]
    
    category_columns = []

    for column_name in row.values[0]:
        category_columns.append(column_name.split('-')[0])
    
    # rename the columns
    categories.columns = category_columns

    for column in categories:
        # set each value to be the last character 
        categories[column] = categories[column].astype(str).str[-1:]

        # convert column from string to integer
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], join='inner', axis=1)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    return df
        
def save_data(df, database_filename):
    """ 
    Save dataframe to a SQLite database.
  
    Parameters: 
    df (dataframe): Full dataframe to write to a SQL Database
    database_filename (string): Relative or Full path to Database file location. 
  
    Returns: 
    Nothing
  
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('categorized_messages_tbl', engine, index=False, if_exists='replace')  


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