import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

def load_data(listings_filepath):
    
    # load dataset
    df = pd.read_csv(listings_filepath)
    
    return df

def remove_missing_bycol(df, column_names):
    '''Takes in a Data Frame (df) and a list of strings (column_names).
Returns a Data Frame without records with missing values in the inputed columns'''
    
    df_output = df.dropna(subset = column_names)
    
    return df_output

def check_char_in_df(df, chars_to_check):
    '''Takes in a Data Frame (df) and a sting (chars_to_check).
Returns a list of strings with the names of the columns in the input df which have almost one record containing the specified characters.'''
    
    
    output_list = []
    for col in df.columns:
        try:
            check_values = df[col].str.contains(chars_to_check).any()
            if check_values:
                output_list.append(col)
        except:
            1
        
    return output_list

def check_value_for_col_in_df(df, chars_to_check):
    '''Takes in a Data Frame (df) and a sting (chars_to_check).
Returns a list of strings with the names of the columns in the input df which have almost one record equal to the specified characters.'''
        
    output_list = []
    for col in df.columns:
        try:
            check_values = df[col].eq(chars_to_check).any()
            if check_values:
                output_list.append(col)
        except:
            1
        
    return output_list

def replace_char_and_cast_float(df, cols, chars_to_replace, replacement):
    '''Takes in a Data Frame (df), 2 sting (cols and replacement) and a list of strings (chars_to_replace).
Returns a data frame where the input columns are casted to float after replacing the specified characters with the replacement string'''
    
    for col in cols:
        df[col] = df[col].replace(regex=chars_to_replace, value=replacement).astype('float64')
        
    return df

def replace_cols_in_df(df, cols_to_drop, cols_to_concat):
    '''Takes in 2 Data Frame (df, cols_to_concat) and a list of stings (cols_to_drop).
Returns a data frame where the fields in cols_to_drop are replaced by fields in cols_to_concat'''
    
    df_output = pd.concat([df.drop(cols_to_drop, axis = 1), cols_to_concat], axis = 1)
        
    return df_output

def clean_data_for_feature_selection(df, cols_of_interest):
    
    # remove records with average rating missing
    df_ratedListings = remove_missing_bycol(df, ['review_scores_rating'])
    
    # define subset dataframes with columns of interest for feature selection
    df_regression = df_ratedListings[cols_of_interest]
    
    # cast columns to numeric whe possible
    
    ## money columns
    money_columns = check_char_in_df(df_regression, '\$')
    df_regression = replace_char_and_cast_float(df_regression, money_columns, [r'\$', r','], '')
    
    ## date columns
    try:
        date_columns = ['host_since', 'first_review', 'last_review']
        for col in date_columns:
            df_regression[col] = - ((pd.to_datetime(df_regression[col]) - pd.to_datetime(max_date)).dt.days)
    except:
        pass
    
    ## % columns
    percent_columns = check_char_in_df(df_regression, '\%')
    df_regression = replace_char_and_cast_float(df_regression, percent_columns, [r'\%', r','], '')

    ## boolean columns
    boolean_columns = check_value_for_col_in_df(df_regression, 't')
    Dummy_Boolean_Columns = pd.get_dummies(df_regression[boolean_columns], prefix=boolean_columns, drop_first=True)
    df_regression = replace_cols_in_df(df_regression, boolean_columns, Dummy_Boolean_Columns)
   
    # encode categorical columns
    Categorical_Columns = df_regression.select_dtypes(include=['object'])
    Dummy_Encoded_Columns = pd.get_dummies(Categorical_Columns, prefix=Categorical_Columns.columns)
    df_regression = replace_cols_in_df(df_regression, Categorical_Columns.columns, Dummy_Encoded_Columns)
    
    # remove binary columns with almost same value for all records
    isfalse = df_regression.eq(0).mean()
    istrue = df_regression.eq(1).mean()
    almostFalse = isfalse[isfalse > 0.9].to_frame().index.values
    almostTrue = istrue[istrue > 0.9].to_frame().index.values
    df_regression = df_regression.drop(almostFalse, axis=1).drop(almostTrue, axis=1)
    
    # impute null values
    nulls_in_cols = df_regression.isnull().mean()
    cols_with_nulls = nulls_in_cols[nulls_in_cols > 0]
    df_regression[cols_with_nulls.index] = df_regression[cols_with_nulls.index].fillna(0)
    
    # remove collinear columns
    collinear_columns = {}
    for f in correlation_matrix.columns:
        collinear_columns[f] = [i for i in correlation_matrix.index if np.abs(correlation_matrix.at[f,i])>.5 and f>i]
    collinear_columns_list = []
    for key in collinear_columns:
        collinear_columns_list.extend(collinear_columns[key])
    df_regression = df_regression.drop(collinear_columns_list, axis = 1)
    
    return df_regression

def select_features(df):
    
    # instance and fit the model for feature selection
    X = df.drop(columns = 'review_scores_rating')
    y = df['review_scores_rating']
    lm_model = LinearRegression(normalize=True)
    lm_model.fit(X, y)
    
    # extract linear coefficients
    coefs_df = pd.DataFrame()
    coefs_df['fields'] = X.columns
    coefs_df['coefs'] = lm_model.coef_
    coefs_df['abs_coefs'] = np.abs(lm_model.coef_)
    
    # select features
    selected_features = coefs_df['fields'][coefs_df['abs_coefs'] > 0.1]
    
    return selected_features

def prepare_data_for_prediction_model(df, text_feature, numeric_features):
    
    df_ratedListings = remove_missing_bycol(df, ['review_scores_rating'])
    
    features = numeric_features
    features.append(text_feature)
    features.append('review_scores_rating')
    
    df_prepared = df_ratedListings[selected_features]
    
    df_prepared[text_feature] = df_prepared[text_feature].fillna('')
    
    return df_prepared

def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('listings', engine, if_exists='replace', index=False)

def main():
    if len(sys.argv) == 4:

        listings_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    LISTINGS: {}'
              .format(listings_filepath))
        df = load_data(listings_filepath)

        print('Cleaning data for feature selection...')
        cols_of_interest = [
            'host_since', 
            'host_response_time',
            'host_response_rate', 
            'host_acceptance_rate', 
            'host_is_superhost',
            'host_listings_count', 
            'host_has_profile_pic', 
            'host_identity_verified',
            'is_location_exact', 
            'property_type', 
            'room_type', 
            'accommodates',
            'bathrooms', 
            'bedrooms', 
            'beds', 
            'bed_type', 
            'price', 
            'security_deposit',
            'cleaning_fee', 
            'guests_included', 
            'extra_people', 
            'minimum_nights', 
            'maximum_nights', 
            'availability_30', 
            'availability_60', 
            'availability_90', 
            'availability_365', 
            'number_of_reviews',
            'first_review', 
            'last_review', 
            'review_scores_rating', 
            'instant_bookable', 
            'cancellation_policy', 
            'require_guest_profile_picture',
            'require_guest_phone_verification', 
            'reviews_per_month'
        ]
        df_selection = clean_data_for_feature_selection(df, cols_of_interest)
        
        print('Selecting fetures...')
        selected_features = select_features(df_selection)
        
        print('Preparing data for prediction model...')
        text_feature = 'description'
        df_prepared = prepare_data_for_prediction_model(df, text_feature, selected_features)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df_prepared, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the listings '\
              'datasets as the first argument, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the second argument. \n\nExample: python process_data.py '\
              'listings.csv AirbnbRatings.db')


if __name__ == '__main__':
    main()