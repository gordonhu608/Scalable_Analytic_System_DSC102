from dask.distributed import Client
import numpy as np
import dask.dataframe as dd

import json

def PA0(user_reviews_csv):
    client = Client()
    client = client.restart()
    
#     dtypes = {
#     'reviewerID': str,
#     'asin': str, 
#     'reviewerName': str,
#     'helpful': object,
#     'reviewText': str,
#     'overall': float64,
#     'summary': str,
#     'unixReviewTime': float,
#     'reviewTime': str}
    
    df = dd.read_csv(user_reviews_csv)

    df['helpful_votes'] = df['helpful'].apply(lambda x: int(str(x[1:-1]).split(', ')[0]), meta=('helpful', 'object'))
    df['total_votes'] = df['helpful'].apply(lambda x: int(str(x[1:-1]).split(', ')[1]), meta=('helpful', 'object'))
    
    df['reviewTime'] = df['reviewTime'].str[-4:].astype(int)
    
    df = df.drop(['unixReviewTime', 'asin', 'helpful', 'reviewText', 'summary', 'reviewerName'], axis=1)
    
    res = df.groupby("reviewerID").agg({
            "reviewerID": 'count',
            "overall": 'mean',
            'reviewTime': 'min',
            'helpful_votes': 'sum',
            'total_votes': 'sum'
            }, split_out=4)
#     res = res.reset_index()
#     res.columns = ['number_products_rated', 'avg_ratings', 'reviewing_since', 'helpful_votes', 'total_votes']

    res.columns = ['number_products_rated', 'avg_ratings', 'reviewing_since', 'helpful_votes', 'total_votes']

    res = res.astype({
        'number_products_rated': int,
        'avg_ratings': float,
        'reviewing_since': int,
        'helpful_votes': int,
        'total_votes': int
    })

    res = res.reset_index()
    res.columns = ['reviewerID', 'number_products_rated', 'avg_ratings', 'reviewing_since', 'helpful_votes', 'total_votes']

    # Change <YOUR_USERS_DATAFRAME> to the dataframe variable in which you have the final users dataframe result
    submit = res.describe().compute().round(2)    
    with open('results_PA0.json', 'w') as outfile: 
        json.dump(json.loads(submit.to_json()), outfile)
