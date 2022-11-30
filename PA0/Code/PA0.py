from dask.distributed import Client
import numpy as np
import dask.dataframe as dd

def PA0(user_reviews_csv):
    client = Client()
    client = client.restart()

    dtypes = {
    'reviewerID': np.str,
    'asin': np.str, 
    'reviewerName': np.str,
    'helpful': np.object,
    'reviewText': np.str,
    'overall': np.float64,
    'summary': np.str,
    'unixReviewTime': np.float64,
    'reviewTime': np.str
    }

    #preprocess 
    df = dd.read_csv('~/data/user_reviews_Release.csv', dtype=dtypes, converters={
        'helpful': lambda x: eval(x)
        })
    #df = df.set_index("reviewerID")
    df['helpful_vote'] = df['helpful'].apply(lambda x: x[0], meta=('helpful', 'object'))
    df['total_vote'] = df['helpful'].apply(lambda x: x[1], meta=('helpful', 'object'))
    df['unixReviewTime'] = df['unixReviewTime'] //31536000 + 1970

    count_num = dd.Aggregation(
        'count_num',
        lambda s: s.count(),
        lambda count: count.sum()
        )

    res = df.groupby("reviewerID").agg({
        "asin": 'count',
        "overall": 'mean',
        'reviewTime': 'min',
        'helpful_vote': 'sum',
        'total_vote': 'sum'
        }, split_out=4)
    res = res.reset_index()
    res.columns = ['reviewerID','number_products_rated', 'avg_ratings', 'reviewing_since', 'helpful_votes', 'total_votes']

    # Change <YOUR_USERS_DATAFRAME> to the dataframe variable in which you have the final users dataframe result
    submit = res.describe().compute().round(2)    
    with open('results_PA0.json', 'w') as outfile: 
        json.dump(json.loads(submit.to_json()), outfile)
