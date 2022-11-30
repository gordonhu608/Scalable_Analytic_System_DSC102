from cProfile import run
from dask.distributed import Client, LocalCluster
import time
import json
import ast
import dask.dataframe as dd
import numpy as np


def PA1(user_reviews_csv,products_csv):
    start = time.time()
    client = Client('172.31.0.124:8786') # 172.31.0.124:8786    127.0.0.1:8786
    client = client.restart()
    print(client)
        
    reviews = dd.read_csv(user_reviews_csv)
    products = dd.read_csv(products_csv)

    q1_reviews = round((100*(reviews.isna().mean())).compute(), 2)
    q1_products = round((100*(products.isna().mean())).compute(), 2)
    
    combined = dd.merge(reviews[['asin', 'overall']], products[['asin', 'price']], on='asin').drop('asin', axis=1)
    q2 = round(combined.price.corr(combined.overall, method = 'pearson').compute(), 2)
    q3 = products.price.describe().compute()[['mean', 'std', '50%', 'min', 'max']].apply(lambda x: round(x, 2))
    q4 = products.categories.dropna().apply(lambda x: eval(x)[0][0], meta=('categories', 'object')).value_counts().compute().apply(lambda x: round(x, 2))
    
    set_of_prod = set(products.asin.values.compute())
    set_of_reviews = set(reviews.asin.values.compute())
    q5 = 1*(len(set_of_reviews - set_of_prod) > 0)
    
    def decide():
        for i in products['related'].dropna():
            y = sum(ast.literal_eval(i).values(),[])
            for j in y:
                if j not in set_of_prod:
                    return 1
        return 0

    q6 = decide()
    end = time.time()
    runtime = end-start
    
    print(runtime)

    # Write your results to "results_PA1.json" here
    with open('OutputSchema_PA1.json','r') as json_file:
        data = json.load(json_file)
        print(data)

        data['q1']['products'] = json.loads(q1_products.to_json())
        data['q1']['reviews'] = json.loads(q1_reviews.to_json())
        data['q2'] = q2
        data['q3'] = json.loads(q3.to_json())
        data['q4'] = json.loads(q4.to_json())
        data['q5'] = q5
        data['q6'] = q6
    
    # print(data)
    with open('results_PA1.json', 'w') as outfile: json.dump(data, outfile)


    return runtime