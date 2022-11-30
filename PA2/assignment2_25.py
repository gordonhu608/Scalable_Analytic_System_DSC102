import os
import pyspark.sql.functions as F
import pyspark.sql.types as T
from utilities import SEED
# import any other dependencies you want, but make sure only to use the ones
# availiable on AWS EMR

# ---------------- choose input format, dataframe or rdd ----------------------
INPUT_FORMAT = 'dataframe'  # change to 'rdd' if you wish to use rdd inputs
# -----------------------------------------------------------------------------
if INPUT_FORMAT == 'dataframe':
    import pyspark.ml as M
    import pyspark.sql.functions as F
    import pyspark.sql.types as T
    from pyspark.ml.regression import DecisionTreeRegressor
    from pyspark.ml.evaluation import RegressionEvaluator
if INPUT_FORMAT == 'koalas':
    import databricks.koalas as ks
elif INPUT_FORMAT == 'rdd':
    import pyspark.mllib as M
    from pyspark.mllib.feature import Word2Vec
    from pyspark.mllib.linalg import Vectors
    from pyspark.mllib.linalg.distributed import RowMatrix
    from pyspark.mllib.tree import DecisionTree
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.linalg import DenseVector
    from pyspark.mllib.evaluation import RegressionMetrics


# ---------- Begin definition of helper functions, if you need any ------------

import pyspark.ml.feature as K

# def task_1_helper():
#   pass

# -----------------------------------------------------------------------------


def task_1(data_io, review_data, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    overall_column = 'overall'
    # Outputs:
    mean_rating_column = 'meanRating'
    count_rating_column = 'countRating'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    grouped = review_data.groupBy('asin').agg(F.avg('overall'), F.count('overall')).toDF('asin', 'meanRating', 'countRating')
    transformed = product_data[['asin']].join(grouped, on='asin', how="left")
    output = transformed.select(F.mean('meanRating'), 
                             F.variance('meanRating'),
                             F.count(F.when(F.col('meanRating').isNull(),True)),
                             F.mean('countRating'),
                             F.variance('countRating'),
                             F.count(F.when(F.col('countRating').isNull(),True))).collect()[0]

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    # Calculate the values programmaticly. Do not change the keys and do not
    # hard-code values in the dict. Your submission will be evaluated with
    # different inputs.
    # Modify the values of the following dictionary accordingly.
    res = {
        'count_total': None,
        'mean_meanRating': None,
        'variance_meanRating': None,
        'numNulls_meanRating': None,
        'mean_countRating': None,
        'variance_countRating': None,
        'numNulls_countRating': None
    }
    # Modify res:
    res['count_total'] = transformed.count()
    res['mean_meanRating'] = output[0]
    res['variance_meanRating'] = output[1]
    res['numNulls_meanRating'] = output[2]
    res['mean_countRating'] = output[3]
    res['variance_countRating'] = output[4]
    res['numNulls_countRating'] = output[5]
    
    
    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_1')
    return res
    # -------------------------------------------------------------------------



def task_2(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    salesRank_column = 'salesRank'
    categories_column = 'categories'
    asin_column = 'asin'
    # Outputs:
    category_column = 'category'
    bestSalesCategory_column = 'bestSalesCategory'
    bestSalesRank_column = 'bestSalesRank'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    def flatten_cate(x):
        if type(x) != list or len(x) == 0 or x[0][0] == "":
            return None
        return x[0][0]
    
    flatten_cate = F.udf(flatten_cate, T.StringType())
    
    cates = product_data.select(F.col(asin_column), flatten_cate(F.col(categories_column)).alias(category_column))
    ranks = product_data.select(asin_column, F.explode(salesRank_column)).toDF(asin_column, bestSalesCategory_column, bestSalesRank_column)
    transformed = cates.join(ranks, on = asin_column, how = 'left')
    output = transformed.select(F.mean(bestSalesRank_column), 
                             F.variance(bestSalesRank_column),
                             F.count(F.when(F.col(category_column).isNull(),True)),
                             F.countDistinct(category_column),
                             F.count(F.when(F.col(bestSalesCategory_column).isNull(),True)),
                             F.countDistinct(bestSalesCategory_column)).collect()[0]
    
    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_bestSalesRank': None,
        'variance_bestSalesRank': None,
        'numNulls_category': None,
        'countDistinct_category': None,
        'numNulls_bestSalesCategory': None,
        'countDistinct_bestSalesCategory': None
    }
    # Modify res:
    res['count_total'] = transformed.count()
    res['mean_bestSalesRank'] =  output[0]
    res['variance_bestSalesRank'] = output[1]
    res['numNulls_category'] = output[2]
    res['countDistinct_category'] = output[3]
    res['numNulls_bestSalesCategory'] = output[4]
    res['countDistinct_bestSalesCategory'] = output[5]
    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_2')
    return res
    # -------------------------------------------------------------------------



def task_3(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    price_column = 'price'
    attribute = 'also_viewed'
    related_column = 'related'
    # Outputs:
    meanPriceAlsoViewed_column = 'meanPriceAlsoViewed'
    countAlsoViewed_column = 'countAlsoViewed'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    
    sub_data = product_data[['asin', 'price']]
    
    # count viewed
    def get_length(x):
        if type(x) != dict or 'also_viewed' not in x:
            return None
        return len(x['also_viewed'])
    
    get_len = F.udf(get_length, T.IntegerType())
    df = product_data.select(F.col('asin'), get_len(F.col("related")).alias("countAlsoViewed"))
    
    mapping_data = product_data[['asin', 'price']]
    sub_data = product_data[['asin', 'related']]
    def get_lst(x):
        if type(x) != dict or 'also_viewed' not in x or x['also_viewed'] == []:
            return None
        return x['also_viewed']
    get_lst = F.udf(get_lst, T.ArrayType(T.StringType()))
    converted_df = sub_data.select(F.col('asin'), get_lst(F.col("related")).alias("extracted"))
    converted_df = converted_df.toDF('asin', 'extracted')#.show()
    exploded_df = converted_df.select(F.col('asin').alias('benchmark'), F.explode(F.col("extracted")).alias('asin'))
    merged = exploded_df.join(mapping_data, on='asin', how="left") # .groupBy('benchmark')
    transformed = merged[['benchmark', 'price']].groupBy('benchmark').mean().toDF('asin', 'meanPriceAlsoViewed')
    transformed = df.join(transformed, on='asin', how='left')
    
    output = transformed.select(F.mean(meanPriceAlsoViewed_column), 
                             F.variance(meanPriceAlsoViewed_column),
                             F.count(F.when(F.col(meanPriceAlsoViewed_column).isNull(),True)),
                             F.mean(countAlsoViewed_column), 
                             F.variance(countAlsoViewed_column),
                             F.count(F.when(F.col(countAlsoViewed_column).isNull(),True))).collect()[0]



    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_meanPriceAlsoViewed': None,
        'variance_meanPriceAlsoViewed': None,
        'numNulls_meanPriceAlsoViewed': None,
        'mean_countAlsoViewed': None,
        'variance_countAlsoViewed': None,
        'numNulls_countAlsoViewed': None
    }
    # Modify res:
    res['count_total'] = transformed.count()
    res['mean_meanPriceAlsoViewed'] = output[0]
    res['variance_meanPriceAlsoViewed'] = output[1]
    res['numNulls_meanPriceAlsoViewed'] = output[2]
    res['mean_countAlsoViewed'] = output[3]
    res['variance_countAlsoViewed'] = output[4]
    res['numNulls_countAlsoViewed'] = output[5]



    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_3')
    return res
    # -------------------------------------------------------------------------
    

def task_4(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    price_column = 'price'
    title_column = 'title'
    # Outputs:
    meanImputedPrice_column = 'meanImputedPrice'
    medianImputedPrice_column = 'medianImputedPrice'
    unknownImputedTitle_column = 'unknownImputedTitle'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------

    imputer = K.Imputer()
    imputer.setInputCols(["price"])
    imputer.setOutputCols(["medianImputedPrice"])
    fit_median = imputer.setStrategy("median").fit(product_data[['price']]).transform(product_data[['price']])
    imputer.setOutputCols(["meanImputedPrice"])
    fit_mean = imputer.setStrategy("mean").fit(product_data[['price']]).transform(product_data[['price']])
    title_df = product_data.select(F.col('title')).fillna('unknown').withColumnRenamed('title', 'unknownImputedTitle')#.show()

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_meanImputedPrice': None,
        'variance_meanImputedPrice': None,
        'numNulls_meanImputedPrice': None,
        'mean_medianImputedPrice': None,
        'variance_medianImputedPrice': None,
        'numNulls_medianImputedPrice': None,
        'numUnknowns_unknownImputedTitle': None
    }
    # Modify res:
    res['count_total'] = title_df.count()
    res['mean_meanImputedPrice'] = fit_mean.select(F.avg(F.col('meanImputedPrice'))).head()[0]
    res['variance_meanImputedPrice'] = fit_mean.select(F.variance(F.col('meanImputedPrice'))).head()[0]
    res['numNulls_meanImputedPrice'] = fit_mean.select([F.count(F.when(F.col('meanImputedPrice').isNull(), 'meanImputedPrice')).alias('meanImputedPrice')]).head()[0]
    res['mean_medianImputedPrice'] = fit_median.select(F.avg(F.col('medianImputedPrice'))).head()[0]
    res['variance_medianImputedPrice'] = fit_median.select(F.variance(F.col('medianImputedPrice'))).head()[0]
    res['numNulls_medianImputedPrice'] = fit_median.select([F.count(F.when(F.col('medianImputedPrice').isNull(), 'medianImputedPrice')).alias('medianImputedPrice')]).head()[0]
    res['numUnknowns_unknownImputedTitle'] = title_df.filter(F.col("unknownImputedTitle") == "unknown").count()



    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_4')
    return res
    # -------------------------------------------------------------------------


def task_5(data_io, product_processed_data, word_0, word_1, word_2):
    # -----------------------------Column names--------------------------------
    # Inputs:
    title_column = 'title'
    # Outputs:
    titleArray_column = 'titleArray'
    titleVector_column = 'titleVector'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    array_df = product_processed_data.select(F.split(F.lower(product_processed_data['title']), ' ').alias('titleArray'))
    word2Vec = M.feature.Word2Vec(vectorSize = 16, minCount=100, seed=102, numPartitions = 4, inputCol = "titleArray", outputCol = "titleVector")
    model = word2Vec.fit(array_df)
    
    product_processed_data_output = model.transform(array_df)


    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'size_vocabulary': None,
        'word_0_synonyms': None,
        'word_1_synonyms': None,
        'word_2_synonyms': None
    }
    # Modify res:
    res['count_total'] = product_processed_data_output.count()
    res['size_vocabulary'] = model.getVectors().count()
    for name, word in zip(
        ['word_0_synonyms', 'word_1_synonyms', 'word_2_synonyms'],
        [word_0, word_1, word_2]
    ):
        res[name] = model.findSynonymsArray(word, 10)
    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_5')
    return res
    # -------------------------------------------------------------------------

def task_6(data_io, product_processed_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    category_column = 'category'
    # Outputs:
    categoryIndex_column = 'categoryIndex'
    categoryOneHot_column = 'categoryOneHot'
    categoryPCA_column = 'categoryPCA'
    # -------------------------------------------------------------------------    

    # ---------------------- Your implementation begins------------------------
    
    sub = product_processed_data[[category_column]]
    string_indexer = K.StringIndexer(inputCol = category_column, outputCol = categoryIndex_column)
    string_indexer_model = string_indexer.fit(sub)
    indexed_df = string_indexer_model.transform(sub)#.show()
    encoder = K.OneHotEncoderEstimator(inputCols = [categoryIndex_column], outputCols = [categoryOneHot_column], dropLast=False)
    encoded_df = encoder.fit(indexed_df).transform(indexed_df)
    pca = K.PCA(k=15, inputCol=categoryOneHot_column, outputCol=categoryPCA_column)
    result_df = pca.fit(encoded_df).transform(encoded_df)



    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'meanVector_categoryOneHot': [None, ],
        'meanVector_categoryPCA': [None, ]
    }
    # Modify res:
    count_total = result_df.count()
    meanVector_categoryOneHot = result_df.select(M.stat.Summarizer.mean(result_df[categoryOneHot_column])).head()[0]
    meanVector_categoryPCA = result_df.select(M.stat.Summarizer.mean(result_df[categoryPCA_column])).head()[0]
    
    res['count_total'] = count_total
    res['meanVector_categoryOneHot'] = meanVector_categoryOneHot
    res['meanVector_categoryPCA'] = meanVector_categoryPCA



    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_6')
    return res
    # -------------------------------------------------------------------------
    
    
def task_7(data_io, train_data, test_data):
    
    # ---------------------- Your implementation begins------------------------
    
    rf = M.regression.RandomForestRegressor(maxDepth=5).setLabelCol("overall").setFeaturesCol("features")
    mdl = rf.fit(train_data)
    preds_df = mdl.transform(test_data)
    metrics = M.evaluation.RegressionEvaluator(predictionCol = "overall", labelCol = "prediction", metricName='rmse')
    test_rmse = metrics.evaluate(preds_df)
    
    
    
    # -------------------------------------------------------------------------
    
    
    # ---------------------- Put results in res dict --------------------------
    res = {
        'test_rmse': None
    }
    # Modify res:
    res['test_rmse'] = test_rmse

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_7')
    return res
    # -------------------------------------------------------------------------
    
    
def task_8(data_io, train_data, test_data):
    
    # ---------------------- Your implementation begins------------------------
    
    train, val = train_data.randomSplit(weights=[0.75, 0.25], seed=102)

    depths = [5, 7, 9, 12]
    candidates = []
    for depth in depths:
        rf = M.regression.RandomForestRegressor(maxDepth=depth).setLabelCol("overall").setFeaturesCol("features")
        mdl = rf.fit(train)
        preds_df = mdl.transform(val)
        metrics = M.evaluation.RegressionEvaluator(predictionCol = "overall", labelCol = "prediction", metricName='rmse')
        val_rmse = metrics.evaluate(preds_df)
        candidates.append((val_rmse, mdl))

    val_rmses = [i[0] for i in candidates]
    valid_rmse_depth_5 = val_rmses[0]
    valid_rmse_depth_7 = val_rmses[1]
    valid_rmse_depth_9 = val_rmses[2]
    valid_rmse_depth_12 = val_rmses[3]

    best_mdl = min(candidates, key=lambda x:x[0])[1]
    final_pred = best_mdl.transform(test_data)
    metrics = M.evaluation.RegressionEvaluator(predictionCol = "overall", labelCol = "prediction", metricName='rmse')
    test_rmse = metrics.evaluate(final_pred)
    
    
    
    # -------------------------------------------------------------------------
    
    
    # ---------------------- Put results in res dict --------------------------
    res = {
        'test_rmse': None,
        'valid_rmse_depth_5': None,
        'valid_rmse_depth_7': None,
        'valid_rmse_depth_9': None,
        'valid_rmse_depth_12': None,
    }
    # Modify res:
    res['test_rmse'] = test_rmse
    res['valid_rmse_depth_5'] = valid_rmse_depth_5
    res['valid_rmse_depth_7'] = valid_rmse_depth_7
    res['valid_rmse_depth_9'] = valid_rmse_depth_9
    res['valid_rmse_depth_12'] = valid_rmse_depth_12

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_8')
    return res
    # -------------------------------------------------------------------------
