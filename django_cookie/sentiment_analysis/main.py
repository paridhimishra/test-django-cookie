from django_cookie.sentiment_analysis.imports import *
from django_cookie.sentiment_analysis.preprocessing import *
from django_cookie.sentiment_analysis.aspect import *
#from django_cookie.sentiment_analysis.models import *

import nltk
nltk.download('stopwords')
import pickle
import boto3
from  config.settings import production

def printOutput(df, Y, outFile):
    results = []
    print('printOutput called ----')
    for index, id in enumerate(df['example_id']):
        result = str(id) + ';;' + str(Y[index])
        results.append(result)
        print(result)
    print('opening file = ', outFile)
    f = open(outFile, "w")
    f.writelines(results)
    f.close()

def run():
    # Read two train datasets
    df_comp_in = pd.read_csv('django_cookie/sentiment_analysis/data/data-2_train.csv', sep='\s*,\s*', engine='python')
    df_comp_out = pd.read_csv('django_cookie/sentiment_analysis/data/Data-2_test.csv', sep='\s*,\s*', engine='python')

    # Your output file name
    outFile = "output.txt"

    df_comp_out['class'] = np.ones(len(df_comp_out))

    df = pd.concat([df_comp_in, df_comp_out])
    df = preprocessData(df)

    df, X, Y = aspectAnalysis(df)
    X_train = X[0:len(df_comp_in)]
    Y_train = Y[0:len(df_comp_in)]
    X_test = X[len(df_comp_in):]

    filename = 'django_cookie/sentiment_analysis/models/finalized_model.pickle'

    # Classifier
    # model = trainBestClassifier(X_train, Y_train)
    # pickle.dump(model, open(filename, 'wb'))
    # print('model dumped into file')
    #
    model = pickle.load(open(filename, 'rb'))
    print('model loaded into file')
    Y_test = model.predict(X_test)
    printOutput(df_comp_out, Y_test, outFile)

if __name__ == "__main__":
    # Read two train datasets
    df_comp_in = pd.read_csv('data/data-2_train.csv', sep='\s*,\s*', engine='python')
    df_comp_out = pd.read_csv('data/Data-2_test.csv', sep='\s*,\s*', engine='python')

    # Your output file name
    outFile = "output.txt"

    df_comp_out['class'] = np.ones(len(df_comp_out))

    df = pd.concat([df_comp_in, df_comp_out])
    df = preprocessData(df)

    df, X, Y = aspectAnalysis(df)
    X_train = X[0:len(df_comp_in)]
    Y_train = Y[0:len(df_comp_in)]
    X_test = X[len(df_comp_in):]

    # local file storage
    # filename = 'models/finalized_model.pickle'

    # aws file storage
    # s3_resource = boto3.resource('s3')
    # first_bucket = s3_resource.Bucket(name=production.AWS_STORAGE_BUCKET_NAME)
    # first_object = s3_resource.Object(
    #     bucket_name=production.AWS_STORAGE_BUCKET_NAME, key='finalized_model.pickle')
    file = 'finalized_model.pickle'

    s3 = boto3.client('s3', aws_access_key_id=production.AWS_ACCESS_KEY_ID, aws_secret_access_key=production.AWS_SECRET_ACCESS_KEY)
    s3.download_file(production.AWS_STORAGE_BUCKET_NAME, file, file)

    # Classifier
    # model = trainBestClassifier(X_train, Y_train)
    # pickle.dump(model, open(filename, 'wb'))
    # print('model dumped into file')
    #
    model = pickle.load(open(filename, 'rb'))
    print('model loaded into file-------------------------------------------')
    #Y_test = model.predict(X_test)
    #printOutput(df_comp_out, Y_test, outFile)
