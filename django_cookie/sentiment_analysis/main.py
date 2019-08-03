from django_cookie.sentiment_analysis.imports import *
from django_cookie.sentiment_analysis.preprocessing import *
from django_cookie.sentiment_analysis.aspect import *
from django_cookie.sentiment_analysis.models import *

import nltk
nltk.download('stopwords')
import pickle

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

    filename = 'models/finalized_model.pickle'

    # Classifier
    # model = trainBestClassifier(X_train, Y_train)
    # pickle.dump(model, open(filename, 'wb'))
    # print('model dumped into file')
    #
    model = pickle.load(open(filename, 'rb'))
    print('model loaded into file')
    Y_test = model.predict(X_test)
    printOutput(df_comp_out, Y_test, outFile)
