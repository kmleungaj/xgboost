from flask import Flask, jsonify, request
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import sqlite3

app = Flask(__name__)

@app.route('/suggestions', methods=['POST'])
def suggestions():

    try:
        ## get user data
        user_info = request.json

        ## using 0, 1, 2, 3 instead of gender, height, weight, bodyType
        ## because need to match column names in training data set
        user = [{"0": user_info["gender"]
                , "1": user_info["height"]
                , "2": user_info["weight"]
                , "3": user_info["bodyType"]
        }]

        user_df = pd.json_normalize(user)

        ## connect to db
        con = sqlite3.connect('../AIFit/wwwroot/NutriDb.db')

        ## get data from db for training
        ## format: Gender | Height | Weight | TargetBodyType | WorkoutId
        df = pd.read_sql_query("select Gender, Height, Weight, TargetBodyType, r.WorkoutId\
                                    from Customer c \
                                    left join Recommendations r\
                                    where c.Id = r.CustomerId"
                                , con)

        ## need to pass in as integer for xgb.train()
        df = df.rename(columns={"Gender": 0, "Height": 1, "Weight": 2, "TargetBodyType": 3, "WorkoutId": 4})

        ## replace by sql statement to get table instead of using external csv later
        ## get training data
        # df = pd.read_csv("Train.csv", header = None)

        ## columns used for prediction
        X = df.drop([4], axis=1).copy()

        ## columns to predict
        y = df[4].copy()

        ## split the data into test set and train set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        ## convert data set into DMatrix for XGBoost
        train = xgb.DMatrix(X_train, label=y_train)
        test = xgb.DMatrix(X_test, label=y_test)

        ## generate a model
        param = {'max_depth': 2, 'eta': 1, 'objective': 'multi:softmax'
                    , 'num_class':11, 'eval_metric': 'mlogloss'}
        evallist = [(test, 'eval'), (train, 'train')]
        num_round = 5
        bst = xgb.train(param, train, num_round, evallist)

        new = xgb.DMatrix(user_df)
        ypred = bst.predict(new)

        return {'WorkoutId': int(ypred[0]), 'Status': 1}

    except:
        return {'Status': 2}

if __name__ == '__main__':
    app.run()