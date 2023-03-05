from flask import Flask,render_template,url_for,request
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html' , methods=['GET', 'POST'])

@app.route('/prediction' , methods=['GET', 'POST'])
def prediction():
    import pickle
    model = pickle.load(open('model.pkl','rb'))
    if request.method == 'POST':
        IsFirstTime = request.form['IsFirstTime']    
        MIP = request.form['MIP']    
        Units  = request.form['Units']    
        OCLTV  = request.form['OCLTV']    
        DTI  = request.form['DTI']    
        OrigUPB  = request.form['OrigUPB']    
        OrigInterestRate = request.form['OrigInterestRate']
        OrigLoanTerm  = request.form['OrigLoanTerm']
        CreditRange  = request.form['CreditRange']
        LTVRange  = request.form['LTVRange']
        RepayRange  = request.form['RepayRange']
        data = [IsFirstTime,MIP,Units,OCLTV,DTI,OrigUPB,OrigInterestRate,OrigLoanTerm,CreditRange,LTVRange,RepayRange]
        input = np.array(data).reshape(1,-1)
        predict = model.predict(input)
    return render_template('home.html',predictValue = predict)
    # return render_template('output.html',predictValue = 1)


if (__name__ == '__main__'):
    app.run(host='0.0.0.0',port=8080)