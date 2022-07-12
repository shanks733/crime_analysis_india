
from flask import Flask , request , render_template
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import mpld3
from mpld3 import plugins

app = Flask(__name__)

#models
murder_model = pickle.load(open('f_murder_model.pkl','rb'))
rape_model = pickle.load(open('f_rapemodel.pkl','rb'))
kidnapping_model = pickle.load(open('f_kidnappingmodel.pkl','rb'))


df1 = pd.read_csv('01_District_wise_crimes_committed_IPC_2001_2012.csv')
df2 = df1.copy()
X = df2.iloc[:,0:3]
le = LabelEncoder()
le.fit(X['STATE/UT'])
X['STATE/UT'] = le.transform(X['STATE/UT'])
le2 = LabelEncoder()
le2.fit(X['DISTRICT'])
X['DISTRICT'] = le2.transform(X['DISTRICT'])



#fnc




def district_chart(case_type,state):
    df_x = df1[df1['DISTRICT'] != 'TOTAL']
    df_x = df_x[df_x['STATE/UT'] == state]
    df_x = df_x[df_x['YEAR'] == 2010]
    plt.figure(figsize=(20,20 ))
    splot = sns.barplot(x='DISTRICT', y=case_type, data=df_x)
    for p in splot.patches:
        splot.annotate(format(int(p.get_height()), 'd'), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                       va='center', xytext=(0, 10), textcoords='offset points')
    plt.xticks(rotation=90)
    plt.savefig('new_plot.png')
def state_chart(case_type):
    state_data = df1[df1['DISTRICT']=='TOTAL']
    plt.figure(figsize=(25, 25))
    plt.title(' CASES IN INDA')
    splot = sns.barplot(x='STATE/UT', y=case_type, data=state_data, ci=None)
    for p in splot.patches:
        splot.annotate(format(int(p.get_height()), 'd'), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',va='center', xytext=(0, 10), textcoords='offset points')
    plt.xticks(rotation=90)
    plt.savefig('new_plot.png')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/pred',methods = ['POST'])
def predict():
    feat = [x for x in request.form.values()]
    print(feat)
    year = int(feat[1])
    states = le.transform([feat[2]])
    districts = le2.transform([feat[3]])
    finl_list = [states,districts,year]
    print(finl_list)
    finl = [np.array(finl_list)]

    if feat[0] == 'MURDER':
        pred = murder_model.predict(finl)
    elif feat[0] == 'RAPE':
        pred = rape_model.predict(finl)
    elif feat[0] == 'KIDNAPPING & ABDUCTION':
        pred = kidnapping_model.predict(finl)
    output = int(pred[0])
    if feat[4] == 'STATE_GRAPH':
        state_chart(feat[0])
    elif feat[4] == 'DISTRICT_GRAPH':
        district_chart(feat[0],feat[2])
    return render_template("index.html", prediction_text="cases are {} ".format(output))



if __name__ =="__main__":
    app.run(debug = True)

