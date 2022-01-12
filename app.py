from flask import Flask, request, jsonify, render_template
import joblib
from pyforest import *
import numpy as np

#naming our app as app
app = Flask(__name__,template_folder='./templates',static_folder='./static')

# #loading the pickle file for creating the web app
colVals = pickle.load(open('colDict.sav', 'rb'))
standardscaler= pickle.load(open('StandardScaler.sav', 'rb'))
mymodel = joblib.load(open('adb-model.pkl', 'rb'))


#function to send all unique values of a column to html
def columnValues():
    return colVals['region'] ,colVals['manufacturer'],colVals['condition'],colVals['cylinders'],colVals['fuel'],colVals['transmission'],colVals['drive'],colVals['size'],colVals['type'],colVals['paint_color']
    
regions, manufacturers, conditions, cylinderss, fuels, transmissions, drives, sizes, types, colors = columnValues()
columnNames = ['region','year', 'manufacturer', 'condition', 'cylinders', 'fuel','odometer', 'transmission', 'drive', 'size', 'type', 'paint_color']
# #function to generate encoded feature set obtained from website
def getFeatures(html_features):

    year = float(html_features[10])
    odometer = html_features[11]
    year_odometer = pd.DataFrame(data=[[year,odometer]],columns=['year','odometer'])
    num_features=standardscaler.transform(year_odometer[['year','odometer']]).flatten()
    year = num_features[0]
    odometer = num_features[1]

    coded_features = np.zeros(10)
    cols = [regions, manufacturers, conditions, cylinderss, fuels, transmissions, drives, sizes, types, colors]
    
    for f in range(len(colVals)):
        code = cols[f].index(html_features[f]) + 1
        coded_features[f] = int(code)

    coded_features = np.insert(coded_features ,1, year )
    coded_features = np.insert(coded_features ,6, odometer)

    return coded_features



# #defining the different pages of html and specifying the features required to be filled in the html form
@app.route("/")
def home():
    return render_template('index.html', regions = regions , manufacturers = manufacturers , conditions = conditions , cylinderss = cylinderss , fuels = fuels , 
    transmissions = transmissions , drives = drives ,sizes = sizes , types = types , colors = colors)

# #creating a function for the prediction model by specifying the parameters and feeding it to the ML model

@app.route("/test" , methods=['GET', 'POST'])

def test():
    # select = request.form.get('region')
    html_features = [x for x in request.form.values()]

    coded_features = getFeatures(html_features)

    final_features=pd.DataFrame(data=[coded_features] , columns = columnNames)


    prediction = mymodel.predict(final_features)
    price=np.exp(prediction[0])
    output = round(price, 1)

    final = "$"+str(output)
    return render_template("index.html",  regions = regions , manufacturers = manufacturers , conditions = conditions , cylinderss = cylinderss , fuels = fuels , 
    transmissions = transmissions , drives = drives ,sizes = sizes , types = types , colors = colors, prediction_text = " {}".format(final))


#running the flask app
if __name__== "__main__":
    app.run(debug=True)