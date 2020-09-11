from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import pickle

app = Flask(__name__) # initializing a flask app
CORS(app)

@app.route('/',methods=['GET'])  # route to display the home page

def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI

def predict():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            fixed_acidity = request.form['fixed_acidity']
            volatile_acidity = request.form['volatile_acidity']
            citric_acid = request.form['citric_acid']
            residual_sugar = request.form['residual_sugar']
            chlorides = request.form['chlorides']
            free_sulphur_dioxide = request.form['free_sulphur_dioxide']
            total_sulphur_dioxide = request.form['total_sulphur_dioxide']
            density = request.form['density']
            pH = request.form['pH']
            sulphates = request.form['sulphates']
            alcohol = request.form['alcohol']

            filename = 'clf.sav'
            loaded_model = pickle.load(open(filename, 'rb'))
            scaler = pickle.load(open('scaler.sav', 'rb'))
            pca_model = pickle.load(open('pca.sav','rb'))
            scale_model=scaler.transform([[fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,
                  free_sulphur_dioxide,total_sulphur_dioxide,density,pH,sulphates,alcohol]])
            principle_data=pca_model.transform(scale_model)
            prediction=loaded_model.predict(principle_data)
            if prediction == 3:
                return render_template('3.html')
            elif prediction == 4:
                return render_template('4.html')
            elif prediction == 5:
                return render_template('5.html')
            elif prediction == 6:
                return render_template('6.html')
            elif prediction == 7:
                return render_template('7.html')
            else:
                return render_template('8.html')

        except Exception as e:
            print('Exception message is: ', e)
            return 'Something is Wrong'
    else:
            return render_template('index.html')







if __name__ == '__main__':
    app.run(debug=True)


