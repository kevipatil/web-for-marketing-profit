from flask import Flask
from flask import render_template,request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)


# model = pickle.load(open("SLR.pkl","rb"))


@app.route("/")
def admin():
    return render_template("admin.html")

@app.route("/slr", methods = ['post'])
def profit_market():
    model = pickle.load(open("MLR_Model.pkl","rb"))

    rnd = float(request.form.get("rnd"))
    admin = float(request.form.get("admin"))
    marketing = float(request.form.get("marketing"))
    state = request.form.get("state")

    profit = model.predict(pd.DataFrame(columns=["R&D Spend","Administration","Marketing Spend","State"], 
                                   data=np.array([rnd, admin, marketing, state]).reshape(1, 4)))
    return render_template("slr.html", profit = profit)
    

if __name__=="__main__":
    app.run(debug=True)
