# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify
import flask

######################## CONFIGURE SERVER ############################
appflask = Flask(__name__)
dash_app = dash.Dash(__name__, server=app, external_stylesheets=external_stylesheets, url_base_pathname='/dashboard/')
######################## CONFIGURE ROUTES ############################
@app.route("/")
def getHomeLayout():
    return render_template("home.html")

@app.route("/dash")
def getDashLayout():
    return flask.redirect('/dashboard/')

@app.route('/portal', methods=["POST"])
def portal_access():
    if request.method == "POST":
        name = request.form['name']
        return render_template("dash_embedded.html", value=name)

@app.route("/contact")
def getContactLayout():
    return render_template("contact.html")

################################# RUN SERVER #####################################
if __name__ == "__main__":
    app.run(debug=True)