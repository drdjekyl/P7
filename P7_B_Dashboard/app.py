# -*- coding: utf-8 -*-
from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def getHomeLayout():
    return render_template("home.html")

@app.route("/portal")
def getPortalLayout():
    return render_template("portal.html")

@app.route("/contact")
def getContactLayout():
    return render_template("contact.html")


if __name__ == "__main__":
    app.run(debug=True)