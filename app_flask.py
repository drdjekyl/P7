# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify
import flask

######################## CONFIGURE SERVER ############################
appflask = Flask(__name__)
######################## CONFIGURE ROUTES ############################
@appflask.route("/")
def getHomeLayout():
    return render_template("home.html")

@appflask.route('/portal', methods=["POST"])
def portal_access():
    if request.method == "POST":
        name = request.form['name']
        return render_template("dash_embedded.html", value=name)

@appflask.route("/contact")
def getContactLayout():
    return render_template("contact.html")

################################# RUN SERVER #####################################
