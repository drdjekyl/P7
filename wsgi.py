from werkzeug.middleware.dispatcher import DispatcherMiddleware
from app_dash import appdash as app_dash
from app_flask import appflask as app_flask
from werkzeug.serving import run_simple


######################## CONFIGURE SERVER ############################
application = DispatcherMiddleware(app_flask, {
    '/dashboard': app_dash.server,
})