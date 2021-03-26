from werkzeug.middleware.dispatcher import DispatcherMiddleware
from app_dash import appdash as app_dash

######################## CONFIGURE SERVER ############################
application = DispatcherMiddleware(app_flask, {
    '/dashboard': app_dash.server,
})  