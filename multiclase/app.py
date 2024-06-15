#region Imports
from flask import Flask, request, jsonify, Response
from flask_cors import CORS, cross_origin
from Modules.Db.Database import dynamodb
from Modules.Error.ErrorsHandler import init_handler

#endregion Imports

#region App Instance (No modify)
app = Flask(__name__)
init_handler(app)
#endregion App Instance

#region Server Root
@app.route('/', methods=['GET'])
def index():
    return "Forbidden", 403
#endregion

#region Predict
@app.route('/predict/', endpoint="predict", methods=["POST","OPTIONS"])
@cross_origin(origin='*')

def Predict(id = None):

    """Predict. Methods Allowed POST GET PUT DELETE OPTIONS"""

    if id is not None:
        if request.method == "OPTIONS":
            return "OK",200
    else:
        #create
        if request.method == 'POST':
            result = upload_predict(dynamodb, request)
            return jsonify(str(result))
        
#endregion Predict

#region Feedback
@app.route('/feedback/', endpoint="feedback", methods=["POST","OPTIONS"])
@cross_origin(origin='*')

def Feedback(id = None):

    """Predict. Methods Allowed POST GET PUT DELETE OPTIONS"""

    if id is not None:
        if request.method == "OPTIONS":
            return "OK",200
    else:
        #create
        if request.method == 'POST':
            from Modules.Feedback.Feedback import upload_feedback
            result = upload_feedback(request)
            return jsonify(result)
        
#endregion Feedback

#region Help Endpoint
@app.route('/help', endpoint="help", methods = ["GET","OPTIONS"])
@cross_origin(origin='*')
def help():
    """Print available functions."""
    func_list = {}
    for rule in app.url_map.iter_rules():
        if rule.endpoint != 'static':
            func_list[rule.rule] = app.view_functions[rule.endpoint].__doc__
    return jsonify(func_list)
#endregion Help Endpoint

#region Main (No modify)
if __name__ == "__main__":
    app.run(debug=True)
#endregion