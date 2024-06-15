from json import dumps
from bson.json_util import loads
import os
from werkzeug.security import generate_password_hash

def upload_feedback(request):
    dir_predict_img = os.path.abspath(os.path.join(os.getcwd()))
    print(dir_predict_img)
    file = request.files['predict']
    category = request.form['form_category']
    try:
        file.save(dir_predict_img+"/data/train/"+category+"/"+file.filename)
    except:
        os.mkdir(dir_predict_img+"/data/train/"+category)
        file.save(dir_predict_img+"/data/train/"+category+"/"+file.filename)
    
    print("File Uploaded")
    
    data = {
            "Category" : category,
            "Message" : "Gracias, la imagen fue almacenada para el siguiente entrenamiento.",
        }
    return data