from json import dumps
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os
import uuid
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError
import json

def upload_predict(dynamodb, request):
    
    dir_predict_img = os.path.abspath(os.path.join(os.getcwd()))
    file = request.files['predict']
    user = request.form['user']
    try:
        file.save(dir_predict_img+"/predictions/"+user+"/"+file.filename)
    except:
        os.mkdir(dir_predict_img+"/predictions/"+user)
        file.save(dir_predict_img+"/predictions/"+user+"/"+file.filename)
    
    print("File Uploaded")
           
    # Load model
    loaded_model = load_model("model.h5")
   
    # Make prediction
    prediction_path = dir_predict_img+"/predictions/"+user+"/"+file.filename
    img = image.load_img(prediction_path, target_size=(260, 260))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    pred = loaded_model.predict(img)
    class_index = np.argmax(pred)

    if class_index == 0:
        prediction = "Agave"
    elif class_index == 1:
        prediction = "Brócoli"
    elif class_index == 2:
        prediction = "Cebolla"
    elif class_index == 3:
        prediction = "Jitomate"
    elif class_index == 4:
        prediction = "Lechuga"
    elif class_index == 5:
        prediction = "Maíz"
    elif class_index == 6:
        prediction = "Trigo"
    elif class_index == 7:
        prediction = "Zanahoria"
    else:
        prediction = "Desconocido"
    
    # Prepare to save
    imagen_link = "/predictions/"+user+"/"+file.filename;
    cultivo_id = str(uuid.uuid4())
    query = {'prediccion_id': cultivo_id,'user':user, 'image_link': imagen_link, 'prediccion' : prediction}
    table = dynamodb.Table("predicciones")
    # Save to dynamo
    id = table.put_item(Item=query)
    table_cultivos = dynamodb.Table("cultivos")

    try:
        response = table_cultivos.get_item(Key={"cultivo_id":int(class_index)})
       
    except ClientError as err:
        print(err.response["Error"]["Message"])

    try:
        json_string = json.dumps(response["Item"], indent=2, default=str)

        print(json_string)
    except Exception as e:
        print(e)

    return json_string