from boto3 import resource
from os import getenv

dynamodb = resource("dynamodb", 
                     aws_access_key_id="ID_KEY",
                     aws_secret_access_key="ACCES_KEY", 
                     region_name="REGION")