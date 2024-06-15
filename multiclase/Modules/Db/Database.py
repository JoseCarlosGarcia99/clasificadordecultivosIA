from boto3 import resource
from os import getenv

dynamodb = resource("dynamodb", 
                     aws_access_key_id="AKIA5ALIJLLGCKX4F6IN",
                     aws_secret_access_key="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", 
                     region_name="us-east-2")