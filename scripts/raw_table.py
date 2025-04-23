import os

import boto3
import pandas as pd
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

# Initialize DynamoDB resource

def init_boto3_session():
    return boto3.Session(
        aws_access_key_id=os.environ["AWS_ACCESS_KEY"],
        aws_secret_access_key=os.environ["AWS_SECRET_KEY"],
        region_name=os.environ["AWS_REGION"]
    )

def get_dynamodb(session: boto3.Session):
    return session.resource("dynamodb")


def upload_to_dynamodb(csv_file_path):
    # Read the CSV file using pandas
    data = pd.read_csv(csv_file_path)
    boto3_session = init_boto3_session()
    dynamodb = get_dynamodb(boto3_session)
    table = dynamodb.Table("social_sentiments_raw")
    data = list(data.iterrows())[:100]

    # Iterate through each row in the DataFrame
    for index, row in data:
        item = row.to_dict()
        print(item)
        new_item = {k: str(v) for k, v in item.items()}
        # Add data to DynamoDB table
        try:
            response = table.put_item(Item=new_item)
            print(f"Successfully inserted item: {new_item}")
        except ClientError as e:
            print(f"Error inserting item {new_item}")




def main():
    print(init_boto3_session())
    csv_file_path = "tweets.csv"
    upload_to_dynamodb(csv_file_path)


if __name__ == "__main__":
    main()
