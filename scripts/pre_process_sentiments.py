import os
import re

import boto3
import nltk
import pandas as pd
from botocore.exceptions import ClientError
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
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


# Download required NLTK resources (if not done already)
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLTK tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Simple Tokenizer (Alternative if punkt fails)
def tokenizer(text):
    """
    Tokenizes the text using regular expressions.
    This function splits the text into words by finding word boundaries.
    """
    return re.findall(r'\b\w+\b', text.lower())  # Tokenize using regex (lowercase words)

def clean_text(text):
    """
    Clean the raw tweet text by removing unwanted characters, links, and normalizing the text.
    """
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove non-alphanumeric characters (keeping spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def tokenize_and_clean(text):
    """
    Tokenize the text, remove stop words, and lemmatize.
    """
    # Tokenize using the simple tokenizer
    tokens = tokenizer(text)

    # Remove stop words and lemmatize the tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return " ".join(tokens)

def preprocess_tweets(csv_file_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # Clean and preprocess the 'text' column
    df['cleaned_text'] = df['text'].apply(lambda x: clean_text(str(x)))

    # Tokenize and further clean the text
    df['processed_text'] = df['cleaned_text'].apply(lambda x: tokenize_and_clean(x))

    # Remove unnecessary columns if not required
    columns_to_keep = ['text', 'processed_text', 'created', 'screenName', 'id']
    df = df[columns_to_keep]

    # Optionally, save the pre-processed data back to a new CSV
    processed_csv_file_path = csv_file_path.replace('.csv', '_processed.csv')
    df.to_csv(processed_csv_file_path, index=False)

    print(f"Preprocessing complete. Processed data saved to: {processed_csv_file_path}")

    return df

def upload_to_dynamodb(csv_file_path):
    # Read the CSV file using pandas
    data = pd.read_csv(csv_file_path)
    boto3_session = init_boto3_session()
    dynamodb = get_dynamodb(boto3_session)
    table = dynamodb.Table("social_sentiments_refined")
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
    # Example usage
    csv_file_path = "tweets.csv"
    preprocess_tweets(csv_file_path)
    csv_processed = "tweets_processed.csv"
    upload_to_dynamodb(csv_processed)



if __name__ == '__main__':
    main()
