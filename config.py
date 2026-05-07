from dotenv import load_dotenv
import os

load_dotenv()

CDS_API_KEY = os.getenv("CDS_API_KEY")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")