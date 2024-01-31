from openai import AsyncAzureOpenAI
from dotenv import load_dotenv
load_dotenv()

# gets the API Key from environment variable AZURE_OPENAI_API_KEY
client = AsyncAzureOpenAI(
    # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning
    api_version="2023-07-01-preview",
    # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
    azure_endpoint="https://simon-aiops.openai.azure.com/",
)