from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
import os
load_dotenv()


def get_database():
    db_name = "Final_Year_Project"
    return client[db_name]

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise EnvironmentError("DATABASE_URL not set in .env file")

client = AsyncIOMotorClient(DATABASE_URL)