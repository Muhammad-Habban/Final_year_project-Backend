from bson import ObjectId

class UserRepository:
    def __init__(self, db):
        self.collection = db['users'] 

    async def create_user(self, user_data):
        result = await self.collection.insert_one(user_data)
        user_data['id'] = str(result.inserted_id)
        return user_data
    
    async def find_user_by_email(self, email):
        user = await self.collection.find_one({"email": email})
        if user:
            user['id'] = str(user['_id'])
            del user['_id']
        return user

    async def find_user_by_id(self, user_id):
        user = await self.collection.find_one({"_id": ObjectId(user_id)})
        if user:
            user['id'] = str(user['_id'])
            del user['_id']
        return user
