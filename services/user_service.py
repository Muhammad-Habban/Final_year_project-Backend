from repositories.user_repository import UserRepository

class UserService:
    def __init__(self, user_repository: UserRepository):
        self.user_repository = user_repository

    async def create_user(self, user_data):
        # user_data['password'] = pwd_context.hash(user_data['password'])
        return await self.user_repository.create_user(user_data)
    
    async def find_user_by_email(self, email):
        return await self.user_repository.find_user_by_email(email)

    async def find_user_by_id(self, user_id):
        return await self.user_repository.find_user_by_id(user_id)
