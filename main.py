from fastapi import FastAPI
from controllers.user_controller import router as user_router
from controllers.chat_controller import router as chat_router
from database import get_database, client

app = FastAPI()
app.include_router(user_router)
app.include_router(chat_router)

@app.on_event("startup")
async def startup_event():
    app.state.db = get_database()

@app.on_event("shutdown")
def shutdown_event():
    client.close()
