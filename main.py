from fastapi import FastAPI
from controllers.user_controller import router as user_router
from controllers.message_controller import router as message_router
from database import get_database, client
from controllers.chat_controller import router as chat_router
from fastapi.middleware.cors import CORSMiddleware
origins = [
    "http://localhost:3000",
]
app = FastAPI()
#CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Include routers
app.include_router(user_router)
app.include_router(message_router)
app.include_router(chat_router)

@app.on_event("startup")
async def startup_event():
    app.state.db = get_database()

@app.on_event("shutdown")
def shutdown_event():
    client.close()
