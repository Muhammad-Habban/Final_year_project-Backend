from fastapi import FastAPI
from controllers.user_controller import router as user_router
from controllers.message_controller import router as message_router
from controllers.chat_controller import router as chat_router
from database import get_database, client
from fastapi.middleware.cors import CORSMiddleware

# List of origins allowed for CORS
origins = [
    "http://localhost:3000",  # You can add other domains as needed
]

# Initialize FastAPI app
app = FastAPI()

# CORS Middleware to allow specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allowing specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Include routers for user, message, and chat routes
app.include_router(user_router)
app.include_router(message_router)
app.include_router(chat_router)

# Startup event handler to initialize the database connection
@app.on_event("startup")
async def startup_event():
    app.state.db = get_database()  # Initialize the database connection

# Shutdown event handler to close the database client connection
@app.on_event("shutdown")
def shutdown_event():
    client.close()  # Close the database connection when the app shuts down
