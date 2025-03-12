from fastapi import FastAPI
from controllers.user_controller import router as user_router
from controllers.message_controller import router as message_router
from database import get_database, client

print("This is a database:", get_database)
print("This is client:", client["Final_Year_Project"])

app = FastAPI()

# Include routers
app.include_router(user_router)
app.include_router(message_router)

@app.on_event("startup")
async def startup_event():
    app.state.db = get_database()

@app.on_event("shutdown")
def shutdown_event():
    client.close()
