from fastapi import FastAPI
from controllers.user_controller import router as user_router
from database import get_database, client
print("This is a database ")
print(get_database)
print("This is client ")
print(client["Final_Year_Project"])

app = FastAPI()
app.include_router(user_router)

@app.on_event("startup")
async def startup_event():
    app.state.db = get_database()

@app.on_event("shutdown")
def shutdown_event():
    client.close()
