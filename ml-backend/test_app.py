from fastapi import FastAPI
from fastapi.responses import JSONResponse
from recommend import recommend_videos

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Root OK"}

@app.get("/recommend")
def recommend(user_id: int):
    print(f"🔍 Received user_id: {user_id}")
    result = recommend_videos(user_id)
    return JSONResponse(content=result)
