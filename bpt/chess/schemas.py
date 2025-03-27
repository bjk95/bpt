from pydantic import BaseModel    

class GameState(BaseModel):
    board: list[list[str]] 
    moves: list[str]
    result: str