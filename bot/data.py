from pydantic import BaseModel


class InputState(BaseModel):
    session_id: str
    description: str | None = None
    state: dict


class OutputAction(BaseModel):
    session_id: str
    action: str
    state: dict
