from pydantic import BaseModel
import typing as T


class InputState(BaseModel):
    session_id: str
    description: T.Optional[str] = None
    state: dict


class OutputAction(BaseModel):
    session_id: str
    action: str
    state: dict
