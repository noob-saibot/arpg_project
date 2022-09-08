import logging
import json
import random

from fastapi import FastAPI

from data import InputState, OutputAction

logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
logger = logging.getLogger(__name__)

app = FastAPI()

with open("../available_methods.json", "r") as file:
    AVAILABLE_METHODS = json.load(file)


@app.get("/")
def status_check() -> str:
    return "<h1>Ready to use</p>"


@app.get("/available_methods")
async def get_methods() -> dict:
    return AVAILABLE_METHODS


@app.post("/get_next_action")
async def get_next_action(input_state: InputState) -> OutputAction:
    logger.info(f"Got request {InputState}")
    logger.info(f"Sent response {OutputAction}")
    return OutputAction(
        **{
            "session_id": input_state.session_id,
            "action": random.choice(list(AVAILABLE_METHODS["methods"].keys())),
            "state": input_state.state,
        }
    )
