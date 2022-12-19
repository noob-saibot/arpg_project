import logging.config
import json

from fastapi import FastAPI
import torch


from data import InputState, OutputAction
from actor_critic import ActorCriticNetwork

import __main__
__main__.ActorCriticNetwork = ActorCriticNetwork

model = torch.load('../solver/trpo_discrete_model.pt', map_location=torch.device('cpu'))
model.eval()

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
    action = model.select_action(input_state.state)[0]
    return OutputAction(
        **{
            "session_id": input_state.session_id,
            "action": str(action),
            "state": input_state.state,
        }
    )
