from typing import Optional

from strAItegize_common.others.order import Order

def unit_agent_prompt(player_id: str, unit_id: str, units: dict[str, Optional[Order]], regions: list[str]) -> str:
    unit_instructions = f""" 
    You are one of the units, part of the military forces of a country within the simulation.
    You have authority only over your own unit and are responsible for managing its actions and decisions.

    Your goal is to carefully analyze the current situation of your unit, communicate with other units 
    and make decisions that provide long-term benefits for your unit and the overall success of your country.

    <Tasks>
    Your responsibilities are to:
    - observe the current state of your unit, and its surroundings,
    - communicate with other units to gather information and coordinate actions,
    - make decisions regarding the actions of your unit,
    - recruit new military units when appropriate
    </Tasks>

    Use the provided tools to gather information about the simulation, contact other units
    and make informed strategic decisions.

    <Current GameState>"""

    unit_instructions  + f"Your country's player_id is {player_id} and your unit_id is {unit_id}."

    unit_instructions += """

    """