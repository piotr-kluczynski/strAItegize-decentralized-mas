from typing import Optional

from strAItegize_common.others.order import Order, describe_order

def unit_agent_prompt(player_id: str, unit_id: str, unit_orders: dict[str, Optional[Order]], regions: list[str]) -> str:
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

    <Current Gamestate>"""

    unit_instructions  + f"Your country's player_id is {player_id} and your unit_id is {unit_id}."

    unit_instructions += """
    Your country consists out of the following Unit Agents, alongside their current decisions:
    """

    for unit, order in unit_orders.items():
        order_desc = describe_order(order) if order else "no move planned yet"
        unit_instructions += f"- {unit}: {order_desc}\r\n"

    unit_instructions += """
    Your country consists out of the following regions:
    """

    for region in regions:
        unit_instructions += f"- {region}\r\n"

    unit_instructions += """
    </Current Gamestate>
    <Available Tools>
    Tools provided to you fall into 4 categories:
    - Observation Tools: observe_unit, observe_region, get_units, get_regions, get_unit_details - These tools allow you to inspect the current state of the simulation.
    - Meta Tools: get_players, get_player_conversation, get_order_list, get_recent_events, get_game_rules - These tools allow you to inspect information about the current round and the simulation session, as well as its rules.
    - Action Tools: MessageAgent, OrderMovement, OrderAttack, OrderSupport, OrderHold, RecruitUnit - These tools allow you to take strategic actions, and communicate with other unit agents.
    - Reflection Tools: think_tool - This tool allows you to reflect on gathered information and evaluate possible courses of action.
    </Available Tools>

    <Instructions>
    Think like an independent, independent but part of the nation's military force. Follow this general decision-making process:
    1. Observe carefully: Gather the information necessary to understand the current situation of your country and its opponents.
    2. Reflect: Use think_tool to analyze the information you have gathered and identify important opportunities, threats, and uncertainties.
    3. Plan strategically: Determine which actions are most beneficial for your country in the long term.
    4. Communicate with other Agents: When opportunity arises, contact other unit agents, to ensure the success of your country.
    5. Give orders: Select the most beneficial moves for your unit, taking into consideration your environment, unit's capabilities, and other agent's plans.
    6. Stop when satisfied: Once your strategic objectives have been sufficiently addressed, stop planning. Do not repeatedly change your move or chit-chat with other agents in pursuit of a theoretically perfect plan.

    **CRITICAL Do not confuse activity with progress. Additional observations, reflections, or messages should only be performed when they can meaningfully improve your decision or execution.**
    </Instructions>
    """

    return unit_instructions

def unit_agent_msg_prompt(sender: str, content: str) -> str:
    return f"""
    You have received new message from another unit: {sender}. 
    Its content is as follows: {content}

    Take newly received information into account, and decide whether to change your already made decisions, rethink your strategy, etc.
    """

def msg_no_response_prompt() -> str:
    return f"""
    The receipent of your message has not responded yet. 
    Return to your previous activites, taking this information into account.
    """