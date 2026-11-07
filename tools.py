from pydantic import BaseModel, Field
from typing import Annotated, Literal
from langchain.tools import InjectedState
from langchain.messages import ToolMessage
from langgraph.types import Command
from langchain_core.tools import tool, InjectedToolCallId

from strAItegize_common.client.communication.communication_context import CommunicationContext
from strAItegize_common.others.recruitment_choice import RecruitmentChoice
from strAItegize_common.others.order import Order
from strAItegize_common.client.tools import get_generic_tools

class AgentMessage(BaseModel):
    sender: str = Field(
        description="Unit Agent ID of the sender agent."
    )
    sendee: str = Field(
        description="Unit Agent ID of the sendee agent."
    )
    content: str = Field(
        description="Text message from the sender to the sendee."
    )

@tool
def message_agent(
    sendee: str, 
    content: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Send message to the specified agent."""
    
    if sendee not in state["units"] and sendee not in state["regions"]:
        return f"Error: Agent with id: {sendee} does not exist."
    
    agent_msg = AgentMessage(
        sender=state["agent_id"],
        sendee=sendee,
        content=content
    )
    
    return Command(
        update={
            "messages_to_send": [
                agent_msg
            ],
            "agent_messages": [
                ToolMessage(
                    content=f"Message sent to agent {sendee}: '{content}'",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )

@tool
def order_move(
    move_q: int,
    move_r: int,
    move_s: int,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Assigning a movement order to your unit."""
    if state["agent_type"] != "unit":
        return f"Error: You are not allowed to give orders."

    if move_q + move_r + move_s != 0:
        return f"Error: Given relative coordinates need to sum up to zero."

    order = Order(
        order_type="move",
        unit_id=state["agent_id"],
        move_q=move_q,
        move_r=move_r,
        move_s=move_s
    )
    
    unit_orders = state["unit_orders"]
    unit_orders[state["agent_id"]] = order    
    
    return Command(
        update={
            "unit_orders": unit_orders,
            "agent_messages": [
                ToolMessage(
                    content=f"Your next action was assigned properly: you will move by the movement vector: ({move_q}, {move_r}, {move_s}).",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )
    
@tool
def order_attack(
    target_id: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Assigning a movement order to your unit."""
    if state["agent_type"] != "unit":
        return f"Error: You are not allowed to give orders."

    order = Order(
        order_type="attack",
        unit_id=state["agent_id"],
        target_id=target_id
    )
    
    unit_orders = state["unit_orders"]
    unit_orders[state["agent_id"]] = order    
    
    return Command(
        update={
            "unit_orders": unit_orders,
            "agent_messages": [
                ToolMessage(
                    content=f"Your next action was assigned properly: you will attempt to attack the {target_id} unit.",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )
    
@tool
def order_support(
    target_id: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Assigning a movement order to your unit."""
    if state["agent_type"] != "unit":
        return f"Error: You are not allowed to give orders."

    order = Order(
        order_type="support",
        unit_id=state["agent_id"],
        target_id=target_id
    )
    
    unit_orders = state["unit_orders"]
    unit_orders[state["agent_id"]] = order    
    
    return Command(
        update={
            "unit_orders": unit_orders,
            "agent_messages": [
                ToolMessage(
                    content=f"Your next action was assigned properly: you will attempt to support the {target_id} unit.",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )
    
@tool
def order_hold(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Assigning a movement order to your unit."""
    if state["agent_type"] != "unit":
        return f"Error: You are not allowed to give orders."

    order = Order(
        order_type="hold",
        unit_id=state["agent_id"]
    )
    
    unit_orders = state["unit_orders"]
    unit_orders[state["agent_id"]] = order    
    
    
    return Command(
        update={
            "unit_orders": unit_orders,
            "agent_messages": [
                ToolMessage(
                    content=f"Your next action was assigned properly: you will hold your current position.",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )

@tool    
def recruit_unit(
    unit_type: Literal["light_infantry", "heavy_infantry", "cavalry"],
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Adding a recruitment choice to the recruitment list."""
    
    if state["agent_id"] not in state["regions"]:
        return f"Error: You are not allowed to make recruitment decisions."
    
    recruitment_choice = RecruitmentChoice(
        unit_type=unit_type,
        region=state["agent_id"]
    )
    
    region_recruitment = state["region_recruitment"]
    region_recruitment[state["agent_id"]] = recruitment_choice    
    
    
    return Command(
        update={
            "region_recruitment": region_recruitment,
            "agent_messages": [
                ToolMessage(
                    content=f"Your recruitment choice was assigned properly: you will attempt to recruit a {unit_type} unit.",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )

@tool
def get_recent_events(state: Annotated[dict, InjectedState]) -> str:
    """Read the list of all events that took place this round.
    
    Returns:
        The list of events from this round.    
    """
    events = state["recent_events"]
    
    # Formatting the tool output to be more readable for the LLM
    if len(events) > 0:
        observation = "Events that had occurred on this round:\r\n"
        for event in events:
            observation += f"- {event},"
        observation += "."
    else:
        observation = "There've been no events since the last round."
    
    return observation