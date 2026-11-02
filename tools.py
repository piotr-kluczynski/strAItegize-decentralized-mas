from pydantic import BaseModel
from typing import Field, Annotated
from langchain.tools import InjectedState
from langchain_core.tools import tool, InjectedState

from strAItegize_common.others.recruitment_choice import RecruitmentChoice

class UnitAgentMessage(BaseModel):
    sender: str = Field(
        description="Unit Agent ID of the sender agent."
    )
    sendee: str = Field(
        description="Unit Agent ID of the sendee agent."
    )
    content: str = Field(
        description="Text message from the sender to the sendee."
    )

# Unit-Agent Tools
def get_unit_tools() -> list:
    @tool
    class MessageAgent(BaseModel):
        """Data class defining a message to be sent to another agent."""
        sendee: str = Field(
            description="The identifier of the agent receiving the message."
        )

        content: str = Field(
            description="The content of the message sent to the sendee."
        )

    @tool
    class OrderMovement(BaseModel):
        """Data class defining a movement order given to your unit."""
        move_q: int = Field(
            description="The movement vector of q axis in cubic coordinate system.",
        )
        move_r: int = Field(
            description="The movement vector of r axis in cubic coordinate system.",
        )
        move_s: int = Field(
            description="The movement vector of s axis in cubic coordinate system.",
        )

    @tool 
    class OrderAttack(BaseModel):
        """Data class defining an attack order given to your unit."""
        target_id: str = Field(
            description="Identifier of the unit targeted by the attack order.",
        )

    @tool 
    class OrderSupport(BaseModel):
        """Data class defining a support order given to your unit."""
        target_id: str = Field(
            description="Identifier of the unit targeted by the support order.",
        )

    @tool
    class OrderHold(BaseModel):
        """
        Data class defining a hold order given to your unit.
        """
        pass

    @tool
    class RecruitUnit(BaseModel):
        """Tool for adding a recruitment choice to the recruitment list."""
        recruitment_choice: RecruitmentChoice = Field(
            description="The choice of where to recruit which type of unit."
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