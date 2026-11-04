from typing import TypedDict, Annotated, Sequence, List, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

from tools import AgentMessage

from strAItegize_common.others.recruitment_choice import RecruitmentChoice
from strAItegize_common.others.order import Order

# Unit Agents
class UnitInputState(TypedDict):
    units: List[str]
    regions: List[str]

    player_id: str
    unit_id: str
    supply_surplus: int = 0
    upkeep_phase: bool = False
    recent_events: List[str]

    unit_messages: Annotated[Sequence[BaseMessage], add_messages]

    unit_orders: dict[str, Order]

class UnitState(TypedDict):
    units: List[str]
    regions: List[str]

    player_id: str
    unit_id: str
    recent_events: List[str]

    unit_messages: Annotated[Sequence[BaseMessage], add_messages]

    messages_to_send: List[AgentMessage]

    unit_orders: dict[str, Order]

class UnitOutputState(TypedDict):
    unit_messages: Annotated[Sequence[BaseMessage], add_messages]

    messages_to_send: List[AgentMessage]

    unit_orders: dict[str, Order]

# Region Agents
class RegionInputState(TypedDict):
    units: List[str]
    regions: List[str]

    player_id: str
    region_id: str
    supply_surplus: int = 0
    upkeep_phase: bool = False
    recent_events: List[str]

    region_messages: Annotated[Sequence[BaseMessage], add_messages]

    region_recruitment: dict[str, RecruitmentChoice]

class RegionState(TypedDict):
    units: List[str]
    regions: List[str]

    player_id: str
    region_id: str
    recent_events: List[str]

    region_messages: Annotated[Sequence[BaseMessage], add_messages]

    messages_to_send: List[AgentMessage]

    region_recruitment: dict[str, RecruitmentChoice]

class RegionOutputState(TypedDict):
    region_messages: Annotated[Sequence[BaseMessage], add_messages]

    messages_to_send: List[AgentMessage]

    region_recruitment: dict[str, RecruitmentChoice]