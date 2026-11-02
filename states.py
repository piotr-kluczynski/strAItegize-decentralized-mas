from typing import TypedDict, Annotated, Sequence, List, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

from strAItegize_common.others.recruitment_choice import RecruitmentChoice
from strAItegize_common.others.order import Order

class UnitInputState(TypedDict):
    units: List[str]
    regions: List[str]

    player_id: str
    supply_surplus: int = 0
    upkeep_phase: bool = False
    recent_events: List[str]

    unit_messages: Annotated[Sequence[BaseMessage], add_messages]


    order_list: List[Order] = []
    recruitment_choice_list: List[RecruitmentChoice] = []

class UnitState(TypedDict):
    units: List[str]
    regions: List[str]

    player_id: str

    unit_messages: Annotated[Sequence[BaseMessage], add_messages]

    order_list: List[Order] = []
    recruitment_choice_list: List[RecruitmentChoice] = []

class UnitOutputState(TypedDict):
    unit_messages: Annotated[Sequence[BaseMessage], add_messages]

    order_list: List[Order] = []
    recruitment_choice_list: List[RecruitmentChoice] = []