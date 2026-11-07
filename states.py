from typing import TypedDict, Annotated, Sequence, List, Optional, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

from tools import AgentMessage

from strAItegize_common.others.recruitment_choice import RecruitmentChoice
from strAItegize_common.others.order import Order

class DecentralizedState(TypedDict):
    units: List[str]
    regions: List[str]
    recent_events: List[str]
    
    input_prompt: Optional[BaseMessage]
    upkeep_phase: bool = False
    supply_surplus: int = 0
    
    player_id: str
    init_msg: str
    
    blocked_tools: list[str]
    
    unit_orders: dict[str, Order]
    region_recruitment: dict[str, RecruitmentChoice] 
    
    agents_messages: dict[str, list[BaseMessage]]
    agent_types: dict[str, Literal["unit", "region"]]
    
    # Navigation variables
    next_action: Literal["unit", "region", "end"]
    active_agent: str
    active_agents: list[str]
    waiting_agents: list[str]
    finished_agents: list[str]
    
    # Shared channels with AgentState
    agent_id: Optional[str]
    agent_type: Optional[Literal["unit", "region"]]
    agent_messages: Optional[Annotated[Sequence[BaseMessage], add_messages]]
    messages_to_send: Optional[List[AgentMessage]]

class AgentState(TypedDict):
    player_id: str
    agent_id: str
    agent_type: Literal["unit", "region"]
    
    units: List[str]
    regions: List[str]
    recent_events: List[str]
    blocked_tools: list[str]

    agent_messages: Annotated[Sequence[BaseMessage], add_messages]
    messages_to_send: List[AgentMessage]
    
    unit_orders: dict[str, Order]
    region_recruitment: dict[str, RecruitmentChoice]
