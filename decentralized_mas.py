from strAItegize_common.client.prompts import round_start_prompt, simulation_start_prompt, units_recruitment_prompt
from strAItegize_common.others.order import Order
from strAItegize_common.others.recruitment_choice import RecruitmentChoice
from langchain_core.messages import SystemMessage

from decentralized_graph import get_decentralized_mas

from strAItegize_common.client.communication.communication_context import CommunicationContext
from strAItegize_common.client.multiagent_system import MultiAgentSystem



class DecentralizedMAS(MultiAgentSystem):
    def __init__(self, communication: CommunicationContext, human_mode: bool=False, **kwargs):
        self.communication: CommunicationContext = communication
        self.human_mode: bool = human_mode
        self.player_id: str = None
        self.thread_id = "session"
        self.recursion_limit = kwargs.get('recursion_limit')

        # Graphs
        self.graph = get_decentralized_mas(communication=communication, unit_agent_model=kwargs.get('unit_agent_model'), region_agent_model=kwargs.get('region_agent_model'))
        
    def initialize_graph(self,
        player_id: str, players: list[str],
        regions: list[str], neighbouring_regions: dict[str, list[str]],
        final_round: int
    ):
        self.player_id = player_id
        
        init_msg = SystemMessage(content=simulation_start_prompt(
                player_id=player_id,
                players=players,
                regions=regions,
                neighbouring_regions=neighbouring_regions,
                final_round=final_round
        ))
        
        self.graph.update_state(
            values={
                "units": [],
                "regions": regions,
                "recent_events": [],
                "upkeep_phase": False,
                "supply_surplus": 0,
                "player_id": player_id,
                "init_msg": init_msg,
                "blocked_tools": [],
                "unit_orders": {},
                "region_recruitment": {},
                "agents_messages": {},
                "agent_types": {},
                "active_agents": [],
                "waiting_agents": [],
                "finished_agents": []
            },
            config={"configurable": {"thread_id": self.thread_id}}
        )

    def invoke_orders(self, 
            current_round: int, final_round: int,
            players: list[str], leaderboard: dict[str, int], 
            units: list[str], regions: list[str],
            tresspasers: dict[str, list[str]], events: list[str]
        ) -> list[Order]:
        
        # Preparing data for graph
        round_msg = SystemMessage(content=round_start_prompt(
            current_round=current_round,
            final_round=final_round,
            players=players,
            leaderboard_dict=leaderboard,
            controlled_units=units,
            controlled_regions=regions,
            enemy_units_per_region=tresspasers,
            events=events
        ))
        
        # Invoking graph
        output_state = self.graph.invoke(
            {
                "input_prompt": round_msg,
                "units": units,
                "regions": regions,
                "recent_events": events,
                "upkeep_phase": False
            },
            {
                "configurable": {
                    "thread_id": f"{self.thread_id}",
                    "human_mode": self.human_mode
                },
                "recursion_limit": self.recursion_limit
            }
        )
        
        # Retrieving the recruitment choices from the output state
        return [order.model_dump() for order in output_state["unit_orders"].values() if order]

    def invoke_upkeep(self, supply_surplus: int, regions: list[str]) -> list[RecruitmentChoice]:
        # Preparing data for graph
        upkeep_msg = SystemMessage(content=units_recruitment_prompt(
            supply_surplus=supply_surplus,
            regions=regions
        ))
        
        # Invoking graph
        output_state = self.graph.invoke(
            {
                "input_prompt": upkeep_msg,
                "supply_surplus": supply_surplus,
                "regions": regions,
                "upkeep_phase": True
            },
            {
                "configurable": {
                    "thread_id": f"{self.thread_id}",
                    "human_mode": self.human_mode
                },
                "recursion_limit": self.recursion_limit
            }
        )
        
        # Retrieving the recruitment choices from the output state
        return [recruitment.model_dump() for recruitment in output_state["region_recruitment"].values() if recruitment]