from typing import Literal

from strAItegize_common.client.prompts import round_start_prompt, simulation_start_prompt, units_recruitment_prompt
from strAItegize_common.others.order import Order
from strAItegize_common.others.recruitment_choice import RecruitmentChoice

from unit_agent import get_unit_agent
from region_agent import get_region_agent
from states import UnitInputState, RegionInputState
from tools import AgentMessage
from prompts import agent_msg_prompt, msg_no_response_prompt

from strAItegize_common.client.communication.communication_context import CommunicationContext
from strAItegize_common.client.multiagent_system import MultiAgentSystem

from langchain_core.messages import SystemMessage

class DecentralizedMAS(MultiAgentSystem):
    def __init__(self, communication: CommunicationContext, human_mode: bool=False, **kwargs):
        self.communication: CommunicationContext = communication
        self.human_mode: bool = human_mode
        self.player_id: str = None

        # Graphs
        self.thread_id_counter = 1
        self.agents_thread_ids: dict[str, int] = {}

        self.agents = {} # agent_id: Graph
        self.agents_types = {} # agent_id: Literal["unit", "region"]
        self.agents_messages = {} # agent_id: List[BaseMessage]

        self.init_msg = None # Variable to cache the initial message for the simulation start prompt
        
    def initialize_graph(self,
        player_id: str, players: list[str],
        regions: list[str], neighbouring_regions: dict[str, list[str]],
        final_round: int
    ):
        self.player_id = player_id

        self.init_msg = SystemMessage(
            content=simulation_start_prompt(
                player_id=player_id, 
                players=players,
                regions=regions, 
                neighbouring_regions=neighbouring_regions,
                final_round=final_round
            )
        )

    def invoke_orders(self, 
            current_round: int, final_round: int,
            players: list[str], leaderboard: dict[str, int], 
            units: list[str], regions: list[str],
            tresspasers: dict[str, list[str]], events: list[str]
        ) -> list[Order]:
        # Correcting the number of agents corresponding to the units within the simulation
        self._manage_agents(agent_ids=units, agent_type="unit")
        self._manage_agents(agent_ids=regions, agent_type="region")

        # Adding the round start prompt to all agents
        for message_list in self.agents_messages.values():
            message_list.append(SystemMessage(content=round_start_prompt(
                current_round=current_round,
                final_round=final_round,
                players=players,
                leaderboard_dict=leaderboard,
                controlled_units=units,
                controlled_regions=regions,
                enemy_units_per_region=tresspasers,
                events=events
            )))

        # Crafting a round start message for each unit
        unit_orders = {unit: None for unit in units}

        # Operation loop - running until all agent activity finishes
        active_agents = [agent_id for agent_id in self.agents.keys()]
        dormant_agents = []
        waiting_agents = []

        while len(active_agents) > 0:
            input_states = {}
            for agent in active_agents:
                if self.agents_types[agent] == "unit":
                    input_states[agent] = UnitInputState(
                        units=units,
                        regions=regions,
                        player_id=self.player_id,
                        unit_id=agent,
                        supply_surplus=0,
                        upkeep_phase=False,
                        recent_events=events,
                        unit_messages=self.agents_messages[agent],
                        unit_orders=unit_orders,
                        recruitment_choice_list=[]
                    )
                elif self.agents_types[agent] == "region":
                    input_states[agent] = RegionInputState(
                        units=[agent for agent, type in self.agents_types.items() if type == "unit"],
                        regions=regions,
                        player_id=self.player_id,
                        region_id=agent,
                        supply_surplus=0,
                        upkeep_phase=False,
                        recent_events=events,
                        region_messages=self.agents_messages[agent],
                        region_recruitment={}
                    )

            # Invoke active agents
            output_states = self._run_agents(input_states=input_states)

            # 1. Update each active agent's history and orders
            for agent, output_state in output_states.items():
                # Saving unit orders from unit agents
                if self.agents_types[agent] == "unit":
                    unit_orders[agent] = output_state['unit_orders'][agent]
                    self.agents_messages[agent] = output_state['unit_messages']
                elif self.agents_types[agent] == "region":
                    self.agents_messages[agent] = output_state['region_messages']

            # 2. Route messages & determine next round of active agents
            next_active_agents = set()
            for agent, output_state in output_states.items():
                messages = output_state.get('messages_to_send', [])
                if len(messages) > 0:
                    for unit_msg in messages:
                        self._pass_agent_message(unit_msg)
                        
                        # The sendee must run in the next step to process this message
                        sendee = unit_msg.sendee
                        next_active_agents.add(sendee)
                        if sendee in dormant_agents:
                            dormant_agents.remove(sendee)
                        if sendee in waiting_agents:
                            waiting_agents.remove(sendee)

                    # Sender waits for a response
                    waiting_agents.append(agent)
                    if agent in active_agents:
                        active_agents.remove(agent)
                else:

                    # Agent finished its turn without sending messages
                    if agent in active_agents:
                        active_agents.remove(agent)
                    dormant_agents.append(agent)

            # Re-activate any agents that were messaged (removing them from dormant if they were added above)
            for sendee in next_active_agents:
                if sendee in dormant_agents:
                    dormant_agents.remove(sendee)
                if sendee not in active_agents:
                    active_agents.append(sendee)

            # Awakening waiting agents only when all communication has resolved
            if len(active_agents) == 0 and len(waiting_agents) > 0:
                for agent in waiting_agents.copy():
                    self.agents_messages[agent].append(
                        SystemMessage(content=msg_no_response_prompt())
                    )
                    active_agents.append(agent)
                    waiting_agents.remove(agent)

        # Return orders for units
        orders_to_send = [order.model_dump() for order in unit_orders.values() if order]
        return orders_to_send

    def invoke_upkeep(self, supply_surplus: int, regions: list[str]) -> list[RecruitmentChoice]:
        # Correcting the number of agents corresponding to the regions within the simulation
        self._manage_agents(agent_ids=regions, agenttype="region")

        # Adding the upkeep prompt to all agents messages
        for messages_list in self.agents_messages.values():
            messages_list.append(SystemMessage(content=units_recruitment_prompt(
                supply_surplus=supply_surplus,
                regions=regions
            )))

        # Operation loop - running until all agent activity finishes
        region_recruitments = {region: None for region in regions}

        active_agents = [agent_id for agent_id in self.agents.keys()]
        dormant_agents = []
        waiting_agents = []

        while len(active_agents) > 0:
            # Creating new state for each active agent
            input_states = {}

            for agent in active_agents:
                if self.agents_types[agent] == "unit":
                    input_states[agent] = UnitInputState(
                        units=[agent for agent, type in self.agents_types.items() if type == "unit"],
                        regions=regions,
                        player_id=self.player_id,
                        unit_id=agent,
                        supply_surplus=supply_surplus,
                        upkeep_phase=True,
                        recent_events=[],
                        unit_messages=self.agents_messages[agent],
                        unit_orders={},
                        recruitment_choice_list=[]
                    )
                elif self.agents_types[agent] == "region":
                    input_states[agent] = RegionInputState(
                        units=[agent for agent, type in self.agents_types.items() if type == "unit"],
                        regions=regions,
                        player_id=self.player_id,
                        region_id=agent,
                        supply_surplus=supply_surplus,
                        upkeep_phase=True,
                        recent_events=[],
                        region_messages=self.agents_messages[agent],
                        region_recruitment={}
                    )

            # Invoke active agents
            output_states = self._run_agents(input_states=input_states)

            # 1. Update each active agent's history and orders
            for agent, output_state in output_states.items():
                # Saving unit recruitment orders from region agents
                if self.agents_types[agent] == "region":
                    region_recruitments[agent] += output_state['region_recruitment'].get(agent)
                    self.agents_messages[agent] = output_state['region_messages']
                elif self.agents_types[agent] == "unit":
                    self.agents_messages[agent] = output_state['unit_messages']


            # 2. Route messages & determine next round of active agents
            next_active_agents = set()
            for agent, output_state in output_states.items():
                messages = output_state.get('messages_to_send', [])
                if len(messages) > 0:
                    for unit_msg in messages:
                        self._pass_agent_message(unit_msg)
                        
                        # The sendee must run in the next step to process this message
                        sendee = unit_msg.sendee
                        next_active_agents.add(sendee)
                        if sendee in dormant_agents:
                            dormant_agents.remove(sendee)
                        if sendee in waiting_agents:
                            waiting_agents.remove(sendee)

                    # Sender waits for a response
                    waiting_agents.append(agent)
                    if agent in active_agents:
                        active_agents.remove(agent)
                else:

                    # Agent finished its turn without sending messages
                    if agent in active_agents:
                        active_agents.remove(agent)
                    dormant_agents.append(agent)

            # Re-activate any agents that were messaged (removing them from dormant if they were added above)
            for sendee in next_active_agents:
                if sendee in dormant_agents:
                    dormant_agents.remove(sendee)
                if sendee not in active_agents:
                    active_agents.append(sendee)

            # Awakening waiting agents only when all communication has resolved
            if len(active_agents) == 0 and len(waiting_agents) > 0:
                for agent in waiting_agents.copy():
                    self.agents_messages[agent].append(
                        SystemMessage(content=msg_no_response_prompt())
                    )
                    active_agents.append(agent)
                    waiting_agents.remove(agent)

        # Return orders for units
        recruitment_to_send = [recruitment.model_dump() for recruitment in region_recruitments.values() if recruitment]
        return recruitment_to_send

    def _pass_agent_message(self, msg: AgentMessage):
        self.agents_messages[msg.sendee].append(SystemMessage(content=agent_msg_prompt(
            sender=msg.sender,
            content=msg.content
        )))

    def _run_agents(self, input_states: dict) -> dict:
        output_states = {}

        for unit, state in input_states.items():
            output_states[unit] = self.agents[unit].invoke(
                state,
                {"configurable": {"thread_id": f"{self.agents_thread_ids[unit]}"}}
            )

        return output_states

    def _manage_agents(self, agent_ids: list[str], agent_type: Literal["unit", "region"]):
    # 1. Remove missing agents of this type
        for agent in list(self.agents.keys()):
            if self.agents_types.get(agent) == agent_type and agent not in agent_ids:
                del self.agents[agent]
                del self.agents_messages[agent]
                del self.agents_thread_ids[agent]
                del self.agents_types[agent]

        # 2. Add new agents of this type
        for agent in agent_ids:
            if agent not in self.agents:
                if agent_type == "unit":
                    self.agents[agent] = get_unit_agent(communication=self.communication, human_mode=self.human_mode)
                elif agent_type == "region":
                    self.agents[agent] = get_region_agent(communication=self.communication, human_mode=self.human_mode)

                self.agents_types[agent] = agent_type
                self.agents_messages[agent] = [self.init_msg]
                self.agents_thread_ids[agent] = self.thread_id_counter
                self.thread_id_counter += 1