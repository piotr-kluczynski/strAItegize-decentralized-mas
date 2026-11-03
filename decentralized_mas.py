import random

from strAItegize_common.client.prompts import round_start_prompt, simulation_start_prompt, units_recruitment_prompt
from strAItegize_common.others.order import Order
from strAItegize_common.others.recruitment_choice import RecruitmentChoice

from unit_agent import get_unit_agent
from states import UnitInputState, UnitOutputState
from tools import UnitAgentMessage
from prompts import unit_agent_prompt, unit_agent_msg_prompt, msg_no_response_prompt

from strAItegize_common.client.communication.communication_context import CommunicationContext
from strAItegize_common.client.multiagent_system import MultiAgentSystem

from langchain_core.messages import SystemMessage

class DecentralizedMAS(MultiAgentSystem):
    def __init__(self, communication: CommunicationContext, human_mode: bool=False, **kwargs):
        self.communication: CommunicationContext = communication
        self.human_mode: bool = human_mode
        self.player_id: str = None

        # Graph
        self.thread_id_counter = 1
        self.agents_thread_ids: dict[str, int] = {}

        self.unit_agents = {} # unit_agent_id: Graph
        self.unit_agents_messages = {} # unit_agent_id: List[BaseMessage]

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
        self._manage_unit_agents(units=units)

        # Crafting a round start message for each unit
        unit_orders = {unit: None for unit in units}

        # Operation loop - running until all agent activity finishes
        active_agents = [unit_agent_id for unit_agent_id in self.unit_agents.keys()]
        dormant_agents = []
        waiting_agents = []

        while len(active_agents) > 0:
            input_states = {}
            for unit_agent in active_agents:
                input_states[unit_agent] = UnitInputState(
                    units=units,
                    regions=regions,
                    player_id=self.player_id,
                    unit_id=unit_agent,
                    supply_surplus=0,
                    upkeep_phase=False,
                    recent_events=events,
                    unit_messages=self.unit_agents_messages[unit_agent],
                    unit_orders=unit_orders,
                    recruitment_choice_list=[]
                )
                
            # Invoke active agents
            output_states = self._run_unit_agents(input_states=input_states)

            # 1. Update each active agent's history and orders
            for unit_agent, output_state in output_states.items():
                unit_orders[unit_agent] = output_state['unit_orders'][unit_agent]
                self.unit_agents_messages[unit_agent] = output_state['unit_messages']

            # 2. Route messages & determine next round of active agents
            next_active_agents = set()
            for unit_agent, output_state in output_states.items():
                messages = output_state.get('messages_to_send', [])
                if len(messages) > 0:
                    for unit_msg in messages:
                        self._pass_unit_agent_message(unit_msg)
                        
                        # The sendee must run in the next step to process this message
                        sendee = unit_msg.sendee
                        next_active_agents.add(sendee)
                        if sendee in dormant_agents:
                            dormant_agents.remove(sendee)
                        if sendee in waiting_agents:
                            waiting_agents.remove(sendee)

                    # Sender waits for a response
                    waiting_agents.append(unit_agent)
                    if unit_agent in active_agents:
                        active_agents.remove(unit_agent)
                else:

                    # Agent finished its turn without sending messages
                    if unit_agent in active_agents:
                        active_agents.remove(unit_agent)
                    dormant_agents.append(unit_agent)

            # Re-activate any agents that were messaged (removing them from dormant if they were added above)
            for sendee in next_active_agents:
                if sendee in dormant_agents:
                    dormant_agents.remove(sendee)
                if sendee not in active_agents:
                    active_agents.append(sendee)

            # Awakening waiting agents only when all communication has resolved
            if len(active_agents) == 0 and len(waiting_agents) > 0:
                for unit_agent in waiting_agents.copy():
                    self.unit_agents_messages[unit_agent].append(
                        SystemMessage(content=msg_no_response_prompt())
                    )
                    active_agents.append(unit_agent)
                    waiting_agents.remove(unit_agent)

        # Return orders for units
        orders_to_send = [order.model_dump() for order in unit_orders.values() if order]
        return orders_to_send

    def invoke_upkeep(self, supply_surplus: int, regions: list[str]) -> list[RecruitmentChoice]:
        # Crafting a round start message for each unit
        input_states = {}
        for unit in self.unit_agents.keys():
            self.unit_agents_messages[unit].append(SystemMessage(content=units_recruitment_prompt(
                supply_surplus=supply_surplus,
                regions=regions
            )))

        # Operation loop - running until all agent activity finishes
        recruitment_choices = []

        active_agents = [unit_agent_id for unit_agent_id in self.unit_agents.keys()]
        dormant_agents = []
        waiting_agents = []

        while len(active_agents) > 0:
            # Creating new state for each active agent
            for unit_agent in active_agents:
                input_states[unit_agent] = UnitInputState(
                    units=self.unit_agents.keys(),
                    regions=regions,
                    player_id=self.player_id,
                    unit_id=unit_agent,
                    supply_surplus=supply_surplus,
                    upkeep_phase=True,
                    recent_events=[],
                    unit_messages=self.unit_agents_messages[unit_agent],
                    unit_orders={},
                    recruitment_choice_list=[]
                )

            # Invoke active agents
            output_states = self._run_unit_agents(input_states=input_states)

            for unit_agent, output_state in output_states.items():
                # Saving all unit recruitment orders
                recruitment_choices += output_state['recruitment_choice_list']

                if len(output_state['messages_to_send']) > 0:
                    # Sending all unit agent messages
                    for unit_msg in output_state['messages_to_send']:
                        self._pass_unit_agent_message(unit_msg)

                        # Awakening the sendee if it was dormant
                        if unit_msg['sendee'] in dormant_agents:
                            active_agents.append(unit_msg['sendee'])
                            dormant_agents.remove(unit_msg['sendee'])

                        # Putting sender on the waiting list
                        waiting_agents.append(unit_agent)
                        active_agents.remove(unit_agent)

                else:
                    # Putting finished agent on the dormant list
                    dormant_agents.append(unit_agent)
                    active_agents.remove(unit_agent)


            # Awakening waiting agents, which didn't get response
            for unit_agent in waiting_agents.copy():
                self.unit_agents_messages[unit_agent] = [SystemMessage(content=msg_no_response_prompt())] + self.unit_agents_messages[unit_agent]

                active_agents.append(unit_agent)
                waiting_agents.remove(unit_agent)

        # Return orders for units
        recruitment_choice = random.choices(recruitment_choices) if len(recruitment_choices) > 0 else []
        recruitment_to_send = [order.model_dump() for order in recruitment_choice]
        return recruitment_to_send

    def _pass_unit_agent_message(self, msg: UnitAgentMessage):
        self.unit_agents_messages[msg.sendee].append(SystemMessage(content=unit_agent_msg_prompt(
            sender=msg.sender,
            content=msg.content
        )))

    def _run_unit_agents(self, input_states: dict[str, UnitInputState]) -> dict[str, UnitOutputState]:
        output_states = {}

        for unit, state in input_states.items():
            output_states[unit] = self.unit_agents[unit].invoke(
                state,
                {"configurable": {"thread_id": f"{self.agents_thread_ids[unit]}"}}
            )

        return output_states

    def _manage_unit_agents(self, units: list[str]):
        # Removing missing unit agents
        for unit in self.unit_agents.keys():
            if unit not in units:
                del self.unit_agents[unit]
                del self.unit_agents_messages[unit]
                del self.agents_thread_ids[unit]

        # Adding new unit agents
        for unit in units:
            if unit not in self.unit_agents:
                self.unit_agents[unit] = get_unit_agent(communication=self.communication, human_mode=self.human_mode)
                self.unit_agents_messages[unit] = [self.init_msg]
                self.agents_thread_ids[unit] = self.thread_id_counter

                self.thread_id_counter += 1