"""Tool for modeling finite state machines.

Classes
-------
State
    Representation of a state.

Transition
    Representation of a transition.

FSMListener
    An interface defining events in a FiniteStateMachine.

FiniteStateMachine
    A class representing a non-deterministic finite state machine.

Functions
---------
load_states(path: str) -> List[State]
    Load states from CSV.
"""

from __future__ import annotations
from typing import List, Any, Optional
from dataclasses import dataclass
import csv

@dataclass
class Transition:
    """Representation of a transition.
    
    Attributes
    ----------
    from_state: int
        ID of the state that the transition leads from.
    
    to_state: int
        ID of the state that the transition leads to.
    
    action: Any
        Action associated with the transition. If the action is None, the transition is treated
        as an epsilon transition.
    """
    from_state: int
    to_state: int
    action: Any = None

@dataclass
class State:
    """Representation of a state.
    
    Attributes
    ----------
    name: str
        Name of the state.
    
    incoming: List[Transition]
        Incoming transitions.
    
    outgoing: List[Transition]
        Outgoing transitions.
    """
    name: str
    incoming: List[Transition]
    outgoing: List[Transition]


class FSMListener:
    """An interface defining events in a FiniteStateMachine.
    
    Classes implementing FSMListener may be registered to a FiniteStateMachine to receive its
    events.
    
    Methods
    -------
    onStateEntered(state: int) -> None
        Execute something when the state with the given ID is entered.
    
    onStateExited(state: int) -> None
        Execute something when the state with the given ID is exited.
    
    onTransition(self, transition: Transition) -> None
        Execute something when the given transition occurs.
    """
    
    def onStateEntered(self, state: int) -> None:
        """Execute something when the given state is entered."""
        pass
    
    def onStateExited(self, state: int) -> None:
        """Execute something when the given state is exited."""
        pass
    
    def onTransition(self, transition: Transition) -> None:
        """Execute something when the given transition occurs."""
        pass


class FiniteStateMachine:
    """A representation of a non-deterministic finite state machine.
    
    FiniteStateMachine is capable of representing any non-deterministic finite state machine with
    the following caveat:
        - Given any two states A and B with more than one transition leading from A to B, one of
          the transitions will be arbitrarily chosen when transition_to(B) is called and A is the
          current state.
    
    Attributes
    -------
    states: List[State]
        All the states contained in the finite state machine.
    
    transitions: List[Transitions]
        All the transitions contained in the finite state machine.
    
    current_state: int
        ID of the current state.
    
    listeners: List[FSMListener]
        Registered event listeners.
    
    Methods
    -------
    register_listener(listener: FSMListener) -> None
        Add an event listener to this finite state machine.
    
    name(state: int) -> str
        Return the name of the state with the given ID.
    
    find_state(name: str) -> int
        Find a state by its name.
    
    simulate_action(self, action: Any) -> List[int]
        Simulate the effect of performing an action.
    
    find_transition(a: int, b: int) -> Optional[Transition]
        Find a transition between states A and B.
    
    transition_to(state: int) -> None
        Transition to the given state.
    
    bfs(end: int, start: int = -1) -> List[int]
        Breadth-first search on the finite state machine.
    
    Class Methods
    -------------
    from_csv(folder: str) -> FiniteStateMachine
        Load a finite state machine from CSV.
    """
    
    def from_csv(folder: str) -> FiniteStateMachine:
        """Load a finite state machine from CSV.
        
        Arguments
        ---------
        folder: str
            A path to a folder containing two files:
              - states.csv
                Each row contains the name of the state.
            
              - transitions.csv
                Each row contains the action, name of the state led from, and name of the state led
                to. Rows with empty action strings are interpreted as epsilon transitions.
        
        Returns
        -------
        The FiniteStateMachine loaded from CSV.
        """
        states = load_states(folder + "/states.csv")
        state_names = [state.name for state in states]
        
        transitions = []
        with open(folder + "/transitions.csv") as transitions_file:
            for row in csv.reader(transitions_file):
                from_state = state_names.index(row[1])
                to_state = state_names.index(row[2])
                transition = Transition(from_state, to_state, None if row[0] == "" else row[0])
                transitions.append(transition)
                states[from_state].outgoing.append(transition)
                states[to_state].incoming.append(transition)
        
        return FiniteStateMachine(0, states, transitions)
    
    def __init__(self, initial_state: int, states: List[State], transitions: List[Transition]) -> None:
        """Construct a FiniteStateMachine.
        
        Arguments
        ---------
        initial_state: int
            ID of the initial state.
        
        states: List[State]
            All the states contained in the finite state machine.
        
        transitions: List[Transition]
            All the transitions contained in the finite state machine.
        """
        self.states = states
        self.transitions = transitions
        self.current_state = initial_state
        self.listeners = []
    
    def register_listener(self, listener: FSMListener) -> None:
        """Add an event listener to this finite state machine."""
        self.listeners.append(listener)
    
    def name(state: int) -> str:
        """Return the name of the state with the given ID."""
        return self.states[state].name
    
    def find_state(self, name: str) -> int:
        """Find a state by its name.
        
        Arguments
        ---------
        name: str
            Name of the state to find.
        
        Returns
        -------
        The ID of the state with the given name, or -1 if such a state does not exist.
        """
        for i in range(0, len(self.states)):
            if self.states[i].name == name:
                return i
        
        return -1
    
    def find_transition(self, a: int, b: int) -> Optional[Transition]:
        """Find a transition between states A and B.
        
        Arguments
        ---------
        a: int
            ID of the state A.
        
        b: int
            ID of the state B.
        
        Returns
        -------
        The transition leading from A to B, or None if such a transition does not exist. All state
        machines implicitly include reflexive epsilon transitions, but those are not modeled here.
        They may still be explicitly defined, however, and would therefore be returned by this
        method when appropriate.
        """
        for transition in self.transitions:
            if transition.from_state == a and transition.to_state == b:
                return transition
        
        return None
    
    def transition_to(self, state: int) -> None:
        """Transition to the given state.
        
        Performs the transition from the current state to the given state and calls FSMListeners.
        Note that if there exists multiple transitions from the current state to the given state,
        a transition will be arbitrarily chosen.
        
        Arguments
        ---------
        state: int
            ID of the state to transition to.
        """
        transition = self.find_transition(self.current_state, state)
        if not transition == None:
            for listener in self.listeners:
                listener.onStateExited(self.current_state)
            
            self.current_state = state
            
            for listener in self.listeners:
                listener.onStateEntered(self.current_state)
                listener.onTransition(transition)
        else:
            raise ValueError(f"No transition defined between {self.states[self.current_state].name} and {self.states[state].name}")
    
    def simulate_action(self, action: Any) -> List[int]:
        """Simulate the effect of performing an action.
        
        Note that this does not affect the current state of the machine. Since an action may result
        in several states, it is unknown which state would become the current without more
        information.
        
        Arguments
        ---------
        action: Any
            An action to perform.
        
        Returns
        -------
        The IDs of the states that can result from performing the action in the current state.
        """
        visited = [False for i in range(0, len(self.states))]
        return self.__perform_action(action, self.current_state, visited)
    
    def __perform_action(self, action: Any, state: int, visited: List[bool]) -> List[int]:
        possible_states = []
        visited[state] = True
        for transition in self.states[state].outgoing:
            if transition.action == action and not visited[transition.to_state]:
                possible_states.append(transition.to_state)
                possible_states.extend(self.__perform_action(None, transition.to_state, visited))
        
        return possible_states
    
    def bfs(self, end: int, start: int = -1) -> List[int]:
        """Breadth-first search on the finite state machine.
        
        Arguments
        ---------
        end: int
            ID of the state to end search at.
        
        start: int = None
            ID of the state to start the search at. If negative, search will start at the current
            state.
        
        Returns
        -------
        A list of state IDs representing the shortest path from start to end. The list will be
        empty if no such path exists.
        """
        q = [start]
        visited = [False for i in range(0, len(self.states))]
        visited[start] = True
        prev = [None for i in range(0, len(self.states))]
        while len(q) > 0:
            current = q.pop(0)
            neighbors = [transition.to_state for transition in self.states[current].outgoing]
            for next in neighbors:
                if not visited[next]:
                    q.append(next)
                    visited[next] = True
                    prev[next] = current
        
        path = []
        at = end
        while not at == None:
            path.append(at)
            at = prev[at]
        path.reverse()
        return path if path[0] == start else []


def load_states(path: str) -> List[State]:
    """Load states from CSV.
    
    Arguments
    ---------
    path: str
        A path to a CSV file defining the states. Each row contains the name of a state.
    
    Returns
    -------
    A list of the loaded states.
    """
    states = []
    with open(path) as states_file:
        for row in csv.reader(states_file):
            states.append(State(row[0], [], []))
    return states
