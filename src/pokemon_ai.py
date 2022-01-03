import sys

if not len(sys.argv) == 3:
    print("Usage: python pokemon_ai.py <ui_folder> <scale>")
    exit(0)

import time
import tensorflow as tf
import numpy as np
import finite_state_machine as fsm

DT = 1.0 / 30

SCALE = int(sys.argv[2])
UI_FOLDER = sys.argv[1]
SCALE_FOLDER = f"{UI_FOLDER}/training_data/scale{SCALE}"
MODEL_FOLDER = f"{SCALE_FOLDER}/trained_model"
UI_FSM = fsm.from_csv("../resources/bdsp_bt")
SENSORY_CORTEX = SensoryCortex(MODEL_FOLDER)
MOTOR_CORTEX = MotorCortex()

# TODO: Make this loop less jank
SENSORY_CORTEX.tick(DT)
while not UI_FSM.current_state().name == "end":
    MOTOR_CORTEX.tick(DT)
    time.sleep(DT)
    SENSORY_CORTEX.tick(DT)


class SensoryCortex(FSMListener):
    def __init__(self, model_folder: str) -> None:
        self.ui_net = tf.keras.models.load_model(model_folder)
        # Adding a softmax layer to convert logits to probabilities
        self.ui_net = tf.keras.Sequential([self.ui_net, tf.keras.layers.Softmax()])
        
        self.processing_info = False
        
        UI_FSM.registerListener(self)
    
    def observe_state(self) -> None:
        # TODO: Get frame from somewhere
        frame = None
        probabilities = self.ui_net.predict(frame)[0]
        observed_state = np.argmax(probabilities)
        if probabilities[observed_state] >= 0.95:
            path = UI_FSM.bfs(observed_state)
            if len(path) >= 2:
                for state in path[1:]:
                    UI_FSM.transition_to(state)
            elif not path:
                raise RuntimeError(f"Observed state {observe_state} ('{UI_FSM.name(observed_state)}') is not accessible from state {UI_FSM.current_state} ('{UI_FSM.name(UI_FSM.current_state)}')")
    
    def onStateEntered(self, state: int) -> None:
        if UI_FSM.name(state) == "info":
            self.processing_info = True
        elif "Member" in UI_FSM.name(state):
            self.process_members()
    
    def onStateExited(self, state: int) -> None:
        if UI_FSM.name(state) == "info":
            self.processing_info = False
    
    def tick(self, dt: float) -> None:
        self.observe_state()
        if self.processing_info:
            self.processing_info = self.process_info()
    
    def process_info(self) -> bool:
        # TODO: Attempt to extract text from current frame
        # TODO: Return False if info is found
        return True
    
    def process_members(self) -> None:
        # TODO: Extract text for entire team from current frame
        pass
    
    def process_member(self) -> None:
        # TODO: Extract text specific to member from current frame
        pass


# TODO: High-level decision-making
class MotorCortex(FSMListener):
    def __init__(self, timeout: float):
        self.timeout = timeout
        self.wait_time = 0.0
        self.desired_plays = [] # Transitions
        self.waiting_for = [] # State IDs
        UI_FSM.registerListener(self)
    
    def do_next_play(self) -> None:
        if not self.desired_plays:
            # TODO: Query NEAT network for next desired play
            pass
        
        path = UI_FSM.bfs(self.desired_plays[0].from_state)
        if path:
            path.append(self.desired_plays[0].to_state)
            desired_action = UI_FSM.find_transition(path[0], path[1]).action
            if not desired_action == None:
                # TODO: Send desired_action as input to the controller
                pass
            
            self.waiting_for = UI_FSM.simulate_action(desired_action)
            self.wait_time = 0.0
        else:
            self.desired_plays.pop(0)
            self.do_next_play()
    
    def onStateEntered(self, state: int) -> None:
        pass
    
    def onStateExited(self, state: int) -> None:
        if UI_FSM.name(state) == "start":
            # TODO: Add info gathering plays to desired_plays
            pass
    
    def onTransition(self, transition: Transition) -> None:
        if transition == self.desired_plays[0]:
            self.desired_plays.pop(0)
    
    def tick(self, dt: float) -> None:
        if self.waiting_for:
            self.wait_time += dt
            if self.wait_time > self.timeout:
                self.do_next_play()
        else:
            self.wait_time = 0.0
