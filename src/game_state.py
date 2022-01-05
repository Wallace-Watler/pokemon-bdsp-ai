"""Represents the state of Pokemon battles.

It is possible for many properties of the state to be unknown. These properties are often modeled
as sets of possibilities - a superposition of sorts - which (hopefully) works nicely with the
neural net that plays the game as most of the inputs are categorical anyways. Multiple
possibilities can simply be entered as multiple true bits in the binary vector. Boolean inputs with
uncertainty are stored as reals, where 0.0, 1.0, and 0.5 are false, true, and unknown,
respectively.
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Set
from enum import Enum, auto
from dataclasses import dataclass

# TODO: Add function to flatten State into a NN input vector (list of floats)
@dataclass
class State:
    allies: List[Pokemon]
    opponents: List[Pokemon]
    weather: Set[Weather]
    weather_counter: Counter
    terrain: Set[Terrain]


@dataclass
class Pokemon:
    species: Set[Species]
    first_type: Set[Type]
    second_type: Set[Type]
    min_level: int
    max_level: int
    gender: Set[Gender]
    nature: Set[Nature]
    ability: Set[Ability]
    min_ivs: Tuple[int, int, int, int, int, int]
    max_ivs: Tuple[int, int, int, int, int, int]
    min_evs: Tuple[int, int, int, int, int, int]
    max_evs: Tuple[int, int, int, int, int, int]
    min_hp_frac: float
    max_hp_frac: float
    min_stat_stages: Tuple[int, int, int, int, int, int, int]
    max_stat_stages: Tuple[int, int, int, int, int, int, int]
    
    major_status_ailment: Set[MajorStatusAilment]
    msa_counter: Counter
    #snore_sleep_talk_counter: Counter
    
    confusion_counter: Counter
    is_seeded: float
    is_infatuated: float
    is_cursed: float
    has_nightmare: float
    
    on_field: bool
    moves: List[MoveInstance]
    move_lock_in: Counter


@dataclass
class MoveInstance:
    move: Set[Move]
    min_pp: int
    max_pp: int
    disabled: float


@dataclass
class Counter:
    min_value: int
    max_value: int
    min_target: int
    max_target: int


class Type(Enum):
    BUG = 0
    DARK = auto()
    DRAGON = auto()
    ELECTRIC = auto()
    FAIRY = auto()
    FIGHTING = auto()
    FIRE = auto()
    FLYING = auto()
    GHOST = auto()
    GRASS = auto()
    GROUND = auto()
    ICE = auto()
    NONE = auto()
    NORMAL = auto()
    POISON = auto()
    PSYCHIC = auto()
    ROCK = auto()
    STEEL = auto()
    WATER = auto()


class Terrain(Enum):
    ELECTRIC = 0
    GRASSY = auto()
    MISTY = auto()
    NORMAL = auto()
    PSYCHIC = auto()


class Weather(Enum):
    EXTREMELY_HARSH_SUNSHINE = 0
    FOG = auto()
    HAIL = auto()
    HARSH_SUNSHINE = auto()
    HEAVY_RAIN = auto()
    NONE = auto()
    RAIN = auto()
    SANDSTORM = auto()
    STRONG_WINDS = auto()


class Ability(Enum):
    CHLOROPHYLL = 0
    OVERGROW = auto()


class Gender(Enum):
    FEMALE = 0
    MALE = auto()
    NONE = auto()


class MajorStatusAilment(Enum):
    ASLEEP = 0
    BADLY_POISONED = auto()
    BURNED = auto()
    FROZEN = auto()
    OKAY = auto()
    PARALYZED = auto()
    POISONED = auto()


class Nature(Enum):
    def __new__(cls, *args, **kwds):
        value = len(cls.__members__)
        obj = object.__new__(cls)
        obj._value_ = value
        return obj
    
    def __init__(self, stat_mods: List[float]) -> None:
        self.stat_mods = stat_mods
    
    def stat_mod(self, stat: Stat) -> float:
        return self.stat_mods[stat.index - 1]
    
    ADAMANT = [1.1, 1.0, 0.9, 1.0, 1.0]
    BASHFUL = [1.0, 1.0, 1.0, 1.0, 1.0]
    BOLD    = [0.9, 1.1, 1.0, 1.0, 1.0]
    BRAVE   = [1.1, 1.0, 1.0, 1.0, 0.9]
    CALM    = [0.9, 1.0, 1.0, 1.1, 1.0]
    CAREFUL = [1.0, 1.0, 0.9, 1.1, 1.0]
    DOCILE  = [1.0, 1.0, 1.0, 1.0, 1.0]
    GENTLE  = [1.0, 0.9, 1.0, 1.1, 1.0]
    HARDY   = [1.0, 1.0, 1.0, 1.0, 1.0]
    HASTY   = [1.0, 0.9, 1.0, 1.0, 1.1]
    IMPISH  = [1.0, 1.1, 0.9, 1.0, 1.0]
    JOLLY   = [1.0, 1.0, 0.9, 1.0, 1.1]
    LAX     = [1.0, 1.1, 1.0, 0.9, 1.0]
    LONELY  = [1.1, 0.9, 1.0, 1.0, 1.0]
    MILD    = [1.0, 0.9, 1.1, 1.0, 1.0]
    MODEST  = [0.9, 1.0, 1.1, 1.0, 1.0]
    NAIVE   = [1.0, 1.0, 1.0, 0.9, 1.1]
    NAUGHTY = [1.1, 1.0, 1.0, 0.9, 1.0]
    QUIET   = [1.0, 1.0, 1.1, 1.0, 0.9]
    QUIRKY  = [1.0, 1.0, 1.0, 1.0, 1.0]
    RASH    = [1.0, 1.0, 1.1, 0.9, 1.0]
    RELAXED = [1.0, 1.1, 1.0, 1.0, 0.9]
    SASSY   = [1.0, 1.0, 1.0, 1.1, 0.9]
    SERIOUS = [1.0, 1.0, 1.0, 1.0, 1.0]
    TIMID   = [0.9, 1.0, 1.0, 1.0, 1.1]


class Stat(Enum):
    def __new__(cls, *args, **kwds):
        value = len(cls.__members__)
        obj = object.__new__(cls)
        obj._value_ = value
        return obj
    
    def __init__(self, index: int, display_name: str) -> None:
        self.index = index
        self.display_name = display_name
    
    HP = 0, "HP"
    ATK = 1, "attack"
    DEF = 2, "defense"
    SP_ATK = 3, "special attack"
    SP_DEF = 4, "special defense"
    SPD = 5, "speed"
    ACC = 6, "accuracy"
    EVA = 7, "evasion"


class Species(Enum):
    BULBASAUR = 0
    # TODO: Add rest of species


class Move(Enum):
    TACKLE = 0
    # TODO: Add rest of moves
