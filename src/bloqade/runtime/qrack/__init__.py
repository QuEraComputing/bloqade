from .base import Memory as Memory, PyQrackInterpreter as PyQrackInterpreter

# NOTE: The following import is for registering the method tables
from .qasm2 import uop as uop, core as core, parallel as parallel
from .target import PyQrack as PyQrack
