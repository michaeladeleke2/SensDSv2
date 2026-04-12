# =================================================================================================
#  Copyright (c) Innovation First 2025. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# =================================================================================================
# NOTE: Inner class names renamed (e.g. DriveVelocityUnits_ instead of DriveVelocityUnits)
# to avoid Python name-shadowing when loaded as a package submodule.
# All public API remains identical.
# =================================================================================================
""" 
AIM WebSocket API - Types
"""
from enum import Enum
from typing import Union
import time

class vexEnum:
    '''Base class for all enumerated types'''
    value = 0
    name = ""

    def __init__(self, value, name):
        self.value = value
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __hash__(self):
        return self.value

class SoundType(str, Enum):
    DOORBELL       = "DOORBELL"
    TADA           = "TADA"
    FAIL           = "FAIL"
    SPARKLE        = "SPARKLE"
    FLOURISH       = "FLOURISH"
    FORWARD        = "FORWARD"
    REVERSE        = "REVERSE"
    RIGHT          = "RIGHT"
    LEFT           = "LEFT"
    BLINKER        = "BLINKER"
    CRASH          = "CRASH"
    BRAKES         = "BRAKES"
    HUAH           = "HUAH"
    PICKUP         = "PICKUP"
    CHEER          = "CHEER"
    SENSING        = "SENSING"
    DETECTED       = "DETECTED"
    OBSTACLE       = "OBSTACLE"
    LOOPING        = "LOOPING"
    COMPLETE       = "COMPLETE"
    PAUSE          = "PAUSE"
    RESUME         = "RESUME"
    SEND           = "SEND"
    RECEIVE        = "RECEIVE"
    ACT_HAPPY      = "ACT_HAPPY"
    ACT_SAD        = "ACT_SAD"
    ACT_EXCITED    = "ACT_EXCITED"
    ACT_ANGRY      = "ACT_ANGRY"
    ACT_SILLY      = "ACT_SILLY"

class FontType(str, Enum):
    MONO20 = "MONO20"
    MONO24 = "MONO24"
    MONO30 = "MONO30"
    MONO36 = "MONO36"
    MONO40 = "MONO40"
    MONO60 = "MONO60"
    PROP20 = "PROP20"
    PROP24 = "PROP24"
    PROP30 = "PROP30"
    PROP36 = "PROP36"
    PROP40 = "PROP40"
    PROP60 = "PROP60"
    MONO15 = "MONO15"
    MONO12 = "MONO12"

class KickType(str, Enum):
   SOFT          = "kick_soft"
   MEDIUM        = "kick_medium"
   HARD          = "kick_hard"

class AxisType(Enum):
    X_AXIS        = 0
    Y_AXIS        = 1
    Z_AXIS        = 2

class TurnType(Enum):
    LEFT        = 0
    RIGHT       = 1

class OrientationType:
    ROLL        = 0
    PITCH       = 1
    YAW         = 2

class AccelerationType:
    FORWARD     = 0
    RIGHTWARD   = 1
    DOWNWARD    = 2

class PercentUnits:
    '''The measurement units for percentage values.'''
    class PercentUnits_(vexEnum):   # renamed: was PercentUnits
        pass
    PERCENT = PercentUnits_(0, "PERCENT")
    '''A percentage unit that represents a value from 0% to 100%'''
    # Keep inner name accessible for Union type hints in aim.py
    PercentUnits = PercentUnits_

class RotationUnits:
    '''The measurement units for rotation values.'''
    class RotationUnits_(vexEnum):  # renamed: was RotationUnits
        pass
    DEG = RotationUnits_(0, "DEG")
    REV = RotationUnits_(1, "REV")
    RAW = RotationUnits_(99, "RAW")
    RotationUnits = RotationUnits_

class DriveVelocityUnits:
    '''The measurement units for drive velocity values.'''
    class DriveVelocityUnits_(vexEnum):  # renamed: was DriveVelocityUnits
        pass
    PERCENT = DriveVelocityUnits_(0, "PCT")
    MMPS    = DriveVelocityUnits_(1, "MMPS")
    DriveVelocityUnits = DriveVelocityUnits_

class TurnVelocityUnits:
    '''The measurement units for turn velocity values.'''
    class TurnVelocityUnits_(vexEnum):  # renamed: was TurnVelocityUnits
        pass
    PERCENT = TurnVelocityUnits_(0, "PCT")
    DPS     = TurnVelocityUnits_(1, "DPS")
    TurnVelocityUnits = TurnVelocityUnits_

class TimeUnits:
    '''The measurement units for time values.'''
    class TimeUnits_(vexEnum):  # renamed: was TimeUnits
        pass
    SECONDS = TimeUnits_(0, "SECONDS")
    MSEC    = TimeUnits_(1, "MSEC")
    TimeUnits = TimeUnits_

class DistanceUnits:
    '''The measurement units for distance values.'''
    class DistanceUnits_(vexEnum):  # renamed: was DistanceUnits
        pass
    MM = DistanceUnits_(0, "MM")
    IN = DistanceUnits_(1, "IN")
    CM = DistanceUnits_(2, "CM")
    DistanceUnits = DistanceUnits_

class VoltageUnits:
    '''The measurement units for voltage values.'''
    class VoltageUnits_(vexEnum):  # renamed: was VoltageUnits
        pass
    VOLT = VoltageUnits_(0, "VOLT")
    MV   = VoltageUnits_(0, "mV")
    VoltageUnits = VoltageUnits_

# ----------------------------------------------------------
# globals
# ----------------------------------------------------------
PERCENT = PercentUnits.PERCENT
LEFT    = TurnType.LEFT
RIGHT   = TurnType.RIGHT
DEGREES = RotationUnits.DEG
TURNS   = RotationUnits.REV
SECONDS = TimeUnits.SECONDS
MSEC    = TimeUnits.MSEC
INCHES  = DistanceUnits.IN
MM      = DistanceUnits.MM
VOLT    = VoltageUnits.VOLT
MV      = VoltageUnits.MV
MMPS    = DriveVelocityUnits.MMPS
DPS     = TurnVelocityUnits.DPS
OFF     = False

vexnumber = Union[int, float]
DriveVelocityPercentUnits = Union[DriveVelocityUnits.DriveVelocityUnits_, PercentUnits.PercentUnits_]
TurnVelocityPercentUnits  = Union[TurnVelocityUnits.TurnVelocityUnits_,  PercentUnits.PercentUnits_]

class LightType(str, Enum):
   LED1      = "light1"
   LED2      = "light2"
   LED3      = "light3"
   LED4      = "light4"
   LED5      = "light5"
   LED6      = "light6"
   ALL_LEDS  = "all"

class Color:
    class DefinedColor:
        def __init__(self, value, transparent=False):
            self.value = value
            self.transparent = transparent

    BLACK       = DefinedColor(0x000000)
    WHITE       = DefinedColor(0xFFFFFF)
    RED         = DefinedColor(0xFF0000)
    GREEN       = DefinedColor(0x00FF00)
    BLUE        = DefinedColor(0x001871)
    YELLOW      = DefinedColor(0xFFFF00)
    ORANGE      = DefinedColor(0xFF8500)
    PURPLE      = DefinedColor(0xFF00FF)
    CYAN        = DefinedColor(0x00FFFF)
    TRANSPARENT = DefinedColor(0x000000, True)

    def __init__(self, *args):
        self.transparent = False
        if len(args) == 1 and isinstance(args[0], int):
            self.value: int = args[0]
        elif len(args) == 3 and all(isinstance(arg, int) for arg in args):
            self.value = ((args[0] & 0xFF) << 16) + ((args[1] & 0xFF) << 8) + (args[2] & 0xFF)
        else:
            raise TypeError("bad parameters")

    def set_rgb(self, *args):
        if len(args) == 1 and isinstance(args[0], int):
            self.value = args[0]
        if len(args) == 3 and all(isinstance(arg, int) for arg in args):
            self.value = ((args[0] & 0xFF) << 16) + ((args[1] & 0xFF) << 8) + (args[2] & 0xFF)


def sleep(duration: vexnumber, units=TimeUnits.MSEC):
    if units == TimeUnits.MSEC:
        time.sleep(duration / 1000)
    else:
        time.sleep(duration)

def wait(duration: vexnumber, units=TimeUnits.MSEC):
    if units == TimeUnits.MSEC:
        time.sleep(duration / 1000)
    else:
        time.sleep(duration)

class EmojiType:
    class EmojiType_(vexEnum):  # renamed: was EmojiType
        pass
    EXCITED      = EmojiType_( 0, "EXCITED")
    CONFIDENT    = EmojiType_( 1, "CONFIDENT")
    SILLY        = EmojiType_( 2, "SILLY")
    AMAZED       = EmojiType_( 3, "AMAZED")
    STRONG       = EmojiType_( 4, "STRONG")
    THRILLED     = EmojiType_( 5, "THRILLED")
    HAPPY        = EmojiType_( 6, "HAPPY")
    PROUD        = EmojiType_( 7, "PROUD")
    LAUGHING     = EmojiType_( 8, "LAUGHING")
    OPTIMISTIC   = EmojiType_( 9, "OPTIMISTIC")
    DETERMINED   = EmojiType_(10, "DETERMINED")
    AFFECTIONATE = EmojiType_(11, "AFFECTIONATE")
    CALM         = EmojiType_(12, "CALM")
    QUIET        = EmojiType_(13, "QUIET")
    SHY          = EmojiType_(14, "SHY")
    CHEERFUL     = EmojiType_(15, "CHEERFUL")
    LOVED        = EmojiType_(16, "LOVED")
    SURPRISED    = EmojiType_(17, "SURPRISED")
    THINKING     = EmojiType_(18, "THINKING")
    TIRED        = EmojiType_(19, "TIRED")
    CONFUSED     = EmojiType_(20, "CONFUSED")
    BORED        = EmojiType_(21, "BORED")
    EMBARRASSED  = EmojiType_(22, "EMBARRASSED")
    WORRIED      = EmojiType_(23, "WORRIED")
    SAD          = EmojiType_(24, "SAD")
    SICK         = EmojiType_(25, "SICK")
    DISAPPOINTED = EmojiType_(26, "DISAPPOINTED")
    NERVOUS      = EmojiType_(27, "NERVOUS")
    ANNOYED      = EmojiType_(30, "ANNOYED")
    STRESSED     = EmojiType_(31, "STRESSED")
    ANGRY        = EmojiType_(32, "ANGRY")
    FRUSTRATED   = EmojiType_(33, "FRUSTRATED")
    JEALOUS      = EmojiType_(34, "JEALOUS")
    SHOCKED      = EmojiType_(35, "SHOCKED")
    FEAR         = EmojiType_(36, "FEAR")
    DISGUST      = EmojiType_(37, "DISGUST")
    EmojiType    = EmojiType_

Emoji = EmojiType

class EmojiLookType:
    class EmojiLookType_(vexEnum):  # renamed: was EmojiLookType
        pass
    LOOK_FORWARD = EmojiLookType_( 0, "LOOK_FORWARD")
    LOOK_RIGHT   = EmojiLookType_( 1, "LOOK_RIGHT")
    LOOK_LEFT    = EmojiLookType_( 2, "LOOK_LEFT")
    EmojiLookType = EmojiLookType_

EmojiLook = EmojiLookType

class StackingType(Enum):
   STACKING_OFF           = 0
   STACKING_MOVE_RELATIVE = 1
   STACKING_MOVE_GLOBAL   = 2

class SensitivityType:
    class SensitivityType_(vexEnum):  # renamed: was SensitivityType
        pass
    LOW    = SensitivityType_( 0, "LOW")
    MEDIUM = SensitivityType_( 1, "MEDIUM")
    HIGH   = SensitivityType_( 2, "HIGH")
    SensitivityType = SensitivityType_