# =================================================================================================
#  Copyright (c) Innovation First 2025. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# =================================================================================================
"""
AIM WebSocket API - Globals

This module defines global constants and enums used in the AIM WebSocket API. These globals
are provided to improve code readability and ease of use, especially for beginners and users
familiar with the VEXcode API.

Globals in this module are designed to match the VEXcode API documentation. 
If these conflict with existing types or variables in your project,
you may choose not to import this module.
"""

# ----------------------------------------------------------
# pylint: disable=unnecessary-pass,unused-argument,line-too-long,too-many-lines
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring
# pylint: disable=invalid-name,unused-import,redefined-outer-name

from vex import KickType
from vex import AxisType
from vex import OrientationType
from vex import AccelerationType
from vex import LightType
from vex import Color
from vex import FontType
from vex import EmojiType
from vex import EmojiLookType
from vex import SoundType
from vex import VisionObject

SOFT = KickType.SOFT
'''A kick type of soft'''
MEDIUM = KickType.MEDIUM
'''A kick type of medium'''
HARD = KickType.HARD
'''A kick type of hard'''

X_AXIS = AxisType.X_AXIS
'''The X axis of the Inertial sensor.'''
Y_AXIS = AxisType.Y_AXIS
'''The Y axis of the Inertial sensor.'''
Z_AXIS = AxisType.Z_AXIS
'''The Z axis of the Inertial sensor.'''

ROLL = OrientationType.ROLL
'''roll, orientation around the X axis of the Inertial sensor.'''
PITCH = OrientationType.PITCH
'''pitch, orientation around the Y axis of the Inertial sensor.'''
YAW = OrientationType.YAW
'''yaw, orientation around the Z axis of the Inertial sensor.'''

FORWARD = AccelerationType.FORWARD
'''The X acceleration axis of the Inertial sensor.'''
RIGHTWARD = AccelerationType.RIGHTWARD
'''The Y acceleration axis of the Inertial sensor.'''
DOWNWARD = AccelerationType.DOWNWARD
'''The Z acceleration axis of the Inertial sensor.'''

LED1 = LightType.LED1
'''The LED at the 315 degree position on AIM'''
LED2 = LightType.LED2
'''The LED at the 265 degree position on AIM'''
LED3 = LightType.LED3
'''The LED at the 210 degree position on AIM'''
LED4 = LightType.LED4
'''The LED at the 155 degree position on AIM'''
LED5 = LightType.LED5
'''The LED at the 100 degree position on AIM'''
LED6 = LightType.LED6
'''The LED at the 45 degree position on AIM'''
ALL_LEDS = LightType.ALL_LEDS
'''All LEDs on AIM'''

BLACK = Color.BLACK
'''predefined Color black'''
WHITE = Color.WHITE
'''predefined Color white'''
RED = Color.RED
'''predefined Color red'''
GREEN = Color.GREEN
'''predefined Color green'''
BLUE = Color.BLUE
'''predefined Color blue'''
YELLOW = Color.YELLOW
'''predefined Color yellow'''
ORANGE = Color.ORANGE
'''predefined Color orange'''
PURPLE = Color.PURPLE
'''predefined Color purple'''
CYAN = Color.CYAN
'''predefined Color cyan'''
TRANSPARENT = Color.TRANSPARENT
'''predefined Color transparent'''

MONO12 = FontType.MONO12
'''monotype font of size 12'''
MONO15 = FontType.MONO15
'''monotype font of size 15'''
MONO20 = FontType.MONO20
'''monotype font of size 20'''
MONO24 = FontType.MONO24
'''monotype font of size 24'''
MONO30 = FontType.MONO30
'''monotype font of size 30'''
MONO36 = FontType.MONO36
'''monotype font of size 36'''
MONO40 = FontType.MONO40
'''monotype font of size 40'''
MONO60 = FontType.MONO60
'''monotype font of size 60'''
PROP20 = FontType.PROP20
'''proportional font of size 20'''
PROP24 = FontType.PROP24
'''proportional font of size 24'''
PROP30 = FontType.PROP30
'''proportional font of size 30'''
PROP36 = FontType.PROP36
'''proportional font of size 36'''
PROP40 = FontType.PROP40
'''proportional font of size 40'''
PROP60 = FontType.PROP60
'''proportional font of size 60'''

EXCITED = EmojiType.EXCITED
CONFIDENT = EmojiType.CONFIDENT
SILLY = EmojiType.SILLY
AMAZED = EmojiType.AMAZED
STRONG = EmojiType.STRONG
THRILLED = EmojiType.THRILLED
HAPPY = EmojiType.HAPPY
PROUD = EmojiType.PROUD
LAUGHING = EmojiType.LAUGHING
OPTIMISTIC = EmojiType.OPTIMISTIC
DETERMINED = EmojiType.DETERMINED
AFFECTIONATE = EmojiType.AFFECTIONATE
CALM = EmojiType.CALM
QUIET = EmojiType.QUIET
SHY = EmojiType.SHY
CHEERFUL = EmojiType.CHEERFUL
LOVED = EmojiType.LOVED
SURPRISED = EmojiType.SURPRISED
THINKING = EmojiType.THINKING
TIRED = EmojiType.TIRED
CONFUSED = EmojiType.CONFUSED
BORED = EmojiType.BORED
EMBARRASSED = EmojiType.EMBARRASSED
WORRIED = EmojiType.WORRIED
SAD = EmojiType.SAD
SICK = EmojiType.SICK
DISAPPOINTED = EmojiType.DISAPPOINTED
NERVOUS = EmojiType.NERVOUS
ANNOYED = EmojiType.ANNOYED
STRESSED = EmojiType.STRESSED
ANGRY = EmojiType.ANGRY
FRUSTRATED = EmojiType.FRUSTRATED
JEALOUS = EmojiType.JEALOUS
SHOCKED = EmojiType.SHOCKED
FEAR = EmojiType.FEAR
DISGUST = EmojiType.DISGUST

LOOK_FORWARD = EmojiLookType.LOOK_FORWARD
'''The emoji will look forwards.'''
LOOK_RIGHT = EmojiLookType.LOOK_RIGHT
'''The emoji will look to the right.'''
LOOK_LEFT = EmojiLookType.LOOK_LEFT
'''The emoji will look to the left.'''

DOORBELL = SoundType.DOORBELL
TADA = SoundType.TADA
FAIL = SoundType.FAIL
SPARKLE = SoundType.SPARKLE
FLOURISH = SoundType.FLOURISH
MOVE_FORWARD = SoundType.FORWARD
MOVE_REVERSE = SoundType.REVERSE
TURN_RIGHT = SoundType.RIGHT
TURN_LEFT = SoundType.LEFT
BLINKER = SoundType.BLINKER
CRASH = SoundType.CRASH
BRAKES = SoundType.BRAKES
HUAH = SoundType.HUAH
PICKUP = SoundType.PICKUP
CHEER = SoundType.CHEER
SENSING = SoundType.SENSING
DETECTED = SoundType.DETECTED
OBSTACLE = SoundType.OBSTACLE
LOOPING = SoundType.LOOPING
COMPLETE = SoundType.COMPLETE
PAUSE = SoundType.PAUSE
RESUME = SoundType.RESUME
SEND = SoundType.SEND
RECEIVE = SoundType.RECEIVE


ACT_HAPPY = SoundType.ACT_HAPPY
ACT_SAD = SoundType.ACT_SAD
ACT_EXCITED = SoundType.ACT_EXCITED
ACT_ANGRY = SoundType.ACT_ANGRY
ACT_SILLY = SoundType.ACT_SILLY

SPORTS_BALL = VisionObject.SPORTS_BALL
'''A description for get_data indicating the AI ball objects to be returned'''
BLUE_BARREL = VisionObject.BLUE_BARREL
'''A description for get_data indicating the AI blue barrel objects to be returned'''
ORANGE_BARREL = VisionObject.ORANGE_BARREL
'''A description for get_data indicating the AI orange barrel objects to be returned'''
AIM_ROBOT = VisionObject.AIM_ROBOT
'''A description for get_data indicating the AI robot objects to be returned'''
TAG0 = VisionObject.TAG0
'''A description for get_data indicating apriltags with id 0 to be returned'''
TAG1 = VisionObject.TAG1
'''A description for get_data indicating apriltags with id 1 to be returned'''
TAG2 = VisionObject.TAG2
'''A description for get_data indicating apriltags with id 2 to be returned'''
TAG3 = VisionObject.TAG3
'''A description for get_data indicating apriltags with id 3 to be returned'''
TAG4 = VisionObject.TAG4
'''A description for get_data indicating apriltags with id 4 to be returned'''
TAG5 = VisionObject.TAG5
'''A description for get_data indicating apriltags with id 5 to be returned'''
TAG6 = VisionObject.TAG6
'''A description for get_data indicating apriltags with id 6 to be returned'''
TAG7 = VisionObject.TAG7
'''A description for get_data indicating apriltags with id 7 to be returned'''
TAG8 = VisionObject.TAG8
'''A description for get_data indicating apriltags with id 8 to be returned'''
TAG9 = VisionObject.TAG9
'''A description for get_data indicating apriltags with id 9 to be returned'''
TAG10 = VisionObject.TAG10
'''A description for get_data indicating apriltags with id 10 to be returned'''
TAG11 = VisionObject.TAG11
'''A description for get_data indicating apriltags with id 11 to be returned'''
TAG12 = VisionObject.TAG12
'''A description for get_data indicating apriltags with id 12 to be returned'''
TAG13 = VisionObject.TAG13
'''A description for get_data indicating apriltags with id 13 to be returned'''
TAG14 = VisionObject.TAG14
'''A description for get_data indicating apriltags with id 14 to be returned'''
TAG15 = VisionObject.TAG15
'''A description for get_data indicating apriltags with id 15 to be returned'''
TAG16 = VisionObject.TAG16
'''A description for get_data indicating apriltags with id 16 to be returned'''
TAG17 = VisionObject.TAG17
'''A description for get_data indicating apriltags with id 17 to be returned'''
TAG18 = VisionObject.TAG18
'''A description for get_data indicating apriltags with id 18 to be returned'''
TAG19 = VisionObject.TAG19
'''A description for get_data indicating apriltags with id 19 to be returned'''
TAG20 = VisionObject.TAG20
'''A description for get_data indicating apriltags with id 20 to be returned'''
TAG21 = VisionObject.TAG21
'''A description for get_data indicating apriltags with id 21 to be returned'''
TAG22 = VisionObject.TAG22
'''A description for get_data indicating apriltags with id 22 to be returned'''
TAG23 = VisionObject.TAG23
'''A description for get_data indicating apriltags with id 23 to be returned'''
TAG24 = VisionObject.TAG24
'''A description for get_data indicating apriltags with id 24 to be returned'''
TAG25 = VisionObject.TAG25
'''A description for get_data indicating apriltags with id 25 to be returned'''
TAG26 = VisionObject.TAG26
'''A description for get_data indicating apriltags with id 26 to be returned'''
TAG27 = VisionObject.TAG27
'''A description for get_data indicating apriltags with id 27 to be returned'''
TAG28 = VisionObject.TAG28
'''A description for get_data indicating apriltags with id 28 to be returned'''
TAG29 = VisionObject.TAG29
'''A description for get_data indicating apriltags with id 29 to be returned'''
TAG30 = VisionObject.TAG30
'''A description for get_data indicating apriltags with id 30 to be returned'''
TAG31 = VisionObject.TAG31
'''A description for get_data indicating apriltags with id 31 to be returned'''
TAG32 = VisionObject.TAG32
'''A description for get_data indicating apriltags with id 32 to be returned'''
TAG33 = VisionObject.TAG33
'''A description for get_data indicating apriltags with id 33 to be returned'''
TAG34 = VisionObject.TAG34
'''A description for get_data indicating apriltags with id 34 to be returned'''
TAG35 = VisionObject.TAG35
'''A description for get_data indicating apriltags with id 35 to be returned'''
TAG36 = VisionObject.TAG36
'''A description for get_data indicating apriltags with id 36 to be returned'''
TAG37 = VisionObject.TAG37
'''A description for get_data indicating apriltags with id 37 to be returned'''

ALL_TAGS = VisionObject.ALL_TAGS
'''A description for get_data indicating any apriltag to be returned'''
ALL_VISION = VisionObject.ALL_VISION
'''A description for get_data indicating any object to be returned'''
ALL_COLORS = VisionObject.ALL_COLORS
'''A description for get_data indicating any color or code to be returned'''
ALL_CARGO = VisionObject.ALL_CARGO
'''A description for get_data indicating AI ball or barrel to be returned'''
