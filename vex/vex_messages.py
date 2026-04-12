# =================================================================================================
#  Copyright (c) Innovation First 2025. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# =================================================================================================
""" 
AIM WebSocket - Messages

This module contains definitions for the Websocket messages. 
"""
class VexWebSocketCommand:
    def __init__(self, cmd_id: str):
        self.cmd_id = cmd_id

    def to_json(self) -> dict:
        return {
            "cmd_id": self.cmd_id
        }
        
#region General Commands
class ProgramInit(VexWebSocketCommand):
    def __init__(self):
        super().__init__("program_init")


#endregion General Commands

#region Movement Commands
class MoveAt(VexWebSocketCommand):
    def __init__(self, angle=0.0, speed=0.0, stacking_type=0):
        super().__init__("drive")
        self.angle = angle
        self.speed = speed
        self.stacking_type = stacking_type

    def to_json(self):
        base_data = super().to_json()
        base_data.update({
            "angle": self.angle,
            "speed": self.speed,
            "stacking_type": self.stacking_type
        })
        return base_data

class MoveFor(VexWebSocketCommand):
    def __init__(self, distance =0.0, angle=0.0, drive_speed=0.0, turn_speed=0.0, final_heading=0,stacking_type=0):
        super().__init__("drive_for")
        self.distance = distance
        self.angle = angle
        self.drive_speed = drive_speed
        self.turn_speed = turn_speed
        self.final_heading = final_heading
        self.stacking_type = stacking_type

    def to_json(self):
        base_data = super().to_json()
        base_data.update({
            "distance": self.distance,
            "angle": self.angle,
            "final_heading" : self.final_heading,
            "drive_speed": self.drive_speed,
            "turn_speed": self.turn_speed,
            "stacking_type": self.stacking_type
        })
        return base_data

class MoveWithVector(VexWebSocketCommand):
    def __init__(self, x=0, t=0, r=0):
        super().__init__("drive_with_vector")
        self.x = x
        self.t  = t
        self.r = r

    def to_json(self):
        base_data = super().to_json()
        base_data.update({
            "x": self.x,
            "t": self.t,
            "r": self.r
        })
        return base_data   

class Turn(VexWebSocketCommand):
    def __init__(self, turn_rate=0.0, stacking_type=0):
        super().__init__("turn")
        self.turn_rate = turn_rate
        self.stacking_type = stacking_type

    def to_json(self):
        base_data = super().to_json()
        base_data.update({
            "turn_rate": self.turn_rate,
            "stacking_type": self.stacking_type
        })
        return base_data

class TurnTo(VexWebSocketCommand):
    def __init__(self, heading=0.0, turn_rate=0.0, stacking_type=0):
        super().__init__("turn_to")
        self.heading = heading
        self.turn_rate = turn_rate
        self.stacking_type = stacking_type

    def to_json(self):
        base_data = super().to_json()
        base_data.update({
            "heading": self.heading,
            "turn_rate": self.turn_rate,
            "stacking_type": self.stacking_type
        })
        return base_data
    
class TurnFor(VexWebSocketCommand):
    def __init__(self, angle=0, turn_rate=0.0, stacking_type=0):
        super().__init__("turn_for")
        self.angle = angle
        self.turn_rate = turn_rate
        self.stacking_type = stacking_type

    def to_json(self):
        base_data = super().to_json()
        base_data.update({
            "angle": self.angle,
            "turn_rate": self.turn_rate,
            "stacking_type": self.stacking_type
        })
        return base_data

class SpinWheels(VexWebSocketCommand):
    def __init__(self, vel1=0, vel2=0, vel3=0):
        super().__init__("spin_wheels")
        self.vel1 = vel1
        self.vel2 = vel2
        self.vel3 = vel3

    def to_json(self):
        base_data = super().to_json()
        base_data.update({
            "vel1": self.vel1,
            "vel2": self.vel2,
            "vel3": self.vel3
        })
        return base_data

class SetPose(VexWebSocketCommand):
    def __init__(self, x=0, y=0):
        super().__init__("set_pose")
        self.x = x
        self.y = y
        
    def to_json(self):
        base_data = super().to_json()
        base_data.update({
            "x": self.x,
            "y": self.y
        })
        return base_data
#endregion Movement Commands

#region Screen Commands
class ScreenPrint(VexWebSocketCommand):
    def __init__(self, string=""):
        super().__init__("lcd_print")
        self.string = string

    def to_json(self):
        base_data = super().to_json()
        base_data.update({
            "string": self.string
        })
        return base_data

class ScreenPrintAt(VexWebSocketCommand):
    def __init__(self, string="", x=0, y=0, b_opaque=True):
        super().__init__("lcd_print_at")
        self.string = string
        self.x = x
        self.y = y
        self.b_opaque = b_opaque

    def to_json(self):
        base_data = super().to_json()
        base_data.update({
            "x": self.x,
            "y": self.y,
            "string": self.string,
            "b_opaque": self.b_opaque
        })
        return base_data

class ScreenSetCursor(VexWebSocketCommand):
    def __init__(self, row=0, col=0):
        super().__init__("lcd_set_cursor")
        self.row = row
        self.col = col

    def to_json(self):
        base_data = super().to_json()
        base_data.update({
            "row": self.row,
            "col": self.col
        })
        return base_data

class ScreenSetOrigin(VexWebSocketCommand):
    def __init__(self, x=0, y=0):
        super().__init__("lcd_set_origin")
        self.x = x
        self.y = y

    def to_json(self):
        base_data = super().to_json()
        base_data.update({
            "x": self.x,
            "y": self.y
        })
        return base_data

class ScreenNextRow(VexWebSocketCommand):
    def __init__(self):
        super().__init__("lcd_next_row")

    def to_json(self):
        return super().to_json()

class ScreenClearRow(VexWebSocketCommand):
    def __init__(self, row=0, r=0,g=0,b=0):
        super().__init__("lcd_clear_row")
        self.row = row
        self.r = r
        self.g = g
        self.b = b

    def to_json(self):
        base_data = super().to_json()
        base_data.update({
            "number": self.row,
            "r": self.r,
            "g": self.g,
            "b": self.b
        })
        return base_data
class ScreenClear(VexWebSocketCommand):
    def __init__(self, r=0, g=0, b=0):
        super().__init__("lcd_clear_screen")
        self.r = r
        self.g = g
        self.b = b

    def to_json(self):
        base_data = super().to_json()
        base_data.update({
            "r": self.r,
            "g": self.g,
            "b": self.b
        })
        return base_data

class ScreenSetFont(VexWebSocketCommand):
    def __init__(self, fontname):
        super().__init__("lcd_set_font")
        self.fontname = fontname

    def to_json(self):
        base_data = super().to_json()
        base_data.update({
            "fontname": self.fontname
        })
        return base_data

class ScreenSetPenWidth(VexWebSocketCommand):
    def __init__(self, width):
        super().__init__("lcd_set_pen_width")
        self.width = width

    def to_json(self):
        base_data = super().to_json()
        base_data.update({
            "width": self.width
        })
        return base_data

class ScreenSetPenColor(VexWebSocketCommand):
    def __init__(self, r=0, g=0, b=0):
        super().__init__("lcd_set_pen_color")
        self.r = r
        self.g = g
        self.b = b

    def to_json(self):
        base_data = super().to_json()
        base_data.update({
            "r": self.r,
            "g": self.g,
            "b": self.b
        })
        return base_data
class ScreenSetFillColor(VexWebSocketCommand):
    def __init__(self, r=0, g=0, b=0, transparent=False):
        super().__init__("lcd_set_fill_color")
        self.r = r
        self.g = g
        self.b = b
        self.b_transparency = transparent

    def to_json(self):
        base_data = super().to_json()
        base_data.update({
            "r": self.r,
            "g": self.g,
            "b": self.b,
            "b_transparency":self.b_transparency
        })
        return base_data

class ScreenDrawLine(VexWebSocketCommand):
    def __init__(self, x1=0, y1=0, x2=0, y2=0):
        super().__init__("lcd_draw_line")
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def to_json(self):
        base_data = super().to_json()
        base_data.update({
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2
        })
        return base_data

class ScreenDrawRectangle(VexWebSocketCommand):
    def __init__(self, x=0, y=0, width=0, height=0, r=0, g=0, b=0, transparent=False):
        super().__init__("lcd_draw_rectangle")
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.r = r
        self.g = g
        self.b = b
        self.b_transparency = transparent

    def to_json(self):
        base_data = super().to_json()
        base_data.update({
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "r": self.r,
            "g": self.g,
            "b": self.b,
            "b_transparency":self.b_transparency
        })
        return base_data

class ScreenDrawCircle(VexWebSocketCommand):
    def __init__(self, x=0, y=0, radius=0, r=0, g=0, b=0, transparent=False):
        super().__init__("lcd_draw_circle")
        self.x = x
        self.y = y
        self.radius = radius
        self.r = r
        self.g = g
        self.b = b
        self.b_transparency = transparent

    def to_json(self):
        base_data = super().to_json()
        base_data.update({
            "x": self.x,
            "y": self.y,
            "radius": self.radius,
            "r": self.r,
            "g": self.g,
            "b": self.b,
            "b_transparency":self.b_transparency
        })
        return base_data

class ScreenDrawPixel(VexWebSocketCommand):
    def __init__(self, x=0, y=0):
        super().__init__("lcd_draw_pixel")
        self.x = x
        self.y = y

    def to_json(self):
        base_data = super().to_json()
        base_data.update({
            "x": self.x,
            "y": self.y
        })
        return base_data

class ScreenDrawImageFromFile(VexWebSocketCommand):
    def __init__(self, filename="", x=0, y=0):
        super().__init__("lcd_draw_image_from_file")
        self.filename = filename
        self.x = x
        self.y = y

    def to_json(self):
        base_data = super().to_json()
        base_data.update({
            "filename": self.filename,
            "x": self.x,
            "y": self.y
        })
        return base_data

class ScreenSetClipRegion(VexWebSocketCommand):
    def __init__(self, x=0, y=0, width=0, height=0):
        super().__init__("lcd_set_clip_region")
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def to_json(self):
        base_data = super().to_json()
        base_data.update({
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height
        })
        return base_data

class ScreenShowEmoji(VexWebSocketCommand):
    def __init__(self, name=0, look=0):
        super().__init__("show_emoji")
        self.name = name
        self.look = look

    def to_json(self):
        base_data = super().to_json()
        base_data.update({
            "name": self.name,
            "look": self.look
        })
        return base_data

class ScreenHideEmoji(VexWebSocketCommand):
    def __init__(self):
        super().__init__("hide_emoji")
    def to_json(self):
        return super().to_json()

class ScreenShowAivision(VexWebSocketCommand):
    def __init__(self, name=0, look=0):
        super().__init__("show_aivision")
    def to_json(self):
        return super().to_json()

class ScreenHideAivision(VexWebSocketCommand):
    def __init__(self, name=0, look=0):
        super().__init__("hide_aivision")
    def to_json(self):
        return super().to_json()
#endregion Screen Commands

#region Interial Commands
class InterialCalibrate(VexWebSocketCommand):
    def __init__(self):
        super().__init__("imu_calibrate")

    def to_json(self):
        return super().to_json()

class InterialSetCrashSensitivity(VexWebSocketCommand):
    def __init__(self, sensitivity=0):
        super().__init__("imu_set_crash_threshold")
        self.sensitivity = sensitivity

    def to_json(self):
        base_data = super().to_json()
        base_data.update({
            "sensitivity": self.sensitivity
        })
        return base_data
#endregion Interial Commands

#region Kicker Commands
class KickerKick(VexWebSocketCommand):
    def __init__(self, kick_type=""):
        super().__init__(kick_type)

    def to_json(self):
        return super().to_json()
#endregion Kicker Commands

#region Sound Commands
class SoundPlay(VexWebSocketCommand):
    def __init__(self, name="", volume=0):
        super().__init__("play_sound")
        self.name = name
        self.volume = volume

    def to_json(self):
        base_data = super().to_json()
        base_data.update({
            "name": self.name,
            "volume": self.volume
        })
        return base_data
class SoundPlayFile(VexWebSocketCommand):
    def __init__(self, name="", volume=0):
        super().__init__("play_file")
        self.name = name
        self.volume = volume

    def to_json(self):
        base_data = super().to_json()
        base_data.update({
            "name": self.name,
            "volume": self.volume
        })
        return base_data
class SoundPlayNote(VexWebSocketCommand):
    def __init__(self, note=0, octave=0, duration=500, volume=0):
        super().__init__("play_note")
        self.note = note
        self.octave = octave
        self.duration = duration
        self.volume = volume

    def to_json(self):
        base_data = super().to_json()
        base_data.update({
            "note": self.note,
            "octave": self.octave,
            "duration": self.duration,
            "volume": self.volume
        })
        return base_data

class SoundStop(VexWebSocketCommand):
    def __init__(self):
        super().__init__("stop_sound")

    def to_json(self):
        return super().to_json()

#endregion Sound Commands

#region LED Commands
class LedSet(VexWebSocketCommand):
    def __init__(self, led="", r=0, g=0, b=0):
        super().__init__("light_set")
        self.led = led
        self.r = r
        self.g = g
        self.b = b

    def to_json(self):
        base_data = super().to_json()
        base_data.update({
            self.led: {
            "r": self.r,
            "g": self.g,
            "b": self.b
            }
        })
        return base_data

#endregion LED Commands

#region AiVision Commands
class VisionColorDescription(VexWebSocketCommand):
    def __init__(self, id, r, g, b, hangle, hdsat ):
        super().__init__("color_description")
        self.id = id
        self.r = r
        self.g = g
        self.b = b
        self.hdsat = hdsat
        self.hangle = hangle

    def to_json(self):
        base_data = super().to_json()
        base_data.update({
            "id": self.id,
            "red": self.r,
            "green": self.g,
            "blue": self.b,
            "hangle": self.hangle,
            "hdsat": self.hdsat
        })
        return base_data

class VisionCodeDescription(VexWebSocketCommand):
    def __init__(self, id, c1, c2, *args):
        super().__init__("code_description")
        self.id = id
        self.c1 = c1.id
        self.c2 = c2.id
        self.c3 = -1
        self.c4 = -1
        self.c5 = -1
        if( len(args) > 0 ):
            self.c3 = args[0].id
        if( len(args) > 1 ):
            self.c3 = args[1].id
        if( len(args) > 2 ):
            self.c3 = args[2].id

    def to_json(self):
        base_data = super().to_json()
        base_data.update({
            "id": self.id,
            "c1": self.c1,
            "c2": self.c2,
            "c3": self.c3,
            "c4": self.c4,
            "c5": self.c5
        })
        return base_data
class VisionTagDetection(VexWebSocketCommand):
    def __init__(self, enable=True):
        super().__init__("tag_detection")
        self.b_enable = enable

    def to_json(self):
        base_data = super().to_json()
        base_data.update({
            "b_enable": self.b_enable
        })
        return base_data

class VisionColorDetection(VexWebSocketCommand):
    def __init__(self, enable=True, merge=True):
        super().__init__("color_detection")
        self.b_enable = enable
        self.b_merge = merge

    def to_json(self):
        base_data = super().to_json()
        base_data.update({
            "b_enable": self.b_enable,
            "b_merge": self.b_merge
        })
        return base_data
class VisionModelDetection(VexWebSocketCommand):
    def __init__(self, enable=True):
        super().__init__("model_detection")
        self.b_enable = enable

    def to_json(self):
        base_data = super().to_json()
        base_data.update({
            "b_enable": self.b_enable
        })
        return base_data
#endregion AiVision Commands
