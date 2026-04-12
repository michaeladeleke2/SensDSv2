# =================================================================================================
#  Copyright (c) Innovation First 2025. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# =================================================================================================
""" 
AIM WebSocket Python Client

This library is designed for use with the VEX AIM Robot. It provides a simple interface
for controlling the robot remotely over Wi-Fi and for receiving status updates.

Communication is handled via the WebSocket protocol. The library includes a set of
classes and methods for controlling robot movement, sensors, and other features.
"""

# pylint: disable=unnecessary-pass,unused-argument,line-too-long,too-many-lines
# pylint: disable=invalid-name,unused-import,redefined-outer-name
import io
import json
import time
import threading
import sys
from typing import Union, cast, Callable
import math
import atexit
import signal
import _thread
import pathlib
import websocket
from .settings import Settings
from . import vex_types as vex #importing from the same package
from . import vex_messages as commands
#module-specific "constant" globals
VERSION_MAJOR = 1
VERSION_MINOR = 0
VERSION_BUILD = 0
VERSION_BETA  = 0
SYS_FLAGS_SOUND_PLAYING     =  1<<0
SYS_FLAGS_IS_SOUND_DNL      =  1<<16
SYS_FLAGS_IS_MOVE_ACTIVE    =  1<<1 # if a move_at or move_for (or any "move" command) is active
SYS_FLAGS_IMU_CAL           =  1<<3
SYS_FLAGS_IS_TURN_ACTIVE    =  1<<4
SYS_FLAGS_IS_MOVING         =  1<<5 # if there is any wheel movement whatsoever
SYS_FLAGS_HAS_CRASHED       =  1<<6
SYS_FLAGS_IS_SHAKE          =  1<<8
SYS_FLAGS_PWR_BUTTON        =  1<<9
SYS_FLAGS_PROG_ACTIVE       =  1<<10

SOUND_SIZE_MAX_BYTES     = 255 * 1024
BARREL_MIN_Y             = 160
BARREL_MIN_CX            = 120
BARREL_MAX_CX            = 200

BALL_MIN_Y          = 170
BALL_MIN_CX         = 120
BALL_MAX_CX         = 200

AIVISION_MAX_OBJECTS               = 24
AIVISION_DEFAULT_SNAPSHOT_OBJECTS  =  8

DRIVE_VELOCITY_MAX_MMPS = 200 # millimeters per second
TURN_VELOCITY_MAX_DPS   = 180 # Degrees per second
class AimException(Exception):
    """VEX AIM Exception Class"""

class DisconnectedException(AimException):
    """Exception that is thrown when robot is not connected over Websocket"""

class NoImageException(AimException):
    """No image was received"""

class ReceiveErrorException(AimException):
    """(internally used) error receiving WS frame"""

class InvalidSoundFileException(AimException):
    """Sound file extension or format is not supported"""

class InvalidImageFileException(AimException):
    """image file extension or format is not supported"""

class WSThread(threading.Thread):
    """ Base class for all Websocket threads"""
    def __init__ (self, host, ws_name):
        threading.Thread.__init__(self)
        self.host = host
        self.ws_name = ws_name
        self.uri = f"ws://{self.host}/{self.ws_name}"
        self.ws = self.connect_websocket(timeout=4)
        self.callback = None
        self.running = True
        #set equal to true of connection needs to be reset (disconnect, reconnect)
        self._ws_needs_reset = False

    def connect_websocket(self, timeout):
        """ connect to the websocket server"""
        ws = websocket.WebSocket()
        try:
            ws.connect(self.uri, timeout=timeout)
        except Exception as error:
            print(
                f"Could not connect to {self.uri} (reason: {error}). "
                f"Verify that \"{self.host}\" is the correct IP/hostname of the AIM robot "
                "and that it is connected to the same network (AP mode is 192.168.4.1)"
            )
            sys.exit(1)
        return ws

    def ws_send(self, payload: Union[bytes, str], opcode: int = websocket.ABNF.OPCODE_TEXT):
        """ send data to the robot over the websocket connection"""
        try:
            self.ws.send(payload, opcode)
        except (ConnectionResetError, websocket.WebSocketException) as error:
            self._ws_needs_reset = True
            raise DisconnectedException(
                f"{self.ws_name}: error sending data to robot, apparently disconnected, "
                f"will try to reconnect; error: '{error}'"
            ) from error

    def ws_receive(self):
        """ receive data from the robot over the websocket connection"""
        try:
            data = self.ws.recv()
        except (ConnectionResetError, websocket.WebSocketException) as error:
            self._ws_needs_reset = True
            raise ReceiveErrorException(
                f"{self.ws_name}: error receiving data from robot, apparently disconnected, "
                f"will try to reconnect; error: '{error}'"
            ) from error
        return data

    def ws_close(self):
        """ close the websocket connection"""
        try:
            self.ws.close()
        except Exception as error:
            print(f"Failed to close WebSocket connection. It might already be closed. Error: {error}")

class WSStatusThread(WSThread):
    """ Websocket thread for listening to different status updates from the robot"""
    def __init__(self, host):
        super().__init__ (host, "ws_status")
        self._empty_status = {
            "controller": {"flags": "0x0000", "stick_x": 0, "stick_y": 0, "battery": 0},
            "robot": {
                "flags": "0x00000000",
                "battery": 0,
                "touch_flags": "0x0000",
                "touch_x": 0,
                "touch_y": 0,
                "robot_x": 0,
                "robot_y": 0,
                "roll": "0",
                "pitch": "0",
                "yaw": "0",
                "heading": "0",
                "acceleration": {"x": "0", "y": "0", "z": "0"},
                "gyro_rate": {"x": "0", "y": "0", "z": "0"},
                "screen": {"row": "1", "column": "1"}
            },
            "aivision": {
                "classnames": {
                    "count": 4,
                    "items": [
                        {"index": 0, "name": "SportsBall"},
                        {"index": 1, "name": "BlueBarrel"},
                        {"index": 2, "name": "OrangeBarrel"},
                        {"index": 3, "name": "Robot"},
                    ],
                },
                "objects": {
                    "count": 0,
                    "items": [
                        {
                            "type": 0,
                            "id": 0,
                            "type_str": "0",
                            "originx": 0,
                            "originy": 0,
                            "width": 0,
                            "height": 0,
                            "score": 0,
                            "name": "0",
                        }
                    ]},
            },
        }
        self.current_status = self._empty_status
        self.is_move_active_flag_needs_setting    = False
        self.is_turn_active_flag_needs_setting    = False
        self.is_moving_flag_needs_setting         = False
        self.is_moving_flag_needs_clearing        = False
        self.imu_cal_flag_needs_setting           = False
        self.sound_playing_flag_needs_setting     = False
        self.sound_downloading_flag_needs_setting = False

        self._packets_lost_counter = 0
        self.heartbeat = 0
        self.program_active = False

        #need to have list of callback for screen pressed
        self._screen_pressed_callbacks = []
        self._screen_released_callbacks = []
        self._inertial_crashed_callbacks = []
        self._last_screen_pressed = False

    def update_status_flags(self):
        """ update the status flags based on the current status received from the robot"""
        if self.is_move_active_flag_needs_setting:
            self.set_is_move_active_flag()
            self.is_move_active_flag_needs_setting = False

        if self.is_turn_active_flag_needs_setting:
            self.set_is_turn_active_flag()
            self.is_turn_active_flag_needs_setting = False

        if self.is_moving_flag_needs_setting:
            self.set_is_moving_flag()
            self.is_moving_flag_needs_setting = False

        if self.is_moving_flag_needs_clearing:
            self.clear_is_moving_flag()
            self.is_moving_flag_needs_clearing = False

        if self.imu_cal_flag_needs_setting:
            self.set_imu_cal_flag()
            self.imu_cal_flag_needs_setting = False

        if self.sound_playing_flag_needs_setting:
            self.set_sound_playing_flag()
            self.sound_playing_flag_needs_setting = False

        if self.sound_downloading_flag_needs_setting:
            self.set_sound_downloading_flag()
            self.sound_downloading_flag_needs_setting = False


    def set_is_move_active_flag(self):
        """ set the is_move_active flag in the robot status"""
        robot_flags = self.current_status["robot"]["flags"]
        #update is_move_active flag (convert robot_flags to int, set bit 1 to 1, then convert back to hex string)
        new_robot_flags = hex(int(robot_flags, 16) | SYS_FLAGS_IS_MOVE_ACTIVE)
        self.current_status["robot"]["flags"] = new_robot_flags

    def set_is_turn_active_flag(self):
        """ set the is_turn_active flag in the robot status"""
        robot_flags = self.current_status["robot"]["flags"]
        #update is_turn_active flag (convert robot_flags to int, set bit 1 to 1, then convert back to hex string)
        new_robot_flags = hex(int(robot_flags, 16) | SYS_FLAGS_IS_TURN_ACTIVE)
        self.current_status["robot"]["flags"] = new_robot_flags

    def set_is_moving_flag(self):
        """ set the is_moving flag in the robot status"""
        robot_flags = self.current_status["robot"]["flags"]
        #update is_moving flag
        #convert robot_flags to int, set bit 1 to 1, then convert back to hex string
        new_robot_flags = hex(int(robot_flags, 16) | SYS_FLAGS_IS_MOVING)
        self.current_status["robot"]["flags"] = new_robot_flags

    def clear_is_moving_flag(self):
        """ reset the is_moving flag in the robot status"""
        robot_flags = self.current_status["robot"]["flags"]
        #update is_moving flag
        #convert robot_flags to int, set bit 1 to 1, then convert back to hex string
        new_robot_flags = hex(int(robot_flags, 16) & ~SYS_FLAGS_IS_MOVING)
        self.current_status["robot"]["flags"] = new_robot_flags

    def set_imu_cal_flag(self):
        """ set the flag to say is IMU calibrating based on the robot status"""
        robot_flags = self.current_status["robot"]["flags"]
        #update imu_cal flag
        #convert robot_flags to int, set bit 1 to 1, then convert back to hex string
        new_robot_flags = hex(int(robot_flags, 16) | SYS_FLAGS_IMU_CAL)
        self.current_status["robot"]["flags"] = new_robot_flags

    def set_sound_playing_flag(self):
        """ set the flag to say sound is playing based on the status"""
        robot_flags = self.current_status["robot"]["flags"]
        #update sound_playing flag
        #convert robot_flags to int, set bit 1 to 1, then convert back to hex string
        new_robot_flags = hex(int(robot_flags, 16) | SYS_FLAGS_SOUND_PLAYING)
        self.current_status["robot"]["flags"] = new_robot_flags

    def set_sound_downloading_flag(self):
        """ set the flag to say sound is downloading based on the status"""
        robot_flags = self.current_status["robot"]["flags"]
        #update is_sound_downloading flag
        # convert robot_flags to int, set bit 1 to 1, then convert back to hex string
        new_robot_flags = hex(int(robot_flags, 16) | SYS_FLAGS_IS_SOUND_DNL)
        self.current_status["robot"]["flags"] = new_robot_flags

    def check_shake_flag(self):
        """ detect if the robot has been shaken"""
        # previously, a shake would cause program to exit.
        # This was problematic, so nothing is done currently.
        robot_flags = self.current_status["robot"]["flags"]
        # if int(robot_flags, 16) & SYS_FLAGS_IS_SHAKE != 0:
            # _thread.interrupt_main()

    def check_power_button_flag(self):
        """ detect if the power button has been pressed to exit the program"""
        robot_flags = self.current_status["robot"]["flags"]
        if int(robot_flags, 16) & SYS_FLAGS_PWR_BUTTON != 0:
            print("detected power button press, exiting program")
            _thread.interrupt_main()

    def check_program_active_flag(self):
        """Going from program_active == True to program_active == False exits the program"""
        robot_flags = self.current_status["robot"]["flags"]
        program_active_old = self.program_active

        if int(robot_flags, 16) & SYS_FLAGS_PROG_ACTIVE != 0:
            self.program_active = True
        else:
            self.program_active = False

        if program_active_old and not self.program_active:
            print("detected that program is no longer active (robot power button pressed?), exiting program")
            _thread.interrupt_main()

    def check_crash_flag(self):
        """ detect if the robot has crashed and fire the appropriate callbacks"""
        robot_flags = self.current_status["robot"]["flags"]
        if int(robot_flags, 16) & SYS_FLAGS_HAS_CRASHED != 0:
            for (cb, args) in self._inertial_crashed_callbacks:
                cb(*args)

    def check_screen_pressing(self):
        """Check if the screen is being pressed, call pressed/released callbacks on transition."""
        is_pressed = bool(int(self.current_status["robot"]["touch_flags"], 16) & 0x0001)

        # If newly pressed
        if is_pressed and not self._last_screen_pressed:
            for (cb, args) in self._screen_pressed_callbacks:
                cb(*args)

        # If newly released
        if not is_pressed and self._last_screen_pressed:
            for (cb, args) in self._screen_released_callbacks:
                cb(*args)

        self._last_screen_pressed = is_pressed

    def is_current_status_empty(self):
        """ check if the current status is empty"""
        if self.current_status == self._empty_status:
            return True
        else:
            return False

    def add_screen_pressed_callback(self, callback: Callable[..., None], args: tuple = ()):
        """Adds a callback for screen press events, storing optional args."""
        # Store a tuple of (callback_function, args)
        self._screen_pressed_callbacks.append((callback, args))

    def add_screen_released_callback(self, callback: Callable[..., None], args: tuple = ()):
        """Adds a callback for screen release events, storing optional args."""
        self._screen_released_callbacks.append((callback, args))

    def add_inertial_crash_callback(self, callback: Callable[..., None], args: tuple = ()):
        """Adds a callback for inertial sensor's crash detected events, storing optional args."""
        # Store a tuple of (callback_function, args)
        self._inertial_crashed_callbacks.append((callback, args))

    def run(self):
        while self.running:
            new_status_json = ''
            new_status = self._empty_status
            if self._ws_needs_reset:
                self.ws_close()
                self._ws_needs_reset = False

            if self.ws.connected:
                status_packet_error = False
                self.ws_send((1).to_bytes(1, 'little'), websocket.ABNF.OPCODE_BINARY)
                try:
                    new_status_json = self.ws_receive()
                except Exception:
                    status_packet_error = True

                if not status_packet_error:
                    try:
                        new_status = json.loads(new_status_json)
                    except Exception:
                        status_packet_error = True

                #If we have an error receiving packet, initially we want to keep current_status unchanged.
                #After enough dropped packets, set current_status to empty values.
                if status_packet_error:
                    self._packets_lost_counter += 1
                    print(f"lost a status packet, counter: {self._packets_lost_counter}")
                #we have received a valid status packet, so update current_status:
                else:
                    self._packets_lost_counter = 0  #reset this counter
                    self.current_status = new_status
                    # print("current_status: ", self.current_status)
                    if self.callback:
                        if callable(self.callback):
                            self.callback()

                    # if a certain commands are sent, the very next status flag won't have the appropriate flags set yet, so over-ride locally.
                    self.update_status_flags()

                    self.check_shake_flag()
                    self.check_crash_flag()
                    self.check_screen_pressing()
                    self.check_power_button_flag()
                    self.check_program_active_flag()
                    self.heartbeat = not self.heartbeat

                if self._packets_lost_counter > 5:
                    self.current_status = self._empty_status

                # print("current_status: ", self.current_status)
                time.sleep(0.05)
            else:
                # print ("trying to reconnect to ws_status")
                try:
                    print(f"{self.ws_name} reconnecting")
                    self.ws.connect(self.uri)
                except:
                    pass # we'll keep trying to reconnect

class WSImageThread(WSThread):
    """ Websocket thread for receiving image stream from the robot"""
    def __init__(self, host):
        super().__init__ (host, "ws_img")
        self.current_image_index = 0
        self.image_list: list[bytes] = [bytes(1), bytes(1)]

        self._next_image_index = 1
        self._streaming = False

    def run(self):
        while self.running:
            if self._ws_needs_reset:
                self.ws_close()
                self._ws_needs_reset = False

            if self.ws.connected and self._streaming:
                if self.current_image_index == 1:
                    self._next_image_index    = 1
                    self.current_image_index = 0
                else:
                    self._next_image_index    = 0
                    self.current_image_index = 1

                try:
                    self.image_list[self._next_image_index] = cast(bytes, self.ws_receive()) # cast to narrow str | bytes down to bytes
                    if self.callback:
                        if callable(self.callback):
                            self.callback()
                except ReceiveErrorException:
                    self.image_list[self._next_image_index] = (0).to_bytes(1, 'little')

            elif not self.ws.connected:
                self._streaming = False
                try:
                    print(f"{self.ws_name} reconnecting")
                    self.ws.connect(self.uri)
                except:
                    pass # we'll keep trying to reconnect
            else:
                time.sleep(0.05)

    def stop_stream(self):
        """ stop the image stream"""
        self._streaming = False
        self.ws_send((0).to_bytes(1, 'little'), websocket.ABNF.OPCODE_BINARY)

    def start_stream(self):
        """ start the image stream"""
        self._streaming = True
        self.ws_send((1).to_bytes(1, 'little'), websocket.ABNF.OPCODE_BINARY)

class WSCommandThread(WSThread):
    """ Websocket thread for sending commands to the robot"""
    def __init__(self, host):
        super().__init__ (host, "ws_cmd")

    def run(self):
        while self.running:
            if self._ws_needs_reset:
                self.ws_close()
                self._ws_needs_reset = False

            if not self.ws.connected:
                try:
                    print(f"{self.ws_name} reconnecting")
                    self.ws.connect(self.uri)
                except:
                    pass # we'll keep on trying

            time.sleep(0.2)

class WSAudioThread(WSThread):
    """ Websocket thread for sending audio to the robot"""
    def __init__(self, host):
        super().__init__ (host, "ws_audio")

    def run(self):
        while self.running:
            if self._ws_needs_reset:
                self.ws_close()
                self._ws_needs_reset = False

            if not self.ws.connected:
                try:
                    print(f"{self.ws_name} reconnecting")
                    self.ws.connect(self.uri)
                except:
                    pass # we'll keep on trying

            time.sleep(0.2)

class ColorRGB:
    """ RGB color class"""
    def __init__(self, r: int, g: int, b: int, t: bool=False):
        self.r = r
        self.g = g
        self.b = b
        self.t = t

class Robot():
    """
    AIM Robot class.
    When initializing, provide a host (IP address, hostname, or even domain name) 
    or leave at empty for default if AIM is in WiFi AP mode.
    """
    #region private internal methods
    def __init__(self, host=""):
        """
        Initialize the Robot with default settings and WebSocket connections.
        """
       # If host is not provided, read the host from the settings file
        if host == "":
             # Create an instance of the Settings class to read the host from the settings file
            settings = Settings()
            host = settings.host

        self.host = host
        print(
            f"Welcome to the AIM Websocket Python Client. "
            f"Running version {VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_BUILD}.{VERSION_BETA} "
            f"and connecting to {self.host}"
        )
        self.move_active_cmd_list = ["drive", "drive_for"]
        self.turn_active_cmd_list = [ "turn", "turn_for", "turn_to"]
        self.stopped_active_cmd_list = self.move_active_cmd_list + self.turn_active_cmd_list


        self._ws_status_thread        = WSStatusThread(self.host)
        self._ws_status_thread.daemon = True
        self._ws_status_thread.start()

        self._ws_img_thread           = WSImageThread(self.host)
        self._ws_img_thread.daemon    = True
        self._ws_img_thread.start()

        self._ws_cmd_thread           = WSCommandThread(self.host)
        self._ws_cmd_thread.daemon    = True
        self._ws_cmd_thread.start()

        self._ws_audio_thread         = WSAudioThread(self.host)
        self._ws_audio_thread.daemon  = True
        self._ws_audio_thread.start()

        atexit.register(self.exit_handler)
        signal.signal(signal.SIGINT, self.kill_handler)
        signal.signal(signal.SIGTERM, self.kill_handler)

        self._program_init()

        self.drive_speed = 100 # Millimeters per second
        self.turn_speed  = 75 # Degrees per second
        # internal class instances to access through robot instance
        self.timer = Timer()
        self.screen   = Screen(self)
        self.inertial = Inertial(self)
        self.kicker   = Kicker(self)
        self.sound    = Sound(self)
        self.led      = Led(self)
        self.vision      = AiVision(self)

        # We don't want to execute certain things (like reset_heading) until we start getting status packets
        while self._ws_status_thread.is_current_status_empty() is True:
            # print("waiting for status")
            time.sleep(0.05)
        self.inertial.reset_heading()

    def exit_handler(self):
        """upon system exit, either due to SIGINT/SIGTERM or uncaught exceptions"""
        if hasattr(self, '_ws_cmd_thread'): #if connection were never established, this property wouldn't exist
            pass
            # print("program terminating, stopping robot")
            # #not attempting to stop robot since upon program end or connection loss, robot stops itself and wouldn't respond anyways.
            # try:
            #     self.stop_all_movement()
            # except Exception as error:
            #     print("exceptions arose during stop_all_movement(), error:", error)
        else:
            print("program terminating (never connected to robot)")

        try:
            self._ws_cmd_thread.running = False
        except:
            # print("ws_cmd_thread doesn't exist")
            pass
        try:
            self._ws_status_thread.running = False
        except:
            # print("ws_status_thread doesn't exist")
            pass
        try:
            self._ws_img_thread.running    = False
        except:
            # print("ws_img_thread doesn't exist")
            pass
        try:
            self._ws_img_thread.stop_stream()
        except:
            # print("error stopping ws_img stream")
            pass

    def kill_handler(self, signum, frame):
        """when kill signal is received, exit the program.  Will result in exit_handler being run"""
        signame = signal.Signals(signum).name
        print(f'Received signal {signame} ({signum})')
        sys.exit(0)

    def __getattribute__(self, name):
        """
        This function gets called before any other Robot function.\n
        If we are not connected to the robot (just looking at ws_cmd), then raise an exception
        and terminate program unless the user handles the exception.
        """
        method = object.__getattribute__(self, name)
        if not method:
            # raise Exception("Method %s not implemented" %name)
            return
        if callable(method):
            if self._ws_cmd_thread.ws.connected is False:
                raise DisconnectedException(f"error calling {name}: not connected to robot")
        return method

    def _program_init(self):
        """
        Sends a command indicating to robot that new program is starting.  
        To be called during __init__
        """
        message = commands.ProgramInit()
        self.robot_send(message.to_json())

    def _block_on_state(self, state_method):
        time_start = time.time()
        blocking = True
        while True:
            if state_method() is False: # debounce
                time.sleep(0.05)
                if state_method() is False:
                    break
            # print("blocking")
            time_elapsed = time.time() - time_start
            time.sleep(0.1)
            #if turning/moving took too long, we want to stop moving and stop blocking.
            if time_elapsed > 10:
                print(f"{state_method.__name__} wait timed out, stopping")
                self.stop_all_movement()
                return

    @property
    def status(self):
        """ returns the current status of the robot """
        return self._ws_status_thread.current_status
    #endregion private internal methods

    #region Generic methods to send commands to the robot
    def robot_send(self, json_cmd):
        """send a command to the robot through websocket connection"""
        disconnected_error = False
        #print(json_cmd)
        json_cmd_string = json.dumps(json_cmd, separators=(',',':'))
        #print("sending: ", json_cmd_string)
        try:
            cmd_id = json_cmd["cmd_id"]
        except:
            print("robot_send did not receive a cmd_id")
            return

        self._ws_cmd_thread.ws_send(str.encode(json_cmd_string), websocket.ABNF.OPCODE_BINARY)
        try:
            response_json = self._ws_cmd_thread.ws_receive()
        except ReceiveErrorException:
            disconnected_error = True
            raise DisconnectedException(f"robot got disconnected after sending cmd_id: {cmd_id}") from None # disable exception chaining
            # not trying to resend command because that would take too long, let user decide.

        try:
            response = json.loads(response_json)
        except Exception as error:
            print(f"{cmd_id} Error: could not parse ws_cmd JSON response: '{error}'")
            print("response_json", response_json)
            return

        # print("response_json", response_json)
        if response["cmd_id"] == "cmd_unknown":
            print("robot: did not recognize command: ", cmd_id)
            return

        if response["status"] == "error":
            try:
                error_info_string = response["error_info"]
            except KeyError:
                error_info_string = "no reason given"
            print("robot: error processing command, reason: ", error_info_string)
            return

        # trigger a local update to the robot status flags in ws_status_thread
        if response["status"] in ["complete", "in_progress"]:
            if response["cmd_id"] in self.move_active_cmd_list:
                self._ws_status_thread.is_move_active_flag_needs_setting = True
            if response["cmd_id"] in self.turn_active_cmd_list:
                self._ws_status_thread.is_turn_active_flag_needs_setting = True
            if response["cmd_id"] in self.stopped_active_cmd_list:
                self._ws_status_thread.is_moving_flag_needs_setting  = True
                self._ws_status_thread.is_moving_flag_needs_clearing = False
            if response["cmd_id"] == "imu_calibrate":
                self._ws_status_thread.imu_cal_flag_needs_setting = True

        return

    def robot_send_audio(self, audio):
        """ send audio to the robot through websocket audio thread"""
        self._ws_audio_thread.ws_send(audio, websocket.ABNF.OPCODE_BINARY)

    def add_screen_pressed_callback(self, callback: Callable[..., None], arg: tuple=()):
        """Adds a screen-pressed callback (delegate to _ws_status_thread)."""
        self._ws_status_thread.add_screen_pressed_callback(callback, arg)

    def add_screen_released_callback(self, callback: Callable[..., None], arg: tuple=()):
        """Adds a screen-released callback (delegate to _ws_status_thread)."""
        self._ws_status_thread.add_screen_released_callback(callback, arg)

    def add_inertial_crash_callback(self, callback: Callable[..., None], arg: tuple=()):
        """Adds a crash detected callback (delegate to _ws_status_thread)."""
        self._ws_status_thread.add_inertial_crash_callback(callback, arg)

    #endregion Generic methods to send commands to the robot
    #region Sensing Motion
    def get_x_position(self):
        """ returns the x position of the robot"""
        origin_x = float(self.status["robot"]["robot_x"])
        origin_y = float(self.status["robot"]["robot_y"])
        # print("raw:", origin_x, origin_y)
        offset_radians = -math.radians(self.inertial.heading_offset)
        x = origin_x * math.cos(offset_radians) + origin_y * math.sin(offset_radians)

        return x

    def get_y_position(self):
        """ returns the y position of the robot"""
        origin_x = float(self.status["robot"]["robot_x"])
        origin_y = float(self.status["robot"]["robot_y"])
        offset_radians = -math.radians(self.inertial.heading_offset)
        y = origin_y * math.cos(offset_radians) - origin_x * math.sin(offset_radians)

        return y

    def is_move_active(self):
        """returns true if a move_at() or move_for() command is active with nonzero speed"""
        if self._ws_status_thread.is_move_active_flag_needs_setting:
            return True
        robot_flags = self.status["robot"]["flags"]
        is_move_active = bool(int(robot_flags, 16) & SYS_FLAGS_IS_MOVE_ACTIVE)
        return is_move_active

    def is_turn_active(self):
        """returns true if a turn(), turn_to(), or turn_for() command is active with nonzero speed"""
        if self._ws_status_thread.is_turn_active_flag_needs_setting:
            return True
        robot_flags = self.status["robot"]["flags"]
        is_turn_active = bool(int(robot_flags, 16) & SYS_FLAGS_IS_TURN_ACTIVE)
        return is_turn_active

    def is_stopped(self):
        """returns true if no move, turn, or spin_wheels command is active (i.e. no wheels should be moving)"""
        if self._ws_status_thread.is_moving_flag_needs_clearing:
            return True
        if self._ws_status_thread.is_moving_flag_needs_setting:
            return False
        robot_flags = self.status["robot"]["flags"]
        is_stopped = not bool(int(robot_flags, 16) & SYS_FLAGS_IS_MOVING)
        return is_stopped
    #endregion Sensing Motion

    #region Sensing Battery
    def get_battery_capacity(self):
        """Get the remaining capacity of the battery (relative state of charge) in percent."""
        battery_capacity = self.status["robot"]["battery"]
        return battery_capacity
    #endregion Sensing Battery

    #region Motion Commands
    def set_move_velocity(self, velocity:vex.vexnumber, units:vex.DriveVelocityPercentUnits=vex.DriveVelocityUnits.PERCENT):
        """
        overrides the default velocity for all subsequent movement methods in the project.\n
        The default move velocity is 50% (100 millimeters per second)
        """
        #if velocity is negative throw value error
        if velocity < 0:
            raise ValueError("velocity must be a positive number")

        if units.value == vex.DriveVelocityUnits.PERCENT.value:
            #cannot exeed 100%
            if velocity > 100:
                velocity = 100
            #convert velocity in percentage to MMPS ex: 100% is 200 MMPS
            velocity = int(velocity * 2)
        elif units.value == vex.DriveVelocityUnits.MMPS.value:
            #cannot exceed MAX velocity MMPS
            if velocity > DRIVE_VELOCITY_MAX_MMPS:
                velocity = DRIVE_VELOCITY_MAX_MMPS
            velocity = int(velocity)

        self.drive_speed = velocity

    def set_turn_velocity(self, velocity:vex.vexnumber, units:vex.TurnVelocityPercentUnits=vex.TurnVelocityUnits.PERCENT):
        """ 
        overrides the default velocity for all subsequent turn methods in the project.\n
        The default turn velocity is 50% (75 degrees per second)
        """
        #if velocity is negative throw value error
        if velocity < 0:
            raise ValueError("velocity must be a positive number")

        if units.value == vex.TurnVelocityUnits.PERCENT.value:
            #cannot exeed 100%
            if velocity > 100:
                velocity = 100
            #if velocity is PERCENT convert to DPS ex: 100% is 180 MMPS
            velocity = int(velocity * 1.8)
        elif units.value == vex.TurnVelocityUnits.DPS.value:
            #cannot exceed MAX velocity MMPS
            if velocity > TURN_VELOCITY_MAX_DPS:
                velocity = TURN_VELOCITY_MAX_DPS
            velocity = int(velocity)

        self.turn_speed = velocity

    def move_at(self, angle:vex.vexnumber, velocity=None,
                units:vex.DriveVelocityPercentUnits=vex.DriveVelocityUnits.PERCENT):
        """
        move indefinitely at angle (-360 to 360 degrees) at velocity (0-100) \n
        if velocity is not provided, use the default set by set_move_velocity command \n
        The velocity unit is PERCENT (default) or millimeters per second MMPS
        """
        if velocity is None:
            velocity = self.drive_speed
        else:
            if units.value == vex.DriveVelocityUnits.PERCENT.value:
                #cannot exeed 100%
                if velocity > 100:
                    velocity = 100
                #if velocity is PERCENT convert to MMPS ex: 100% is 200 MMPS
                velocity = int(velocity * 2)
            elif units.value == vex.DriveVelocityUnits.MMPS.value:
                #cannot exceed MAX velocity MMPS
                if velocity > DRIVE_VELOCITY_MAX_MMPS:
                    velocity = DRIVE_VELOCITY_MAX_MMPS
                velocity = int(velocity)
        # passing negaitive velocity will flip the direction on the firmware side. No need to handle it here
        stacking_type = vex.StackingType.STACKING_OFF
        message = commands.MoveAt(angle, velocity,stacking_type.value)
        self.robot_send(message.to_json())

    def move_for(self, distance:vex.vexnumber, angle:vex.vexnumber, velocity=None, units:vex.DriveVelocityPercentUnits=vex.DriveVelocityUnits.PERCENT, wait=True):
        """move for a distance (mm) at angle (-360 to 360 degrees) at velocity (PERCENT/MMPS). 
        if velocity or turn_speed is not provided, use the default set by set_move_velocity and set_turn_velocity commands"""
        if velocity is None:
            velocity = self.drive_speed
        else:
            if units.value == vex.DriveVelocityUnits.PERCENT.value:
                #cannot exeed 100%
                if velocity > 100:
                    velocity = 100
                #if velocity is PERCENT convert to MMPS ex: 100% is 200 MMPS
                velocity = int(velocity * 2)
            elif units.value == vex.DriveVelocityUnits.MMPS.value:
                #cannot exceed MAX velocity MMPS
                if velocity > DRIVE_VELOCITY_MAX_MMPS:
                    velocity = DRIVE_VELOCITY_MAX_MMPS
                velocity = int(velocity)

        #if velocity is negative flip the direction
        if velocity < 0:
            velocity = -velocity
            distance = -distance

        turn_speed = self.turn_speed
        # if not final_heading:
        #     final_heading = self.get_heading_raw()
        heading = 0
        stacking_type = vex.StackingType.STACKING_OFF
        message = commands.MoveFor(distance, angle, velocity, turn_speed, heading, stacking_type.value)
        self.robot_send(message.to_json())
        if wait:
            self._block_on_state(self.is_move_active)

    def move_with_vectors(self, x, y, r):
        """
        moves the robot using vector-based motion, combining horizontal (X-axis) and 
        vertical (Y-axis) movement and having the robot to rotate at the same time
        
        x: X-axis velocity (in %). negative values move left and positive values move right\n
        y: Y-axis velocity (in %). negative values move backward and positive values move forward\n
        r: rotation velocity (in %). negative values move counter-clockwise and positive values move clockwise\n
        """
        # clip to +/- 100
        if x > 100:
            x = 100
        if x < -100:
            x = -100
        if y > 100:
            y = 100
        if y < -100:
            y = -100
        if r > 100:
            r = 100
        if r < -100:
            r = -100

        # scale
        x = x * 2.0
        y = y * 2.0
        r = r * 1.8

        # calculate wheel velocities
        w1 = (0.5 * x) + ((0.866) * y) + r
        w2 = (0.5 * x) - ((0.866) * y) + r
        w3 = r - x
        self.spin_wheels(int(w1), int(w2), int(w3))

    def turn(self, direction: vex.TurnType, velocity=None, units:vex.TurnVelocityPercentUnits=vex.TurnVelocityUnits.PERCENT):
        """turn indefinitely at velocity (DPS) in the direction specified by turn_direction (TurnType.LEFT or TurnType.RIGHT). 
        if velocity is not provided, use the default set by set_turn_velocity command"""
        if velocity is None:
            velocity = self.turn_speed
        else:
            if units.value == vex.TurnVelocityUnits.PERCENT.value:
                #cannot exeed 100%
                if velocity > 100:
                    velocity = 100
                #if velocity is PERCENT convert to DPS ex: 100% is 180 MMPS
                velocity = int(velocity * 1.8)
            elif units.value == vex.TurnVelocityUnits.DPS.value:
                #cannot exceed MAX velocity MMPS
                if velocity > TURN_VELOCITY_MAX_DPS:
                    velocity = TURN_VELOCITY_MAX_DPS
                velocity = int(velocity)
        #handle direction flip
        if direction == vex.TurnType.LEFT:
            velocity = -velocity
        stacking_type = vex.StackingType.STACKING_OFF
        message = commands.Turn(velocity, stacking_type.value)
        self.robot_send(message.to_json())

    def turn_for(self, direction: vex.TurnType, angle, velocity=None, units:vex.TurnVelocityPercentUnits=vex.TurnVelocityUnits.PERCENT, wait=True):
        """turn for a 'angle' number of degrees at turn_rate (deg/sec)"""
        if velocity is None:
            velocity = self.turn_speed
        else:
            if units.value == vex.TurnVelocityUnits.PERCENT.value:
                #cannot exeed 100%
                if velocity > 100:
                    velocity = 100
                #if velocity is PERCENT convert to DPS ex: 100% is 180 MMPS
                velocity = int(velocity * 1.8)
            elif units.value == vex.TurnVelocityUnits.DPS.value:
                #cannot exceed MAX velocity MMPS
                if velocity > TURN_VELOCITY_MAX_DPS:
                    velocity = TURN_VELOCITY_MAX_DPS
                velocity = int(velocity)
        #handle direction flip
        if direction == vex.TurnType.LEFT:
            angle = -angle
        stacking_type = vex.StackingType.STACKING_OFF
        message = commands.TurnFor(angle, velocity, stacking_type.value)
        self.robot_send(message.to_json())
        if wait:
            self._block_on_state(self.is_turn_active)

    def turn_to(self, heading:vex.vexnumber, velocity=None, units:vex.TurnVelocityPercentUnits=vex.TurnVelocityUnits.PERCENT, wait=True):
        """turn to a heading (degrees) at velocity (deg/sec)\n
        heading can be -360 to 360"""
        if not (-360 < heading < 360):
            raise ValueError("heading must be between -360 and 360")
        if velocity is None:
            velocity = self.turn_speed
        else:
            if units.value == vex.TurnVelocityUnits.PERCENT.value:
                #cannot exeed 100%
                if velocity > 100:
                    velocity = 100
                #if velocity is PERCENT convert to DPS ex: 100% is 180 MMPS
                velocity = int(velocity * 1.8)
            elif units.value == vex.TurnVelocityUnits.DPS.value:
                #cannot exceed MAX velocity MMPS
                if velocity > TURN_VELOCITY_MAX_DPS:
                    velocity = TURN_VELOCITY_MAX_DPS
                velocity = int(velocity)
        #handle negative velocity
        if velocity < 0:
            velocity = -velocity

        heading_offset = self.inertial.heading_offset
        heading = math.fmod (heading_offset + heading, 360)
        stacking_type = vex.StackingType.STACKING_OFF
        message = commands.TurnTo(heading, velocity, stacking_type.value)
        self.robot_send(message.to_json())
        if wait:
            self._block_on_state(self.is_turn_active)

    def stop_all_movement(self):
        """stops all movements of the robot"""
        self.move_at(0,0)
        self.turn(vex.TurnType.RIGHT, 0)
        # clear the is_moving_flag now (used by robot.is_stopped())
        self._ws_status_thread.clear_is_moving_flag()
        # trigger this flag to be cleared again next time a status msg is received, in case the robot hasn't update the state yet:
        self._ws_status_thread.is_moving_flag_needs_clearing = True
        self._ws_status_thread.is_moving_flag_needs_setting  = False

    def spin_wheels(self, velocity1: int, velocity2: int, velocity3: int):
        """spin all three wheels of the robot at the specified velocities"""
        message = commands.SpinWheels(velocity1, velocity2, velocity3)
        self.robot_send(message.to_json())

    def set_xy_position(self, x, y):
        """sets the robot’s current position to the specified values"""
        offset_radians = -math.radians(self.inertial.heading_offset)
        origin_x = x * math.cos(offset_radians) - y * math.sin(offset_radians)
        origin_y = y * math.cos(offset_radians) + x * math.sin(offset_radians)

        message = commands.SetPose(origin_x, origin_y)
        self.robot_send(message.to_json())

        for status_counter in range(2):
            heartbeat_state = self._ws_status_thread.heartbeat
            # wait till we get a new status packet
            while heartbeat_state == self._ws_status_thread.heartbeat:
                # print("status_counter %d" %(status_counter))
                time.sleep(0.050)

    #endregion Motion Commands

    #region Vision Commands
    def has_any_barrel(self):
        """returns true if a barrel is held by the kicker"""
        ai_objects = list(self.vision.get_data(AiVision.ALL_AIOBJS))
        has_barrel = False
        for object in range(len(ai_objects)):
            cx = ai_objects[object].originX + ai_objects[object].width/2
            if ai_objects[object].classname in ["BlueBarrel", "OrangeBarrel"] and \
               BARREL_MIN_CX < cx < BARREL_MAX_CX and \
               ai_objects[object].originY > BARREL_MIN_Y:
                has_barrel = True

        return has_barrel

    def has_blue_barrel(self):
        """returns true if a barrel is held by the kicker"""
        ai_objects = list(self.vision.get_data(AiVision.ALL_AIOBJS))
        has_barrel = False
        for object in range(len(ai_objects)):
            cx = ai_objects[object].originX + ai_objects[object].width/2
            if ai_objects[object].classname in ["BlueBarrel"] and \
               BARREL_MIN_CX < cx < BARREL_MAX_CX and \
               ai_objects[object].originY > BARREL_MIN_Y:
                has_barrel = True
        return has_barrel

    def has_orange_barrel(self):
        """returns true if a barrel is held by the kicker"""
        ai_objects = list(self.vision.get_data(AiVision.ALL_AIOBJS))
        has_barrel = False
        for object in range(len(ai_objects)):
            cx = ai_objects[object].originX + ai_objects[object].width/2
            if ai_objects[object].classname in ["OrangeBarrel"] and \
               BARREL_MIN_CX < cx < BARREL_MAX_CX and \
               ai_objects[object].originY > BARREL_MIN_Y :
                has_barrel = True
        return has_barrel

    def has_sports_ball(self):
        """returns true if a ball is held by the kicker"""
        ai_objects = list(self.vision.get_data(AiVision.ALL_AIOBJS))
        has_ball = False
        for object in range(len(ai_objects)):
            cx = ai_objects[object].originX + ai_objects[object].width/2
            if ai_objects[object].classname in ["SportsBall"] and \
               BALL_MIN_CX < cx < BALL_MAX_CX and \
               ai_objects[object].originY > BALL_MIN_Y :
                has_ball = True
        return has_ball
    #endregion Vision Commands
class Inertial():
    """ AIM Inertial class.  Provides methods to interact with the robot's inertial sensor."""
    def __init__(self, robot_instance: Robot):
        """Initialize the Gyro with default settings."""
        self.robot_instance = robot_instance
        self.heading_offset = 0
        self.rotation_offset = 0
    #region Inertial - Action
    def calibrate(self):
        """calibrate the IMU.  Can't check if calibration is done, so probably do not call for now"""
        message = commands.InterialCalibrate()
        self.robot_instance.robot_send(message.to_json())
    #endregion Inertial - Action
    #region Inertial - Mutators
    def set_heading(self, heading):
        """sets the robot’s heading to a specified value"""
        raw_heading = self.get_heading_raw()
        # print("reset heading to %f, get_heading_raw(): %f" %(heading, raw_heading))
        self.heading_offset = raw_heading - heading

    def reset_heading(self):
        """robot heading will be set to 0"""
        self.set_heading(0)

    def set_rotation(self, rotation):
        """sets the robot’s rotation to a specified value"""
        raw_rotation = self.get_rotation_raw()
        self.rotation_offset = raw_rotation - rotation

    def reset_rotation(self):
        """resets the robot’s rotation to 0"""
        self.set_rotation(0)

    def set_crash_sensitivity(self, sensitivity=vex.SensitivityType.LOW):
        """set the sensitivity of the crash sensor"""
        message = commands.InterialSetCrashSensitivity(sensitivity.value)
        self.robot_instance.robot_send(message.to_json())
    #endregion Inertial - Mutators
    #region Inertial - Getters
    def get_heading(self):
        """reports the robot’s heading angle. This returns a float in the range 0 to 359.99 degrees"""
        raw_heading = self.get_heading_raw()
        heading = math.fmod(raw_heading - self.heading_offset, 360)
        #round to 2 decimal places
        heading = round(heading, 2)
        if heading < 0:
            heading += 360
        return heading

    def get_heading_raw(self):
        """reports raw heading value from IMU"""
        raw_heading = self.robot_instance.status["robot"]["heading"]
        if type(raw_heading) == str:
            raw_heading = float(raw_heading)
        return raw_heading

    def get_rotation(self):
        """returns the robot’s total rotation in degrees as a float. 
        This measures how much the robot has rotated relative to its last reset point"""
        raw_rotation = self.get_rotation_raw()
        rotation = raw_rotation - self.rotation_offset
        #round to 2 decimal places
        rotation = round(rotation, 2)
        return rotation

    def get_rotation_raw(self):
        """reports raw rotation value from IMU"""
        raw_rotation = self.robot_instance.status["robot"]["rotation"]
        if type(raw_rotation) == str:
            raw_rotation = float(raw_rotation)
        return raw_rotation

    def get_acceleration(self, axis: Union[vex.AxisType, vex.AccelerationType]):
        """ returns the robot’s acceleration for a given axis."""
        if axis in [vex.AxisType.X_AXIS, vex.AccelerationType.FORWARD]:
            value = self.robot_instance._ws_status_thread.current_status["robot"]["acceleration"]["x"]
        elif axis in [vex.AxisType.Y_AXIS, vex.AccelerationType.RIGHTWARD]:
            value = self.robot_instance._ws_status_thread.current_status["robot"]["acceleration"]["y"]
        elif axis in [vex.AxisType.X_AXIS, vex.AccelerationType.DOWNWARD]:
            value = self.robot_instance._ws_status_thread.current_status["robot"]["acceleration"]["z"]
        if type(value) == str:
            value = float(value)
        return value

    def get_turn_rate(self, axis: Union[vex.AxisType, vex.OrientationType]):
        """returns the robot’s gyro rate for a given axis in degrees per second (DPS). It returns a float from –1000.00 to 1000.00 degrees per second"""
        if axis in [vex.AxisType.X_AXIS, vex.OrientationType.ROLL]:
            value = self.robot_instance._ws_status_thread.current_status["robot"]["gyro_rate"]["x"]
        elif axis in [vex.AxisType.Y_AXIS, vex.OrientationType.PITCH]:
            value = self.robot_instance._ws_status_thread.current_status["robot"]["gyro_rate"]["y"]
        elif axis in [vex.AxisType.Z_AXIS, vex.OrientationType.YAW]:
            value = self.robot_instance._ws_status_thread.current_status["robot"]["gyro_rate"]["z"]
        if type(value) == str:
            value = float(value)
        return value

    def get_roll(self):
        """returns the robot’s roll angle in the range –180.00 to 180.00 degrees as a float"""
        value = self.robot_instance.status["robot"]["roll"]
        if type(value) == str:
            value = float(value)
        #round to 2 decimal places
        value = round(value, 2)
        return value

    def get_pitch(self):
        """returns the robot’s pitch angle in the range –90.00 to 90.00 degrees as a float"""
        value = self.robot_instance.status["robot"]["pitch"]
        if type(value) == str:
            value = float(value)
        #round to 2 decimal places
        value = round(value, 2)
        return value

    def get_yaw(self):
        """returns the robot’s yaw angle in the range –180.00 to 180.00 degrees as a float"""
        value = self.robot_instance.status["robot"]["yaw"]
        if type(value) == str:
            value = float(value)
        #round to 2 decimal places
        value = round(value, 2)
        return value

    def is_calibrating(self):
        """reports whether the gyro is calibrating."""
        if self.robot_instance._ws_status_thread.imu_cal_flag_needs_setting == True:
            return True
        robot_flags = self.robot_instance._ws_status_thread.current_status["robot"]["flags"]
        calibrating = bool(int(robot_flags, 16) & SYS_FLAGS_IMU_CAL)
        return calibrating
    #endregion Inertial - Getters
    #region Inertial - Callbacks
    def crashed(self, callback: Callable[...,None], arg: tuple=()):
        """calls the specified function when the robot crashes"""
        self.robot_instance.add_inertial_crash_callback(callback, arg)
    #endregion Inertial - Callbacks

class Screen():
    """ 
    Screen class for accessing the robot's screen features 
    like drawing shapes,text,and showing emojis
    """
    def __init__(self, robot_instance: Robot):
        self.robot_instance = robot_instance
        # store r,b,g values of default fill color
        fill_r,fill_g,fill_b = self._return_rgb(vex.Color.BLUE)
        self.default_fill_color = ColorRGB(fill_r,fill_g,fill_b)
        pen_r,pen_g,pen_b = self._return_rgb(vex.Color.BLUE)
        self.default_pen_color = ColorRGB(pen_r,pen_g,pen_b)

    def _return_rgb(self, color):
        if isinstance(color, (vex.Color, vex.Color.DefinedColor)):
            r = (color.value >> 16) & 0xFF
            g = (color.value >>  8) & 0xFF
            b =  color.value & 0xFF
        elif isinstance(color, int):
            r = (color >> 16) & 0xFF
            g = (color >>  8) & 0xFF
            b =  color & 0xFF
        else:
            raise AimException("parameter must be a vex.Color instance or int rgb value")
        return r,g,b

    def _return_transparency(self, color):
        if isinstance(color, (vex.Color, vex.Color.DefinedColor)):
            return color.transparent
        return False

    #region Screen - Cursor Print

    def print(self,  *args, **kwargs):
        """displays text on the robot’s screen at the current cursor position and font"""
        try:
            out=io.StringIO()
            if 'end' not in kwargs:
                kwargs['end'] = ""
            print(*args,**kwargs, file=out)
            text = out.getvalue()
            #print("text: ", text)
            out.close()
            message = commands.ScreenPrint(text)
            self.robot_instance.robot_send(message.to_json())
        except Exception as e:
            print(f"Error displaying text on screen: {e}")

    def set_cursor(self, row, column):
        """sets the cursor’s (row, column) screen position"""
        message = commands.ScreenSetCursor(row, column)
        self.robot_instance.robot_send(message.to_json())

    def next_row(self):
        """moves the cursor to the next row"""
        message = commands.ScreenNextRow()
        self.robot_instance.robot_send(message.to_json())

    def clear_row(self, row: int, color=vex.Color.BLUE):
        """clears a row of text."""
        r,g,b = self._return_rgb(color)
        message = commands.ScreenClearRow(row, r, g, b)
        self.robot_instance.robot_send(message.to_json())

    def get_row(self):
        """returns the current row of the cursor"""
        value = self.robot_instance.status["robot"]["screen"]["row"]
        if type(value) == str:
            value = int(value)
        return value

    def get_column(self):
        """returns the current column of the cursor"""
        value = self.robot_instance.status["robot"]["screen"]["column"]
        if type(value) == str:
            value = int(value)
        return value

    #endregion Screen - Cursor Print
    #region Screen - XY Print
    def print_at(self, *args, x=0, y=0, **kwargs):
        """displays text on the robot’s screen at a specified (x, y) screen coordinate. This method disregards the current cursor position"""
        out=io.StringIO()
        if 'end' not in kwargs:
            kwargs['end'] = ""
        print(*args,**kwargs, file=out)
        text = out.getvalue()
        #print("text: ", text)
        out.close()
        message = commands.ScreenPrintAt(text, x, y, True)
        self.robot_instance.robot_send(message.to_json())

    def set_origin(self, x, y):
        """sets the origin of the screen to the specified (x, y) screen coordinate"""
        message = commands.ScreenSetOrigin(x, y)
        self.robot_instance.robot_send(message.to_json())
    #endregion Screen - XY Print
    #region Screen - Mutators

    def clear_screen(self, color=vex.Color.BLUE):
        """clears the robot’s screen of all drawings and text"""
        r,g,b = self._return_rgb(color)
        message = commands.ScreenClear(r, g, b)
        self.robot_instance.robot_send(message.to_json())

    def set_font(self, fontname: vex.FontType):
        """sets the font used for displaying text on the robot’s screen"""
        message = commands.ScreenSetFont(fontname.lower())
        self.robot_instance.robot_send(message.to_json())

    def set_pen_width(self, width: int):
        """sets the pen width used for drawing lines and shapes"""
        message = commands.ScreenSetPenWidth(width)
        self.robot_instance.robot_send(message.to_json())

    def set_pen_color(self, color):
        """sets the pen color used for drawing lines, shapes, and text"""
        r,g,b = self._return_rgb(color)
        self.default_pen_color = ColorRGB(r,g,b)
        message = commands.ScreenSetPenColor(r, g, b)
        self.robot_instance.robot_send(message.to_json())

    def set_fill_color(self, color):
        """sets the fill color used when shapes are drawn"""
        r,g,b = self._return_rgb(color)
        t = self._return_transparency(color)
        self.default_fill_color = ColorRGB(r,g,b,t)
        message = commands.ScreenSetFillColor(r, g, b, t)
        self.robot_instance.robot_send(message.to_json())
    #endregion Screen - Mutators
    #region Screen - Draw
    def draw_pixel(self, x: int, y: int):
        """draws a pixel at the specified (x, y) screen coordinate in the current pen color"""
        message = commands.ScreenDrawPixel(x, y)
        self.robot_instance.robot_send(message.to_json())

    def draw_line(self, x1: int, y1: int, x2: int, y2: int):
        """draws a line from the first specified screen coordinate (x1, y1) 
        to the second specified screen coordinate (x2, y2). 
        It uses the current the pen width set by set_pen_width and pen color set by set_pen_color"""
        message = commands.ScreenDrawLine(x1, y1, x2, y2)
        self.robot_instance.robot_send(message.to_json())

    def draw_rectangle(self, x: int, y: int, width: int, height: int, color=None):
        """draws a rectangle. It uses the current pen width set by set_pen_width and the pen color 
        set by set_pen_color for the outline. The fill color, set by set_fill_color, determines the interior color"""
        #if color is not provided, use the default fill color
        if color:
            r,g,b = self._return_rgb(color)
            t = self._return_transparency(color)
        else:
            r = self.default_fill_color.r
            g = self.default_fill_color.g
            b = self.default_fill_color.b
            t = self.default_fill_color.t
        message = commands.ScreenDrawRectangle(x, y, width, height, r, g, b, t)
        self.robot_instance.robot_send(message.to_json())

    def draw_circle(self, x: int, y: int, radius: int, color=None):
        """draws a circle. It uses the current pen width set by set_pen_width and the pen color set by set_pen_color for the outline. 
        The fill color, set by set_fill_color, determines the interior color"""
         #if color is not provided, use the default fill color
        if color:
            r,g,b = self._return_rgb(color)
            t = self._return_transparency(color)
        else:
            r = self.default_fill_color.r
            g = self.default_fill_color.g
            b = self.default_fill_color.b
            t = self.default_fill_color.t
        message = commands.ScreenDrawCircle(x, y, radius, r, g, b, t)
        self.robot_instance.robot_send(message.to_json())

    def show_file(self, filename: str, x: int, y: int):
        """draws a custom user-uploaded image on the robot’s screen at the specified (x, y) screen coordinate"""
        #TODO: alowed extensions are correct?
        if filename[-3:] not in ("bmp", "png"):
            raise InvalidImageFileException(f"extension is {filename[-3:]}; expected extension to be bmp or png")

        message = commands.ScreenDrawImageFromFile(filename, x, y)
        self.robot_instance.robot_send(message.to_json())

    def set_clip_region(self, x: int, y: int, width: int, height: int):
        """ 
        defines a rectangular area on the screen where all drawings and text will be confined. Any content outside this region will not be displayed
        """
        message = commands.ScreenSetClipRegion(x, y, width, height)
        self.robot_instance.robot_send(message.to_json())
    #endregion Screen - Draw
    #region Screen - Emoji
    def show_emoji(self, emoji: vex.EmojiType.EmojiType, look: vex.EmojiLookType.EmojiLookType = vex.EmojiLookType.LOOK_FORWARD):
        """Show an emoji from a list of preset emojis"""
        message = commands.ScreenShowEmoji(emoji.value, look.value)
        self.robot_instance.robot_send(message.to_json())

    def hide_emoji(self):
        """hide emoji from being displayed, so that any underlying graphics can be viewed"""
        message = commands.ScreenHideEmoji()
        self.robot_instance.robot_send(message.to_json())
    #endregion Screen - Emoji
    #region Screen - Callbacks
    def pressed(self, callback: Callable[...,None], arg: tuple=()):
        """Calls the specified function when the screen is pressed, with optional args."""
        self.robot_instance.add_screen_pressed_callback(callback, arg)

    def released(self, callback: Callable[...,None], arg: tuple=()):
        """Calls the specified function when the screen is released, with optional args."""
        self.robot_instance.add_screen_released_callback(callback, arg)

    #endregion Screen - Callbacks
    #region Screen - Touch
    def pressing(self):
        """returns true if the screen is being pressed"""
        is_pressing = bool(int(self.robot_instance.status["robot"]["touch_flags"], 16) & 0x0001)
        return is_pressing

    def x_position(self):
        """returns the x position of the screen press"""
        touch_x = float(self.robot_instance.status["robot"]["touch_x"])
        return touch_x

    def y_position(self):
        """returns the y position of the screen press"""
        touch_y = float(self.robot_instance.status["robot"]["touch_y"])
        return touch_y
    #endregion Screen - Touch
    #region Screen - Vision
    def show_aivision(self):
        """show the aivision output on the screen"""
        message = commands.ScreenShowAivision()
        self.robot_instance.robot_send(message.to_json())

    def hide_aivision(self):
        """hide the aivision output"""
        message = commands.ScreenHideAivision()
        self.robot_instance.robot_send(message.to_json())
    #endregion Screen - Vision

class Kicker():
    """ Kicker class for accessing the robot's kicker features like kicking and pushing"""
    def __init__(self, robot_instance: Robot):
        self.robot_instance = robot_instance
        pass

    def kick(self, kick_type: vex.KickType):
        """activates the Kicker to kick an object with specified levels of force"""
        message = commands.KickerKick(kick_type)
        self.robot_instance.robot_send(message.to_json())

    def place(self):
        """activates the Kicker in order to place an object gently in front of the robot"""
        # json = {"cmd_id": "push"}
        self.kick(vex.KickType.SOFT)

class Sound():
    """ 
    Sound class for accessing the robot's sound features like playing built-in and uploaded sounds
    """
    def __init__(self, robot_instance: Robot):
        self.robot_instance = robot_instance
    def __note_to_midi(self, note_str):
        """
        Converts a musical note string (e.g., "C#5") into a MIDI note number (0-11) and octave.
        """
       # Check at least one character is present
        if len(note_str) < 1:
            raise TypeError("invalid note string")

        # Map first character to note
        c = note_str[0].lower()
        if c == 'c':
            note = 0
        elif c == 'd':
            note = 2
        elif c == 'e':
            note = 4
        elif c == 'f':
            note = 5
        elif c == 'g':
            note = 7
        elif c == 'a':
            note = 9
        elif c == 'b':
            note = 11
        else:
            raise TypeError("invalid note string")

        octave = 0

        # If length=2, second char should be the octave 5–8
        if len(note_str) == 2:
            if note_str[1] < '5' or note_str[1] > '8':
                raise TypeError("invalid note string")
            octave = int(note_str[1]) - 5
            if octave < 0:
                octave = 0

        # If length=3, middle char should be '#' or 'b', last char is octave 5–8
        elif len(note_str) == 3:
            if (note_str[2] < '5' or note_str[2] > '8' or (note_str[1] not in ('#', 'b'))):
                raise TypeError("invalid note string")
            accidental = note_str[1]
            if accidental == '#' and note < 11:
                note += 1
            elif accidental == 'b' and note > 0:
                note -= 1

            octave = int(note_str[2]) - 5
            if octave < 0:
                octave = 0
        else:
            raise TypeError("invalid note string")

        return note, octave

    def __set_sound_active(self):
        self.robot_instance._ws_status_thread.set_sound_playing_flag()
        self.robot_instance._ws_status_thread.set_sound_downloading_flag()
        self.robot_instance._ws_status_thread.sound_downloading_flag_needs_setting = True # have it be set again after next status message
        self.robot_instance._ws_status_thread.sound_playing_flag_needs_setting = True # have it be set again after next status message

    def play(self, sound: vex.SoundType, volume = 50):
        """plays one of the robot’s built-in sounds at a specified volume percentage. 
        Since this is a non-waiting method, the robot plays the built-in sound and moves to 
        the next command without waiting for it to finish"""
        message = commands.SoundPlay(sound.lower(), volume)
        self.robot_instance.robot_send(message.to_json())

    def play_file(self, name: str, volume = 50):
        """plays a custom sound loaded by the user at a specified volume percentage. \n
        Current uploading sounds to AIM is supported onnly the VEXcode AIM app.
        Since this is a non-waiting method, the robot plays the built-in sound and moves to 
        the next command without waiting for it to finish"""
        message = commands.SoundPlayFile(name, volume)
        self.robot_instance.robot_send(message.to_json())

    def play_local_file(self, filepath: str, volume = 100):
        """play a WAV or MP3 file stored on the client side; file will be transmitted to robot\n
           Maximum filesize is 255 KB"""
        file = pathlib.Path(filepath)
        size = file.stat().st_size
        if size > SOUND_SIZE_MAX_BYTES:
            raise InvalidSoundFileException(f"file size of {size} bytes is too big; max size allowed is {SOUND_SIZE_MAX_BYTES} bytes ({SOUND_SIZE_MAX_BYTES/1024:.1f} kB)")

        extension = file.suffix
        filename = file.name
        audio = bytearray(64)
        if not (extension == ".wav" or extension == ".mp3"):
            raise InvalidSoundFileException(f"extension is {extension}; expected extension to be wav or mp3")
        try:
            f = open(filepath, 'rb')
        except FileNotFoundError:
            print ("File", filepath, "was not found")
        else:
            with f:
                data = f.read()

            # do some sanity checks to make sure it's really a wave:
            if extension == ".wav":
                if not (data[0:4] == b'RIFF' and data[8:12] == b'WAVE'):
                    raise InvalidSoundFileException("file extension was .wav but does not appear to actually be a WAVE file")
                channels = int.from_bytes(data[22:24], "little")
                if channels > 2:
                    raise InvalidSoundFileException(f"only mono or stereo is supported, detected {channels} channels.")
                if channels == 2:
                    print("%s is stereo; mono is recommended")
                # first 64 bytes of audio is header
                audio[0:1] = (0).to_bytes(1, 'little')

            # assuming the mp3 is valid:
            elif extension == ".mp3":
                audio[0:1] = (1).to_bytes(1, 'little')

            # set volume
            audio[1:2] = (volume).to_bytes(1, 'little')

            audio[4:8] = (len(data)).to_bytes(4, 'little') # length of data
            audio[8:12] = (0).to_bytes(4, 'little') # file chunk number
            audio[32:32+len(filename)] = map(ord, filename[:32]) # filename
            audio.extend(data) # append the data
            self.robot_instance.robot_send_audio(audio)
            self.__set_sound_active()

    def play_note(self, note_string: str, duration=750, volume=50):
        """ 
        plays a specific note for a specific duration. . Since this is a non-waiting method, the robot plays 
        the specific note and moves to the next command without waiting for it to finish.

        ### Example:
        robot.sound.play_note("C5", 2000)
        robot.sound.play_note("F#6", 2000, 100)
        """
        #get the note number and octave
        note, octave = self.__note_to_midi(note_string)
        if duration > 4000:
            duration = 4000
        if volume > 100:
            volume = 100
        if volume < 0:
            volume = 0
        message = commands.SoundPlayNote(note, octave, duration, volume)
        self.robot_instance.robot_send(message.to_json())
        self.__set_sound_active()

    def is_active(self):
        """returns true if sound is currently playing or if it is being transmitted for playing"""
        robot_flags = self.robot_instance.status["robot"]["flags"]
        sound_active = bool(int(robot_flags, 16) & SYS_FLAGS_SOUND_PLAYING) or bool(int(robot_flags, 16) & SYS_FLAGS_IS_SOUND_DNL)
        return sound_active

    def stop(self):
        """
        stops a sound that is currently playing.  
        It will take some time for the sound to actually stop playing.
        """
        message = commands.SoundStop()
        self.robot_instance.robot_send(message.to_json())

class Led():
    """ Led class for accessing the robot's LED features like setting the color of the LEDs"""
    def __init__(self, robot_instance: Robot):
        self.robot_instance = robot_instance
        pass
    def __set_led_rgb(self, led: str, r: int, g: int, b: int):
        """Turns on the specified LED with the specified RGB values"""
        message = commands.LedSet(led, r, g, b)
        self.robot_instance.robot_send(message.to_json())
    def on(self, *args):
        """
        Sets the color of any one of six LEDs, with RGB values \n
        ### Example:
        robot.led.on(vex.LightType.ALL_LEDS, vex.Color.BLUE)\n
        robot.led.on(1, vex.Color.BLUE)
        """
        light_index = "all"
        r, g, b = 0, 0, 0
        if len(args) not in [2, 4]:
            raise TypeError("must have two or four arguments")

        if isinstance(args[0], int):
            if args[0] in range(0,6):
                light_index = f"light{args[0]+1}"
            else:
                light_index = "all"
        elif isinstance(args[0], vex.LightType):
            light_index = args[0]

        else:
            raise TypeError("first argument must be of type int or vex.LightType")

        if len(args) == 2:
            if isinstance(args[1], (vex.Color, vex.Color.DefinedColor)):
                r = (args[1].value >> 16) & 0xFF
                g = (args[1].value >>  8) & 0xFF
                b =  args[1].value & 0xFF
            elif isinstance(args[1], (bool)): # turn white if True, off if False
                r = 128 if args[1] else 0
                g = 128 if args[1] else 0
                b = 128 if args[1] else 0
            elif args[1] is None:
                r = 0
                g = 0
                b = 0
            else:
                raise TypeError("second argument must be of type vex.Color, vex.Color.DefinedColor, bool, or None")

        elif len(args) == 4:
            r = args[1]
            g = args[2]
            b = args[3]

        else:
            raise TypeError(f"bad parameters, n_args: {len(args)}")

        self.__set_led_rgb(light_index, r, g, b)

    def off(self, led: vex.LightType):
        """turns off the specified LED"""
        message = commands.LedSet(led, 0, 0, 0)
        self.robot_instance.robot_send(message.to_json())

class Colordesc:
    '''### Colordesc class - a class for holding an AI vision sensor color definition

    #### Arguments:
        index : The color description index (1 to 7)
        red : the red color value
        green : the green color value
        blue : the blue color value
        hangle : the range of allowable hue
        hdsat : the range of allowable saturation

    #### Returns:
        An instance of the Colordesc class

    #### Examples:
        COL1 = Colordesc(1,  13, 114, 227, 10.00, 0.20)\\
        COL2 = Colordesc(2, 237,  61,  74, 10.00, 0.20)\\
    '''
    def __init__(self, index, red, green, blue, hangle, hdsat):
        self.id = index
        self.red = red
        self.green = green
        self.blue = blue
        self.hangle = hangle
        self.hdsat = hdsat
        pass

class Codedesc:
    '''### Codedesc class - a class for holding AI vision sensor codes

    A code description is a collection of up to five AI vision color descriptions.
    #### Arguments:
        index : The code description index (1 to 5)
        c1 : An AI vision Colordesc
        c1 : An AI vision Colordesc
        c3 (optional) : An AI vision Colordesc
        c4 (optional) : An AI vision Colordesc
        c5 (optional) : An AI vision Colordesc

    #### Returns:
        An instance of the Codedesc class

    #### Examples:
        COL1 = Colordesc(1,  13, 114, 227, 10.00, 0.20)\\
        COL2 = Colordesc(2, 237,  61,  74, 10.00, 0.20)\\
        C1 = Codedesc( 1, COL1, COL2 )
    '''
    def __init__(self, index, c1:Colordesc, c2:Colordesc, *args):
        self.id = index
        self.cols = [c1, c2]
        for arg in args:
            if isinstance(arg, Colordesc):
                self.cols.append(arg)

class Tagdesc:
    '''### Tagdesc class - a class for holding AI vision sensor tag id

    A tag description holds an apriltag id
    #### Arguments:
        id : The apriltag id (positive integer, not 0)

    #### Returns:
        An instance of the Tagdesc class

    #### Examples:
        T1 = Tagdesc( 23 )
    '''
    def __init__(self, index):
        self.id = index
        pass

    def __int__(self):
        return self.id

    def __eq__(self, other):
        if isinstance(other, Tagdesc):
            return self.id == other.id
        elif isinstance(other, int):
            return self.id == other
        return False

class AiObjdesc:
    '''### AiObjdesc class - a class for holding AI vision sensor AI object id

    A tag description holds an apriltag id
    #### Arguments:
        id : The AI Object (model) id (positive integer, not 0)

    #### Returns:
        An instance of the AiObjdesc class

    #### Examples:
        A1 = AiObjdesc( 2 )
    '''
    def __init__(self, index):
        self.id = index
        pass

    def __int__(self):
        return self.id

    def __eq__(self, other):
        if isinstance(other, AiObjdesc):
            return self.id == other.id
        elif isinstance(other, int):
            return self.id == other
        return False

class ObjDesc:
    """ 
    ObjDesc class - to represent any object type
    """
    def __init__(self, index):
        self.id = index

class _ObjectTypeMask:
    unkownObject = 0
    colorObject  = (1 << 0)
    codeObject   = (1 << 1)
    modelObject  = (1 << 2)
    tagObject    = (1 << 3)
    allObject    = (0x3F)

MATCH_ALL_ID = 0xFFFF
class AiVision():
    """ 
    AiVision class for accessing the robot's AI Vision Sensor features
    """
    ALL_TAGS = Tagdesc(MATCH_ALL_ID)
    '''A tag description for get_data indicating all tag objects to be returned'''
    ALL_COLORS = Colordesc(MATCH_ALL_ID, 0, 0, 0, 0, 0)
    '''A tag description for get_data indicating all color objects to be returned'''
    ALL_CODES = Codedesc(MATCH_ALL_ID, ALL_COLORS, ALL_COLORS)
    '''A tag description for get_data indicating all code objects to be returned'''
    ALL_AIOBJS = AiObjdesc(MATCH_ALL_ID)
    '''A tag description for get_data indicating all AI model objects to be returned'''
    ALL_OBJECTS = ObjDesc(MATCH_ALL_ID)
    '''A tag description for get_data indicating all objects to be returned'''
    ALL_AIOBJECTS = AiObjdesc(MATCH_ALL_ID)
    '''A description for get_data indicating all AI model objects to be returned'''

    def __init__(self, robot_instance: Robot):
        self.robot_instance     = robot_instance
        self._object_count_val  = 0
        self._largest_object    = None

    def get_data(self, type, count=8):
        '''### filters the data from the AI Vision Sensor frame to return a tuple. 
        The AI Vision Sensor can detect signatures that include pre-trained objects, AprilTags, or configured Colors and Color Codes.

        Color Signatures and Color Codes must be configured first in the AI Vision Utility in VEXcode before they can be used with this method.
        The tuple stores objects ordered from largest to smallest by width, starting at index 0. Each object’s properties can be accessed using its index. 
        An empty tuple is returned if no matching objects are detected.
        
        #### Arguments:
            type : A color, code or other object type
            count (optional) : the maximum number of objects to obtain.  default is 8.

        #### Returns:
            tuple of AiVisionObject, this will be an empty tuple if nothing is available.

        #### Examples:
            #### look for and return 1 object matching COL1
            objects = robot.vision.get_data(COL1)

            #### look for and return a maximum of 4 objects matching SIG_1
            objects = robot.vision.get_data(COL1, 4)

            #### return apriltag objects
            objects = robot.vision.get_data(ALL_TAGS, AIVISION_MAX_OBJECTS)
        '''
        match_tuple=None
        if isinstance(type, Colordesc):
            type_mask = _ObjectTypeMask.colorObject
            id = type.id
        elif isinstance(type, Codedesc):
            type_mask = _ObjectTypeMask.codeObject
            id = type.id
        elif isinstance(type, AiObjdesc):
            type_mask = _ObjectTypeMask.modelObject
            id = type.id
        elif isinstance(type, Tagdesc):
            type_mask = _ObjectTypeMask.tagObject
            id = type.id
        elif isinstance(type, ObjDesc):
            type_mask = _ObjectTypeMask.allObject
            id = type.id
        elif isinstance(type, tuple):
            match_tuple = type
            if not match_tuple:
                raise AimException("tuple passed to get_data is empty")
            type_mask = _ObjectTypeMask.allObject
        else:
            type_mask = _ObjectTypeMask.allObject # default value, changed to match uP by James
            id = type # assume the first argument is any object id including match all.

        if count > AIVISION_MAX_OBJECTS:
            count = AIVISION_MAX_OBJECTS

        objects = self.robot_instance.status["aivision"]["objects"]
        item_count = objects["count"]
        ai_object_list = [AiVisionObject() for item in range(item_count)]
        # first just extract everything we got from ws_status
        for item in range(item_count):
            ai_object_list[item].type       = objects["items"][item]["type"]
            ai_object_list[item].id         = objects["items"][item]["id"]
            ai_object_list[item].originX    = objects["items"][item]["originx"]
            ai_object_list[item].originY    = objects["items"][item]["originy"]
            ai_object_list[item].width      = objects["items"][item]["width"]
            ai_object_list[item].height     = objects["items"][item]["height"]
            ai_object_list[item].centerX    = int(ai_object_list[item].originX + (ai_object_list[item].width/2))
            ai_object_list[item].centerY    = int(ai_object_list[item].originY + (ai_object_list[item].height/2))

            if ai_object_list[item].type ==  _ObjectTypeMask.colorObject:
                ai_object_list[item].angle = objects["items"][item]["angle"] * 0.01

            if ai_object_list[item].type ==  _ObjectTypeMask.codeObject:
                ai_object_list[item].angle = objects["items"][item]["angle"] * 0.01

            if ai_object_list[item].type ==  _ObjectTypeMask.modelObject: #AI model objects can have a classname
                ai_object_list[item].classname  = self.robot_instance.status["aivision"]["classnames"]["items"][ai_object_list[item].id]["name"]
                ai_object_list[item].score = objects["items"][item]["score"]

            if ai_object_list[item].type ==  _ObjectTypeMask.tagObject:
                ai_object_list[item].tag.x = (objects["items"][item]["x0"],objects["items"][item]["x1"],objects["items"][item]["x2"],objects["items"][item]["x3"])
                ai_object_list[item].tag.y = (objects["items"][item]["y0"],objects["items"][item]["y1"],objects["items"][item]["y2"],objects["items"][item]["y3"])

            ai_object_list[item].rotation = ai_object_list[item].angle
            ai_object_list[item].area = ai_object_list[item].width * ai_object_list[item].height
            cx = ai_object_list[item].centerX
            cy = ai_object_list[item].centerY
            ai_object_list[item].bearing = -34.656 + (cx * 0.22539) + (cy * 0.011526) + (cx * cx * -0.000042011) + (cx * cy * 0.000010433) + (cy * cy * -0.00007073)

        # print("diagnostic: ai_object_list: ", ai_object_list)
        num_matches = 0
        sublist = []
        for item in range(item_count):
            match_found = False
            if match_tuple:
                # check all tuple members for a match
                for obj in match_tuple:
                    if isinstance(obj, Colordesc):
                        if ai_object_list[item].type == _ObjectTypeMask.colorObject and (ai_object_list[item].id == obj.id or MATCH_ALL_ID == obj.id):
                            match_found = True
                    elif isinstance(obj, Codedesc):
                        if ai_object_list[item].type == _ObjectTypeMask.codeObject and (ai_object_list[item].id == obj.id or MATCH_ALL_ID == obj.id):
                            match_found = True
                    elif isinstance(obj, AiObjdesc):
                        if ai_object_list[item].type == _ObjectTypeMask.modelObject and (ai_object_list[item].id == obj.id or MATCH_ALL_ID == obj.id):
                            match_found = True
                    elif isinstance(obj, Tagdesc):
                        if ai_object_list[item].type == _ObjectTypeMask.tagObject and (ai_object_list[item].id == obj.id or MATCH_ALL_ID == obj.id):
                            match_found = True
                    elif isinstance(obj, ObjDesc):
                        if ai_object_list[item].id == obj.id or MATCH_ALL_ID == obj.id:
                            match_found = True
                    else:
                        # assume obj is an int
                        if ai_object_list[item].id == int(obj) or MATCH_ALL_ID == int(obj):
                            match_found = True
            else:
                if ai_object_list[item].id == id or MATCH_ALL_ID == id:
                    match_found = True

            if ai_object_list[item].type & type_mask:
                if match_found:
                    num_matches += 1
                    #sort objects by object area in descending order
                    current_object_area = ai_object_list[item].height * ai_object_list[item].width
                    current_object_smallest = True
                    for i in range(len(sublist)):
                        if current_object_area >= (sublist[i].width * sublist[i].height):
                            sublist.insert(i, ai_object_list[item]) # insert item at position i of sublist.
                            current_object_smallest = False
                            break
                    if current_object_smallest:
                        sublist.append(ai_object_list[item]) #add to the end

        if num_matches > count:
            num_matches = count

        self._object_count_val = num_matches
        if sublist:
            self._largest_object = sublist[0]
        else:
            self._largest_object = None
        return sublist[:num_matches]

    def largest_object(self):
        '''### Request the largest object from the last get_data(...) call

        #### Arguments:
            None

        #### Returns:
            An AiVisionObject object or None if it does not exist
        '''
        return self._largest_object

    def object_count(self):
        '''### Request the number of objects found in the last get_data call

        #### Arguments:
            None

        #### Returns:
            The number of objects found in the last get_data call
        '''
        return self._object_count_val


    def tag_detection(self, enable: bool):
        '''### Enable or disable apriltag processing

        #### Arguments:
            enable : True or False

        #### Returns:
            None
        '''
        message = commands.VisionTagDetection(enable)
        self.robot_instance.robot_send(message.to_json())

    def color_detection(self, enable: bool, merge: bool = False):
        '''### Enable or disable color and code object processing

        #### Arguments:
            enable : True or False
            merge (optional) : True to enable merging of adjacent color detections

        #### Returns:
            None
        '''
        message = commands.VisionColorDetection(enable, merge)
        self.robot_instance.robot_send(message.to_json())

    def model_detection(self, enable: bool):
        '''### Enable or disable AI model object processing

        #### Arguments:
            enable : True or False

        #### Returns:
            None
        '''
        message = commands.VisionModelDetection(enable)
        self.robot_instance.robot_send(message.to_json())

    def color_description(self, desc: Colordesc):
        '''### set a new color description

        #### Arguments:
            desc: a color description

        #### Returns:
            None
        '''
        message = commands.VisionColorDescription(desc.id, desc.red, desc.green, desc.blue, desc.hangle, desc.hdsat)
        self.robot_instance.robot_send(message.to_json())

    def code_description(self, desc: Codedesc ):
        '''### set a new code description

        #### Arguments:
            desc: a code description

        #### Returns:
            None
        '''
        message = commands.VisionCodeDescription(desc.id, *desc.cols)
        self.robot_instance.robot_send(message.to_json())


    def get_camera_image(self):
        """
        returns a camera image; starts stream when first called; first image will take about 0.3 seconds to return.\n
        Subsequently, images will continually stream from robot and therefore will be immediately available.
        """
        if self.robot_instance._ws_img_thread._streaming == False:
            # print("starting the stream")
            start_time = time.time()
            time_elapsed = 0
            self.robot_instance._ws_img_thread.start_stream()
            while (self.robot_instance._ws_img_thread.image_list[self.robot_instance._ws_img_thread.current_image_index] == bytes(1) and time_elapsed < 0.5):
                time.sleep(0.01)
                time_elapsed = time.time() - start_time
        image = self.robot_instance._ws_img_thread.image_list[self.robot_instance._ws_img_thread.current_image_index]
        if image == bytes(1):
            raise NoImageException("no image was received")
        return image

class VisionObject:
    '''### VisionObject class - a class with some predefined objects for get_data'''
    SPORTS_BALL = AiObjdesc(0)
    '''A description for get_data indicating the AI ball objects to be returned'''
    BLUE_BARREL = AiObjdesc(1)
    '''A description for get_data indicating the AI blue barrel objects to be returned'''
    ORANGE_BARREL = AiObjdesc(2)
    '''A description for get_data indicating the AI orange barrel objects to be returned'''
    AIM_ROBOT = AiObjdesc(3)
    '''A description for get_data indicating the AI robot objects to be returned'''
    TAG0 = Tagdesc(0)
    '''A description for get_data indicating apriltags with id 0 to be returned'''
    TAG1 = Tagdesc(1)
    '''A description for get_data indicating apriltags with id 1 to be returned'''
    TAG2 = Tagdesc(2)
    '''A description for get_data indicating apriltags with id 2 to be returned'''
    TAG3 = Tagdesc(3)
    '''A description for get_data indicating apriltags with id 3 to be returned'''
    TAG4 = Tagdesc(4)
    '''A description for get_data indicating apriltags with id 4 to be returned'''
    TAG5 = Tagdesc(5)
    '''A description for get_data indicating apriltags with id 5 to be returned'''
    TAG6 = Tagdesc(6)
    '''A description for get_data indicating apriltags with id 6 to be returned'''
    TAG7 = Tagdesc(7)
    '''A description for get_data indicating apriltags with id 7 to be returned'''
    TAG8 = Tagdesc(8)
    '''A description for get_data indicating apriltags with id 8 to be returned'''
    TAG9 = Tagdesc(9)
    '''A description for get_data indicating apriltags with id 9 to be returned'''
    TAG10 = Tagdesc(10)
    '''A description for get_data indicating apriltags with id 10 to be returned'''
    TAG11 = Tagdesc(11)
    '''A description for get_data indicating apriltags with id 11 to be returned'''
    TAG12 = Tagdesc(12)
    '''A description for get_data indicating apriltags with id 12 to be returned'''
    TAG13 = Tagdesc(13)
    '''A description for get_data indicating apriltags with id 13 to be returned'''
    TAG14 = Tagdesc(14)
    '''A description for get_data indicating apriltags with id 14 to be returned'''
    TAG15 = Tagdesc(15)
    '''A description for get_data indicating apriltags with id 15 to be returned'''
    TAG16 = Tagdesc(16)
    '''A description for get_data indicating apriltags with id 16 to be returned'''
    TAG17 = Tagdesc(17)
    '''A description for get_data indicating apriltags with id 17 to be returned'''
    TAG18 = Tagdesc(18)
    '''A description for get_data indicating apriltags with id 18 to be returned'''
    TAG19 = Tagdesc(19)
    '''A description for get_data indicating apriltags with id 19 to be returned'''
    TAG20 = Tagdesc(20)
    '''A description for get_data indicating apriltags with id 20 to be returned'''
    TAG21 = Tagdesc(21)
    '''A description for get_data indicating apriltags with id 21 to be returned'''
    TAG22 = Tagdesc(22)
    '''A description for get_data indicating apriltags with id 22 to be returned'''
    TAG23 = Tagdesc(23)
    '''A description for get_data indicating apriltags with id 23 to be returned'''
    TAG24 = Tagdesc(24)
    '''A description for get_data indicating apriltags with id 24 to be returned'''
    TAG25 = Tagdesc(25)
    '''A description for get_data indicating apriltags with id 25 to be returned'''
    TAG26 = Tagdesc(26)
    '''A description for get_data indicating apriltags with id 26 to be returned'''
    TAG27 = Tagdesc(27)
    '''A description for get_data indicating apriltags with id 27 to be returned'''
    TAG28 = Tagdesc(28)
    '''A description for get_data indicating apriltags with id 28 to be returned'''
    TAG29 = Tagdesc(29)
    '''A description for get_data indicating apriltags with id 29 to be returned'''
    TAG30 = Tagdesc(30)
    '''A description for get_data indicating apriltags with id 30 to be returned'''
    TAG31 = Tagdesc(31)
    '''A description for get_data indicating apriltags with id 31 to be returned'''
    TAG32 = Tagdesc(32)
    '''A description for get_data indicating apriltags with id 32 to be returned'''
    TAG33 = Tagdesc(33)
    '''A description for get_data indicating apriltags with id 33 to be returned'''
    TAG34 = Tagdesc(34)
    '''A description for get_data indicating apriltags with id 34 to be returned'''
    TAG35 = Tagdesc(35)
    '''A description for get_data indicating apriltags with id 35 to be returned'''
    TAG36 = Tagdesc(36)
    '''A description for get_data indicating apriltags with id 36 to be returned'''
    TAG37 = Tagdesc(37)
    '''A description for get_data indicating apriltags with id 37 to be returned'''
    ALL_TAGS = Tagdesc(MATCH_ALL_ID)
    '''A description for get_data indicating any apriltag to be returned'''
    ALL_VISION = ObjDesc(MATCH_ALL_ID)
    '''A description for get_data indicating any object to be returned'''
    ALL_COLORS = (AiVision.ALL_COLORS, AiVision.ALL_CODES)
    '''A description for get_data indicating any color or code to be returned'''
    ALL_CARGO = (SPORTS_BALL, BLUE_BARREL, ORANGE_BARREL)
    '''A description for get_data indicating AI ball or barrel to be returned'''

class AiVisionObject():
    """ 
    AiVisionObject class - a class for holding AI vision sensor object properties
    """
    class Tag:
        """
        Tag class - a class for holding AI vision sensor tag properties
        The tag class is used to hold the coordinates of the four corners of the tag
        """
        def __init__(self):
            self.x = (0,0,0,0)
            self.y = (0,0,0,0)
            pass

    def __init__(self):
        self.type      = 0
        self.id        = 0
        self.originX   = 0
        self.originY   = 0
        self.centerX   = 0
        self.centerY   = 0
        self.width     = 0
        self.height    = 0
        self.exists    = True
        self.angle     = 0.0
        self.rotation  = 0.0
        self.score     = 0
        self.area      = 0
        self.bearing   = 0.0
        self.classname = ''
        self.color     = None # not used in remote
        self.tag       = AiVisionObject.Tag()

class Timer:
    '''### Timer class - create a new timer

    This class is used to create a new timer\\
    A timer can be used to measure time, access the system time and run a function at a time in the future.
    
    #### Arguments:
        None

    #### Returns:
        An instance of the Timer class

    #### Examples:
        t1 = Timer()
    '''
    def __init__(self):
        self.start_time = time.time()

    def time(self, units=vex.TimeUnits.MSEC):
        '''### return the current time for this timer

        #### Arguments:
            units (optional) : the units that the time should be returned in, default is MSEC

        #### Returns:
            An the current time in specified units.

        #### Examples:
        '''
        elapsed_time = time.time() - self.start_time
        if units == vex.TimeUnits.SECONDS:
            # seconds as float in 2 decimal places
            return round(elapsed_time, 2)
        elif units == vex.TimeUnits.MSEC:
            # miliseconds as int - no decimal
            return int(elapsed_time * 1000)
        else:
            raise ValueError("Invalid time unit")

    def reset(self):
        '''### reset the timer to 0

        #### Arguments:
            None

        #### Returns:
            None

        #### Examples:
        '''
        self.start_time = time.time()

    def event(self, callback: Callable[...,None], delay: int, arg: tuple=()):
        '''### register a function to be called in the future

        #### Arguments:
            callback : A function that will called after the supplied delay
            delay : The delay before the callback function is called.
            arg (optional) : A tuple that is used to pass arguments to the function.

        #### Returns:
            None

        #### Examples:
            def foo(arg):
                print('timer has expired ', arg)

            t1 = Timer()\\
            t1.event(foo, 1000, ('Hello',))
        '''
        def delayed_call():
            time.sleep(delay / 1000)
            callback(*arg)

        threading.Thread(target=delayed_call).start()

class Thread():
    """ Thread class for running a function in a separate thread"""
    def __init__(self, func, args=None):
        if args:
            self.t = threading.Thread(target=func, args=args)
        else:
            self.t = threading.Thread(target=func)
        self.t.start()
    