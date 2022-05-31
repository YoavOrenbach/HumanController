import time
import ctypes

# scan codes: http://www.flint.jp/misc/?q=dik&lang=en
keycodes = {
    'Esc': 0x01,
    '1': 0x02,
    '2': 0x03,
    '3': 0x04,
    '4': 0x05,
    '5': 0x06,
    '6': 0x07,
    '7': 0x08,
    '8': 0x09,
    '9': 0x0A,
    '0': 0x0B,
    '-': 0x0C,
    '=': 0x0D,
    'Backspace': 0x0E,
    'Tab': 0x0F,
    'Q': 0x10,
    'W': 0x11,
    'E': 0x12,
    'R': 0x13,
    'T': 0x14,
    'Y': 0x15,
    'U': 0x16,
    'I': 0x17,
    'O': 0x18,
    'P': 0x19,
    '[': 0x1A,
    ']': 0x1B,
    'Enter': 0x1C,
    'LCtrl': 0x1D,  # left Ctrl
    'A': 0x1E,
    'S': 0x1F,
    'D': 0x20,
    'F': 0x21,
    'G': 0x22,
    'H': 0x23,
    'J': 0x24,
    'K': 0x25,
    'L': 0x26,
    ';': 0x27,
    "'": 0x28,
    '`': 0x29,
    'LShift': 0x2A,  # left Shift
    'backslash': 0x2B,
    'Z': 0x2C,
    'X': 0x2D,
    'C': 0x2E,
    'V': 0x2F,
    'B': 0x30,
    'N': 0x31,
    'M': 0x32,
    ',': 0x33,
    'period': 0x34,
    'slash': 0x35,
    'RShift': 0x36,
    'multiply': 0x37,  # * (Numpad)
    'LAlt': 0x38,  # left Alt
    'Space': 0x39,
    'Caps': 0x3A,
    'F1': 0x3B,
    'F2': 0x3C,
    'F3': 0x3D,
    'F4': 0x3E,
    'F5': 0x3F,
    'F6': 0x40,
    'F7': 0x41,
    'F8': 0x42,
    'F9': 0x43,
    'F10': 0x44,
    'NumLock': 0x45,
    'ScrollLock': 0x46,
    '7Numpad': 0x47,
    '8Numpad': 0x48,
    '9Numpad': 0x49,
    'subtract': 0x4A,  # - numpad
    '4Numpad': 0x4B,
    '5Numpad': 0x4C,
    '6Numpad': 0x4D,
    'add': 0x4E,  # + numpad
    '1Numpad': 0x4F,
    '2Numpad': 0x50,
    '3Numpad': 0x51,
    '0Numpad': 0x52,
    'DecimalNumpad': 0x53,
    'F11': 0x57,
    'F12': 0x58,
    'EnterNumpad': 0x9C,
    'RCtrl': 0x9D,
    'Divide': 0xB5,
    'PrintScreen': 0xB7,
    'RAlt': 0xB8,
    'Pause': 0xC5,
    'Home': 0xC7,
    'Up': 0xC8,
    'PageUp': 0xC9,
    'Left': 0xCB,
    'Right': 0xCD,
    'End': 0xCF,
    'Down': 0xD0,
    'PageDown': 0xD1,
    'Insert': 0xD2,
    'Delete': 0xD3,
    'LWin': 0xDB,  # left win
    'RWin': 0xDC,  # right win
    'menu': 0xDD
}


mouse_codes = {
    'mouse-left-click': (0x0002, 0x0004),
    'mouse-right-click': (0x0008, 0x0010),
    'mouse-middle-click': (0x0020, 0x0040)
}


wheel_codes = {
    'mouse-wheel-up': 100,
    'mouse-wheel-down': -100
}


movement_codes = {
    'mouse-move-up': (0, -100),
    'mouse-move-down': (0, 100),
    'mouse-move-right': (100, 0),
    'mouse-move-left': (-100, 0)
}


SendInput = ctypes.windll.user32.SendInput
# C struct redefinitions
PUL = ctypes.POINTER(ctypes.c_ulong)


class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]


class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("dwData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]


# Actual Functions
def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def mouse_click(code):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(0, 0, 0, code, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(0), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def wheel_movement(move):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(0, 0, move, 0x0800, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(0), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def move_mouse(x, y):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    #x = int(x*(65536/ctypes.windll.user32.GetSystemMetrics(0))+1)
    #y = int(y*(65536/ctypes.windll.user32.GetSystemMetrics(1))+1)
    ii_.mi = MouseInput(x, y, 0, 0x0001, 0, ctypes.pointer(extra))
    cmd = Input(ctypes.c_ulong(0), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(cmd), ctypes.sizeof(cmd))


def get_numlock_state():
    hllDll = ctypes.WinDLL("User32.dll")
    VK_CAPITAL = 0x90
    return hllDll.GetKeyState(VK_CAPITAL)
