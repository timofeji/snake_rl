# --- ANSI Color Codes ---
import os
import sys

GLYPH_SNAKE_HEAD = "◇"
GLYPH_SNAKE_BODY = "■"
GLYPH_FOOD = "🍉"
GLYPH_WALL_H = "═"
GLYPH_WALL_V = "║"
GLYPH_CORNER_TL = "╔"
GLYPH_CORNER_TR = "╗"
GLYPH_CORNER_BL = "╚"
GLYPH_CORNER_BR = "╝"


COLOR_RESET = '\033[0m'
COLOR_BOLD = '\033[1m'

COLOR_BLACK = '\033[30m'
COLOR_RED = '\033[31m'
COLOR_GREEN = '\033[32m'
COLOR_YELLOW = '\033[33m'
COLOR_BLUE = '\033[34m'
COLOR_MAGENTA = '\033[35m'
COLOR_CYAN = '\033[36m'
COLOR_WHITE = '\033[37m'

COLOR_BLACK_ALT = '\033[90m'
COLOR_RED_ALT = '\033[91m'
COLOR_GREEN_ALT = '\033[92m'
COLOR_YELLOW_ALT = '\033[93m'
COLOR_BLUE_ALT = '\033[94m'
COLOR_MAGENTA_ALT = '\033[95m'
COLOR_CYAN_ALT = '\033[96m'
COLOR_WHITE_ALT = '\033[97m'

# Background colors
COLOR_BLACK_BG = '\033[40m'
COLOR_RED_BG = '\033[41m'
COLOR_GREEN_BG = '\033[42m'
COLOR_YELLOW_BG = '\033[43m'
COLOR_BLUE_BG = '\033[44m'
COLOR_MAGENTA_BG = '\033[45m'
COLOR_CYAN_BG = '\033[46m'
COLOR_WHITE_BG = '\033[47m'


COLOR_FOOD = COLOR_MAGENTA + COLOR_BLACK_BG + COLOR_BOLD
COLOR_WALL = COLOR_RED + COLOR_BLACK_BG + COLOR_BOLD
COLOR_SNAKE = COLOR_GREEN + COLOR_BLACK_BG + COLOR_BOLD

# Platform-specific imports for non-blocking keyboard input
try:
    # Windows
    import msvcrt
except ImportError:
    # Unix-like systems
    import tty
    import termios
    import select

# Enable ANSI color support on Windows
if os.name == 'nt':
    os.system('')  # This enables ANSI escape sequences on Windows 10+


def clear_screen():
    """Clears the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def get_key_press():
    """
    Gets a single character from the user without waiting for Enter.
    This function is platform-dependent.
    """
    if 'msvcrt' in sys.modules:
        # Windows implementation
        if msvcrt.kbhit():
            key = msvcrt.getch()
            # Arrow keys are sent as two bytes, starting with 0xE0 (224)
            if key == b'\xe0':
                return msvcrt.getch()
            return key
        return None
    else:
        # Unix-like implementation
        # Check if there's data available to be read on stdin
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1)
        return None

def process_input():
    """Processes user input to change the snake's direction."""
    direction = (1, 0)  # Default direction: right

    key = get_key_press()
    if not key:
        return

    # Handle input for Windows (bytes)
    if "msvcrt" in sys.modules:
        if key == b"H" and direction != (0, 1):  # Up Arrow
            direction = (0, -1)
        elif key == b"P" and direction != (0, -1):  # Down Arrow
            direction = (0, 1)
        elif key == b"K" and direction != (1, 0):  # Left Arrow
            direction = (-1, 0)
        elif key == b"M" and direction != (-1, 0):  # Right Arrow
            direction = (1, 0)
    # Handle input for Unix (strings)
    else:
        key = key.lower()
        if key == "w" and direction != (0, 1):
            direction = (0, -1)
        elif key == "s" and direction != (0, -1):
            direction = (0, 1)
        elif key == "a" and direction != (1, 0):
            direction = (-1, 0)
        elif key == "d" and direction != (-1, 0):
            direction = (1, 0)

    return direction

def color(text, color):
    """Apply color to text using ANSI codes."""
    return f"{color}{text}{COLOR_RESET}"


def draw_frame(env):

    # Create a grid buffer for the frame
    buffer = [[color("  ",COLOR_BLACK_BG) for _ in range(env.WIDTH)] for _ in range(env.HEIGHT)]

    for i, (x, y) in enumerate(env.snake):
        char = GLYPH_SNAKE_HEAD if i == 0 else GLYPH_SNAKE_BODY
        buffer[y][x] = color(char, COLOR_SNAKE) + color(" ", COLOR_BLACK_BG)

    fx, fy = env.food_pos
    buffer[fy][fx] = color(GLYPH_FOOD, COLOR_FOOD)

    score_text = f"SCORE: {env.score}"
    score_len = len(score_text)//2

    # Assemble the final output string with walls
    output = color(GLYPH_CORNER_TL + (GLYPH_WALL_H * 2 * env.WIDTH) + GLYPH_CORNER_TR + "\n", COLOR_WALL)
    for row in buffer:
        output += color(GLYPH_WALL_V, COLOR_WALL) + "".join(row) + color(GLYPH_WALL_V + "\n", COLOR_WALL)


    bottom_text = [
        f"{color(GLYPH_CORNER_BL + ((GLYPH_WALL_H * (env.WIDTH - score_len))), COLOR_WALL)}",
        f"{color(score_text, COLOR_WALL + COLOR_BOLD)}",
        f"{color(((GLYPH_WALL_H * (env.WIDTH - score_len)) + GLYPH_CORNER_BR), COLOR_WALL)}"]

    output += "".join(bottom_text)

    sys.stdout.write("\033[?25l") 
    sys.stdout.write("\033[H")  
    sys.stdout.write(output)
    sys.stdout.flush()
    sys.stdout.write("\033[?25h")
