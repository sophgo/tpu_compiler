from functools import partial, wraps

class Color:
    # Foreground:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    # Formatting
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    # End colored text
    END = '\033[0m'
    NC = '\x1b[0m'  # No Color



def docolor(color):
    def decorator(func):
        def wrap(*args, **kwargs):
            if color == 'blue':
                print(Color.OKBLUE)
            elif color == 'green':
                print(Color.OKGREEN)
            elif color == 'yellow':
                print(Color.WARNING)
            elif color == 'red':
                print(Color.FAIL)
            f = func(*args)
            print(Color.END)
            return f
        return wrap
    return decorator

