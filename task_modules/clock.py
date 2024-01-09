import time


def clock_get(item):
    return time.asctime()


def execute(command, query=None):
    split = command.split()

    if len(split) != 2:
        print("error : incorrect command")
        return None

    if split[0] == 'get':
        return clock_get(split[1]), True


if __name__ == "__main__":
    print(execute("get time"))
