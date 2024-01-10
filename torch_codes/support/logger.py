import time


class Logger:
    progressbar_element = chr(9608)

    def __init__(self):
        self.last_len = 0
        return

    def progress(self, f, length):
        num = int(f * length)
        return "[" + self.progressbar_element * num + " " * (length - num) + "]"

    def print(self, line, progress=None, length=30):
        if progress is not None:
            line += f" | progress : {int(progress * 100)}%" + self.progress(progress, length)

        print("\b" * self.last_len, end="")
        print(line, end="")
        self.last_len = len(line)


if __name__ == "__main__":
    logger = Logger()
    print(ord("█"))
    for i in range(100):
        logger.print(f"num: {i + 1} ", progress=(i + 1) / 100)
        time.sleep(0.5)
