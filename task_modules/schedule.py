from datetime import datetime
import time


class Schedule:
    def __init__(self, check_period=5):
        self.schedules = []
        self.lastTime = time.time()
        self.check_period = check_period

    def add_schedule(self, command, query):
        split = command.split()
        assert len(split) >= 3

        if len(split) == 3:
            self.schedules.append({
                "weekdays": split[1],
                "time": split[2],
                "date": None,
                "last_done": False,
                "query": f"{query}\n!module schedule {command}"
            })
        elif len(split) > 3:
            self.schedules.append({
                "weekdays": split[1],
                "time": split[2],
                "date": ' '.join(split[3:]),
                "last_done": False,
                "query": f"{query}\n!module schedule {command}"
            })

        return None, False

    def evaluate(self, schedule):
        if schedule['weekdays'] != 'x':
            if time.asctime().split()[0] not in schedule['weekdays']:
                return False
        if schedule['date'] is not None:
            if datetime.now().date() != datetime.strptime(schedule['date'], "%b %d %Y").date():
                return False
        if schedule['time'] is not None:
            try:
                dt = datetime.strptime(schedule['time'], '%H:%M').time()
                nt = datetime.now()
                return nt.hour == dt.hour and nt.minute == dt.minute
            except:
                return False

    def check(self):
        if time.time() - self.lastTime > self.check_period:
            self.lastTime = time.time()
            for schedule in self.schedules:
                try:
                    eval = self.evaluate(schedule)

                    if eval and not schedule["last_done"]:
                        schedule["last_done"] = True
                        if schedule["weekdays"] == 'x':
                            self.schedules.remove(schedule)
                        return schedule["query"] + "\n> " + time.asctime(), True
                    elif not eval:
                        schedule["last_done"] = False
                except:
                    continue

            return None, False
        else :
            return None, False

if __name__ == "__main__":
    s = Schedule()
    s.add_schedule("set x 18:55 Jan 09 2024", "나중에 알려줘")
    print(s.schedules)
    while True:
        _, res = s.check()
        if res:
            print(_, res)