import task_modules.clock as clock
import task_modules.myiot as myiot
import task_modules.schedule as schedule

sch = schedule.Schedule()

task_list = {"clock": clock, "myiot": myiot, "schedule": None}


def split_command(value):
    split = value.split('\n')
    texts = []
    commands = []

    for line in split:
        if line[:7] == "!module":
            command = line[8:].split()
            commands.append([command[0], " ".join(command[1:])])
        else:
            texts.append(line)

    text = "\n".join(texts)

    return text, commands


def do_task(commands, query):
    return_val = query
    require_response = False

    for command in commands:
        assert command[0] in task_list.keys()

        if command[0] == "schedule":
            val, res = sch.add_schedule(command[1], query)
        else:
            val, res = task_list[command[0]].execute(command[1], query)

        if val is not None:
            return_val += f"\n!module {command[0]} {command[1]}\n> {val}"
        require_response = require_response or res

    return return_val, require_response


if __name__ == "__main__":
    line = "!module clock get time"
    print(split_command(line))
    print(do_task(split_command(line)[1],"지금 시간이 몇시야?"))
