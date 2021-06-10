import os


def format(value):
    x, y = value
    return float("%.4f" % x), int(y)


def save(data, name):
    formatted = [format(v) for v in data]
    with open(name, 'w+') as file:
        file.writelines(f'{v}\n' for v in formatted)


def save_line(data, name):
    if os.path.exists(name):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not
    with open(name, append_write) as file:
        file.write(f'{data}\n')
    file.close()


def save_obj(data, name):
    if os.path.exists(name):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not
    with open(name, append_write) as file:
        file.write(f'{data}\n')
    file.close()
