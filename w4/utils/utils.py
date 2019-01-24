def config_to_str(config):
    s = []
    for k, v in sorted(config.items(), key=lambda t: t[0]):
        s.append('{}={}'.format(k, v))
    return '__'.join(s)


def str_to_config(line):
    d = {}
    for pair in line.split('__'):
        k, v = pair.split('=')
        d[k] = v
    return d
