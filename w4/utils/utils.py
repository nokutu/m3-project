def config_to_str(config):
    s = []
    for k, v in sorted(config.items(), key=lambda t: t[0]):
        s.append('{}={}'.format(k, v))
    return '__'.join(s)

