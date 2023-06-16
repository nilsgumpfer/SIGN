def get_methods(dataset_id, model_id, variant=None):
    if variant is None:
        path = 'data/config/methods_{}_{}.txt'.format(dataset_id, model_id)
    else:
        path = 'data/config/methods_{}_{}_{}.txt'.format(dataset_id, model_id, variant)
    with open(path) as f:
        methods = f.read().splitlines()

    # Filter methods (comment)
    methods_filtered = []
    for m in methods:
        if not m.startswith('#'):
            methods_filtered.append(m)

    return methods_filtered


def get_modelids(dataset_id):
    path = 'data/config/models_{}.txt'.format(dataset_id)

    with open(path) as f:
        methods = f.read().splitlines()

    # Filter methods (comment)
    methods_filtered = []
    for m in methods:
        if not m.startswith('#'):
            methods_filtered.append(m)

    return methods_filtered

