

def clean_dict_stats(stats):
    for i in range(len(stats)):
        if 'plots' in stats[i].keys():
            del stats[i]['plots']

    return stats