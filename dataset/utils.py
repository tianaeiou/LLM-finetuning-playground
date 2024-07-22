import random
def normalize(value, min_value, max_value, min_output=0.0, max_output=1000.0):
    normalized_value = (value - min_value) * (max_output - min_output) / (max_value - min_value) + min_output
    normalized_value = min(max_output, max(normalized_value, min_value))
    return int(normalized_value)


def denormalize(normalized_value, min_value, max_value, min_output=0.0, max_output=1000.0):
    value = (normalized_value - min_output) * (max_value - min_value) / (max_output - min_output) + min_value
    return value

def get_random_index(num_samples, lower_bound, upper_bound, forbid_nums=[]):
    random_index = set()
    while len(random_index) < num_samples:
        idx = random.randint(lower_bound, upper_bound)
        if idx not in forbid_nums:
            random_index.add(idx)
    return random_index