import random
import string


def get_random_alphanum_string(num_chars=16):
    return''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(num_chars))