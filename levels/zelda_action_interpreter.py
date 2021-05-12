

def interpret_action(a: int) -> list:
    """

    :param a: integer action. e.g. argmax( NN(state) )
    :return: list of properly formatted actions for a multi-discrete action space
    """
    action = [0, 0]
    if a in [0, 1, 2, 3, 4]:
        action = [0, a]
    elif a in [5, 6]:
        action = [1, a - 5]
    return action
