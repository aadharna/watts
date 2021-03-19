

def interpretAction(a: int) -> list:
    """

    :param a: integer action. e.g. argmax( NN(state) )
    :return: list of properly formatted actions for a multi-discrete action space
    """
    action = [0, 0]
    if a in [0, 1, 2, 3, 4]:
        # Todo: REMOVE INT CAST
        # THIS will happen with Chris's update to allow for
        # e.g. nparray(2)
        action = [0, int(a)]
    elif a in [5, 6]:
        action = [1, int(a) - 5]
    return action