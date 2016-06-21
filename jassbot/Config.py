
suits = ['spade','heart','diamond','club']
ranks = ['6', '7', '8', '9', '10', 'j', 'q', 'k', 'a']

trump_point = {
    '6': 0,
    '7': 0,
    '8': 0,
    '9': 14,
    '10': 10,
    'j': 20,
    'q': 3,
    'k': 4,
    'a': 11
}
trump_ranking = ['j', '9', 'a', 'k', 'q', '10', '8', '7', '6']

non_trump_point = {
    '6': 0,
    '7': 0,
    '8': 0,
    '9': 0,
    '10': 10,
    'j': 2,
    'q': 3,
    'k': 4,
    'a': 11
}

non_trump_ranking = ['a', 'k', 'q', 'j', '10', '9', '8', '7', '6']

assert len(trump_ranking) == 9
assert len(non_trump_ranking) == 9
assert len(trump_point) == 9
assert len(non_trump_point) == 9
