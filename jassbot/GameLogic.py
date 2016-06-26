import random
import numpy as np

from Config import *
from utils.Logger import *

def init():
    logger.info("init game")

    state = {}
    for i in range(2):
        state['t%s_game_points' % i] = 0
        state['t%s_points' % i] = 0
    return state

def new_game(state):
    logger.info("new game")
    game = []
    for suit in suits:
        for rank in ranks:
            game.append("%s_%s" % (suit, rank))

    random.shuffle(game)

    for i in range(4):
        state['p%i_hand' % i] = game[i*9:i*9+9]

    state['trump'] = random.choice(suits)

    state['played'] = []
    state['current'] = []
    state['player_idx'] = 0

def card_suit(card):
    return card.split('_')[0]

def card_rank(card):
    return card.split('_')[1]

def sorted_trump(cards, trump):
    def get_key(item):
        return trump_ranking.index(card_rank(item))
    return sorted([c for c in cards if card_suit(c) == trump], key=get_key)

assert sorted_trump(["heart_k", "club_a", "heart_j", "diamond_k", "heart_9"], "heart") == ["heart_j", "heart_9", "heart_k"]

def count_card(card, trump):
    suit = card_suit(card)
    rank = card_rank(card)
    if suit == trump:
        return trump_point[rank]
    else:
        return non_trump_point[rank]

assert count_card("heart_k", "club") == 4
assert count_card("heart_k", "heart") == 4
assert count_card("heart_j", "club") == 2
assert count_card("heart_j", "heart") == 20
assert count_card("club_6", "club") == 0
assert count_card("diamond_7", "club") == 0
assert count_card("diamond_9", "heart") == 0
assert count_card("diamond_9", "diamond") == 14

def count_round(current, trump):
    return sum([count_card(c, trump) for c in current])

assert count_round(["heart_k", "heart_j", "heart_10", "club_6"], "heart") == 34
assert count_round(["heart_k", "heart_j", "heart_10", "club_6"], "club") == 16

def player_holding_hand(current, trump, player_start_idx):
    assert len(current) <= 4
    ranking = []
    served_suit = card_suit(current[0])
    rank_not_served_suit = 100
    for card in current:
        suit = card_suit(card)
        rank = card_rank(card)
        if suit == trump:
            ranking.append(trump_ranking.index(rank))
        elif suit == served_suit:
            ranking.append(non_trump_ranking.index(rank) + 10)
        else:
            ranking.append(rank_not_served_suit)

    win_idx = np.argmin(np.array(ranking))
    return (player_start_idx + win_idx) % 4

assert player_holding_hand(["heart_k", "heart_j", "heart_10", "club_6"], "heart", 0) == 1
assert player_holding_hand(["heart_k", "heart_j", "heart_10", "club_6"], "heart", 1) == 2
assert player_holding_hand(["heart_k", "heart_j", "heart_10", "club_6"], "heart", 2) == 3
assert player_holding_hand(["heart_k", "heart_j", "heart_10", "club_6"], "heart", 3) == 0
assert player_holding_hand(["heart_k", "heart_j", "heart_10", "club_6"], "club", 0) == 3
assert player_holding_hand(["heart_k", "heart_j", "heart_10", "club_6"], "club", 1) == 0
assert player_holding_hand(['diamond_7', 'heart_10', 'heart_6', 'spade_9'], "club", 0) == 0
assert player_holding_hand(['diamond_7', 'heart_10', 'heart_6'], "club", 0) == 0
assert player_holding_hand(["heart_k", "heart_j"], "heart", 1) == 2


def possible_cards(player_cards, current, trump):
    if len(current) == 0:
        return set(player_cards)
    else:
        served_suit = card_suit(current[0])
        served_trump = served_suit == trump
        if served_trump:
            # todo I don't consider that the 9 or buur can be not played
            any_trump = any([card_suit(c) == trump for c in player_cards])
            if any_trump:
                return set([c for c in player_cards if card_suit(c) == trump])
            else:
                return set(player_cards)
        else: # not served trump
            player_trump_cards = [c for c in player_cards if card_suit(c) == trump]
            player_all_trump = len(player_trump_cards) == len(player_cards)
            player_any_trump = len(player_trump_cards) > 0
            any_trump_current = any([card_suit(c) == trump for c in current])
            #print("player_all_trump: %s" % player_all_trump)
            #print("any_trump_current: %s" % any_trump_current)
            #print("player_any_trump: %s" % player_any_trump)
            if not player_all_trump and any_trump_current and player_any_trump:
                highest_trump_current = sorted_trump(current, trump)[0]
                #print("hightest_trump_current: %s" % highest_trump_current)
                player_possible_trump_cards = [c for c in player_trump_cards \
                                                 if trump_ranking.index(card_rank(c)) < \
                                                    trump_ranking.index(card_rank(highest_trump_current))]
            else:
                player_possible_trump_cards = player_trump_cards
            player_same_color_cards = [c for c in player_cards if card_suit(c) == served_suit]
            if len(player_same_color_cards) > 0:
                possible_non_trump_cards = player_same_color_cards
            else:
                possible_non_trump_cards = [c for c in player_cards if card_suit(c) not in [trump, served_suit]]
            return set(player_possible_trump_cards + possible_non_trump_cards)

assert possible_cards(["heart_9", "diamond_10"], ["heart_k"], "heart") == \
    set(["heart_9"])
assert possible_cards(["heart_9", "diamond_10"], ["heart_k"], "diamond") == \
    set(["heart_9", "diamond_10"])
assert possible_cards(["heart_9", "diamond_10"], [], "diamond") == \
    set(["heart_9", "diamond_10"])
assert possible_cards(["heart_9", "diamond_10"], ["diamond_6", "heart_j"], "heart") == \
    set(["diamond_10"])

def play_once(state, choose_cbk):
    player_idx = state['player_idx']
    player_cards = state['p%i_hand' % player_idx]
    cards = possible_cards(player_cards, state['current'], state['trump'])

    #print("cards: %s" % cards)

    card = choose_cbk(list(cards))

    #print("player cards: %s, possible cards: %s, selected card: %s" % (player_cards,cards,card))
    state['p%i_hand' % player_idx].remove(card)
    state['current'].append(card)

    logger.debug("player %i plays %s" % (player_idx, card))

    if len(state['current']) == 4:
        count = count_round(state['current'], state['trump'])
        player = player_holding_hand(state['current'], state['trump'], (player_idx + 1) % 4)
        team = player % 2
        assert player in range(4)
        assert team in [0,1]
        state["t%i_game_points" % team] += count
        state['played'] += state['current']
        state['current'] = []
        state['player_idx'] = player # the winning player starts the next round

        logger.info("player %d & team %d wins the round: %d points" % (player, team, count))
        if len(state['played']) == 36:
            state["t%i_game_points" % team] += 5 # five points for last round
            logger.info("game finished: team 0 %d pts, team 1 %d pts" % (state['t0_game_points'], state['t1_game_points']))

            # which team did win the game?
            # todo equal nb of points?
            team = 0 if state['t0_game_points'] > state['t1_game_points'] else 1

            state['t0_points'] += state['t0_game_points']
            state['t1_points'] += state['t1_game_points']

            ratio = 1.0 * state['t0_game_points'] / (state['t0_game_points'] + state['t1_game_points'])

            state['t0_game_points'] = 0
            state['t1_game_points'] = 0

            return {'team': team, 'ratio': ratio, 'final': True}
        else:
            # the round is finished but not final, we know which team did win it
            return {'team': team, 'final': False}

    else:
        # next player idx
        state['player_idx'] = (player_idx + 1) % 4
        # the round is not yet finished
        return None

def play_round(state, choose_cbk):
    for i in range(4):
        play_once(state, choose_cbk)

def print_state(state):
    logger.debug("-------------------------------")
    logger.debug("trump: %s" % state["trump"])
    logger.debug("current: %s" % state["current"])
    logger.debug("player 0: %s" % state["p0_hand"])
    logger.debug("player 1: %s" % state["p1_hand"])
    logger.debug("player 2: %s" % state["p2_hand"])
    logger.debug("player 3: %s" % state["p3_hand"])
    logger.debug("team 0: %d pts, team 1: %d pts" % (state["t0_points"], state["t1_points"]))
    logger.debug("played: %s" % state['played'])
    logger.debug("-------------------------------")


import datetime

def now_as_string():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")

