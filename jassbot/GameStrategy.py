import IPython.display as dp
import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

from Config import *
from GameLogic import *
from Chrono import *
from utils.Logger import *

#pd.set_option('display.max_columns', None)


def sort_suits(suit_specs):
    '''
    Sort the given suit specifications according to the following criteria:
    - take the suit with the largest nb first (more cards of that suits are better)
    - take the suit with the lowest best_rank first (best_rank is an index in the rank list from best to worst)
    :param suit_specs:
    :return: sorted suit spec list
    '''
    def rank_order(i1,i2):
        return i2['nb'] - i1['nb'] if i1['nb'] != i2['nb'] else i1['best_rank'] - i2['best_rank']
    return sorted(suit_specs, cmp=rank_order)

assert sort_suits([{'nb': 1, 'best_rank':4}, {'nb': 2, 'best_rank': 2}]) == \
    [{'nb': 2, 'best_rank': 2}, {'nb': 1, 'best_rank':4}]
assert sort_suits([{'nb': 1, 'best_rank':4}, {'nb': 1, 'best_rank': 2}]) == \
    [{'nb': 1, 'best_rank': 2},{'nb': 1, 'best_rank':4}]

def order_suits(hand, trump):
    non_trump_suits = [s for s in suits if s != trump]

    suit_specs = []
    for suit in non_trump_suits:
        cards_of_suit = [non_trump_ranking.index(card_rank(c)) for c in hand if c.startswith(suit)]
        nb = len(cards_of_suit)
        best_rank = min(cards_of_suit) if nb > 0 else 10
        suit_specs.append({'suit': suit, 'nb': nb, 'best_rank': best_rank})

    sorted_non_trump_suits = sort_suits(suit_specs)

    return [trump] + [d['suit'] for d in sorted_non_trump_suits]

assert order_suits(['club_8', 'club_10', 'club_k', 'heart_7', 'heart_a', 'diamond_j'], 'diamond') == \
    ['diamond', 'club', 'heart', 'spade']
assert order_suits(['club_8', 'club_10', 'club_k', 'heart_7', 'heart_a', 'heart_6', 'diamond_j'], 'diamond') == \
    ['diamond', 'heart', 'club', 'spade']

def cards2features(feat, offset, cards, ordered_suits):
    for card in cards:
        suit = card_suit(card)
        rank = card_rank(card)
        suit_idx = ordered_suits.index(suit)
        rank_idx = ranks.index(rank)
        feat[offset + suit_idx*9 + rank_idx] = 1
    return feat

assert(all(cards2features(np.zeros(36), 0, \
                      ['heart_7', 'diamond_7', 'club_8', 'club_7', 'heart_q', 'spade_9', 'diamond_10'], \
                      ['club', 'heart', 'diamond', 'spade']) == \
                      np.array([0,1,1,0,0,0,0,0,0, \
                                0,1,0,0,0,0,1,0,0, \
                                0,1,0,0,1,0,0,0,0, \
                                0,0,0,1,0,0,0,0,0])))

assert(all(cards2features(np.zeros(36), 0, \
                      ['club_10', 'spade_8', 'spade_6', 'club_k', 'heart_j', 'spade_j', 'spade_10', 'spade_q', 'heart_6'], \
                      ['club', 'heart', 'diamond', 'spade']) == \
                      np.array([0,0,0,0,1,0,0,1,0, \
                                1,0,0,0,0,1,0,0,0, \
                                0,0,0,0,0,0,0,0,0, \
                                1,0,1,0,1,1,1,0,0])))

def state2features(suit_order,trump,player_hand,played,current,player_idx):

    feat = np.zeros(36 * 3 + 2)

    # the first 36 features: only the possible cards
    #poss_cards = possible_cards(player_hand,current,trump)

    cards2features(feat, 0*36, player_hand,  suit_order)
    cards2features(feat, 1*36, played,      suit_order)
    cards2features(feat, 2*36, current,     suit_order)

    # does our team hold the hand
    if len(current) > 0:
        start_player_idx = player_idx - len(current) % 4
        team_holding_hand = player_holding_hand(current, trump, start_player_idx) % 2
        feat[3*36] = 1 if team_holding_hand == (player_idx % 2) else 0

    # any team played
    feat[3*36+1] = 1 if len(current) > 0 else 0
    return feat

def print_feat(feat):
    tmp = np.append(feat, np.zeros(36*4 - len(feat)))
    df = pd.DataFrame({'player_hand':tmp[0:36], 'played':tmp[36:36*2], 'current':tmp[36*2:36*3], 'misc':tmp[36*3:36*4]})
    df = df[['current', 'player_hand', 'played', 'misc']]
    dp.display(df.transpose())


input_layer_nb = 3*36+2
hidden_layer_1_nb = 220
hidden_layer_2_nb = 300
output_layer_nb = 36

def create_model():
    model = Sequential()
    model.add(Dense(hidden_layer_1_nb, init='lecun_uniform', input_shape=(input_layer_nb,)))
    model.add(Activation('relu'))
    #model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?

    model.add(Dense(hidden_layer_2_nb, init='lecun_uniform'))
    model.add(Activation('relu'))
    #model.add(Dropout(0.2))

    model.add(Dense(output_layer_nb, init='lecun_uniform'))
    model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

    rms = RMSprop()
    model.compile(loss='mse', optimizer=rms)
    return model

hidden_layer_3_nb = 150
hidden_layer_4_nb = 75

def create_model_4layers():
    model = Sequential()
    model.add(Dense(hidden_layer_1_nb, init='lecun_uniform', input_shape=(input_layer_nb,)))
    model.add(Activation('relu'))
    #model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?

    model.add(Dense(hidden_layer_2_nb, init='lecun_uniform'))
    model.add(Activation('relu'))
    #model.add(Dropout(0.2))

    model.add(Dense(hidden_layer_3_nb, init='lecun_uniform'))
    model.add(Activation('relu'))
    #model.add(Dropout(0.2))

    model.add(Dense(hidden_layer_4_nb, init='lecun_uniform'))
    model.add(Activation('relu'))
    #model.add(Dropout(0.2))

    model.add(Dense(output_layer_nb, init='lecun_uniform'))
    model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

    rms = RMSprop()
    model.compile(loss='mse', optimizer=rms)
    return model

def choose(suit_order,state,possible_cards,model,temp_memory,epsilon):
    player_idx = state['player_idx']
    trump = state['trump']
    player_hand = state['p%i_hand' % player_idx]
    played = state['played']
    current = state['current']

    # get features for this state
    feat = state2features(suit_order,trump,player_hand,played,current,player_idx)

    team_idx = player_idx % 2

    # make a prediction
    qval = model.predict(feat.reshape(1,input_layer_nb), batch_size=1)

    if (random.random() < epsilon): #choose random action
        card = random.choice(possible_cards)

        # list + [] in order to copy the list
        temp_memory.append((team_idx,feat,qval,card,suit_order,trump,player_hand+[],played+[],current+[],player_idx))

        logger.info('random card choice: %s' % card)

    else: #choose best action from Q(s,a) values
        # first we get a feature vect corresponding to the possible move

        mask_possible = np.zeros(36)
        cards2features(mask_possible, 0, possible_cards, suit_order)

        mask_unpossible = (1-mask_possible)

        # now we can mask the qval we predicted with the possible moves
        # todo: proper solution
        moves = qval + (mask_possible * 1000) - (mask_unpossible * 1000)

        idx = (np.argmax(moves))

        # now we have a prediction, we reconstruct the card out of it
        suit_idx = idx / 9
        rank_idx = idx % 9
        card = '%s_%s' % (suit_order[suit_idx], ranks[rank_idx])

        # list + [] in order to copy the list
        temp_memory.append((team_idx,feat,qval,card,suit_order,trump,player_hand+[],played+[],current+[],player_idx))

        logger.info('ml card choice: %s' % card)

    return card

def choose_for_test(suit_order,state,possible_cards,model):
    player_idx = state['player_idx']

    # player_idx in [0,2] run using the model, while player_idx in [1,3] use random card selection

    if player_idx in [1,3]:
        card = random.choice(possible_cards)

    else:
        trump = state['trump']
        player_hand = state['p%i_hand' % player_idx]
        played = state['played']
        current = state['current']

        # get features for this state
        feat = state2features(suit_order,trump,player_hand,played,current,player_idx)

        team_idx = player_idx % 2

        # make a prediction
        qval = model.predict(feat.reshape(1,input_layer_nb), batch_size=1)[0]

        mask_possible = np.zeros(36)
        cards2features(mask_possible, 0, possible_cards, suit_order)

        mask_unpossible = (1-mask_possible)

        # now we can mask the qval we predicted with the possible moves
        # todo: proper solution
        moves = qval + (mask_possible * 1000) - (mask_unpossible * 1000)
        idx = (np.argmax(moves))
        #print('suit_order: %s, mask_qval: %s, moves: %s, idx: %i' % (suit_orders, mask_qval, moves, idx))

        # now we have a prediction, we reconstruct the card out of it
        suit_idx = idx / 9
        rank_idx = idx % 9
        card = '%s_%s' % (suit_order[suit_idx], ranks[rank_idx])

        if card not in player_hand:
            print("Warning: card %s not in hand %s" % (card, player_hand))
            print("Possible cards: %s" % possible_cards)
            print("Qval is: %s" % qval)
            print("Moves: %s" % moves)
            return random.choice(possible_cards)

        #print('possible cards: %s, chosen card: %s' % (possible_cards, card))

    return card



def get_reward(t0_points,t1_points,team_idx,final_round):
    sign = 1.0 if team_idx == 0 else -1.0

    # we don't reward winning a round, only winning the game
    if t0_points == t1_points:
        return 0
    if not final_round:
        pts = 0.5 if t0_points > t1_points else -0.5
    else: # t0_points <> t1_points
        if abs(t0_points - t1_points) >= 60:
            pts = 10 if t0_points > t1_points else -10
        else:
            pts = 1 if t0_points > t1_points else -1

    return sign * pts

def play_card(card,player_hand,current,played):
    player_hand.remove(card)
    current.append(card)
    if len(current) == 4:
        played += current
        current[:] = []

def yindex_of_card(card, suit_order):
    suit = card_suit(card)
    suit_idx = suit_order.index(suit)
    rank = card_rank(card)
    rank_idx = ranks.index(rank)
    return suit_idx * 9 + rank_idx

def update_model_game_end(model,temp_memory,round_wins,epsilon):
    # we reverse the memory, since we want to learn from the destination to the start
    # (since q-learning is going to look forward, it is better if we have already learned
    # the winning or loosing rewards)
    temp_memory.reverse()
    round_wins.reverse()

    i = 0
    for (team_idx,feat,qval,card,suit_order,trump,player_hand,played,current,player_idx) \
          in temp_memory:
        logger.info('team %i, card %s, trump %s, player %i' % (team_idx,card,trump,player_idx))
        win_result = round_wins[i / 4]
        is_final = win_result['final']

        if is_final:
            t0_points = win_result['t0_points']
            t1_points = win_result['t1_points']

            reward = get_reward(t0_points,t1_points,team_idx,is_final)

            # final state, we don't have a next max qval -> we put 0 so that the term
            # vanishes
            max_next_qval = 0

        else:
            reward = 0

            # update state with the move + feature of next state
            play_card(card,player_hand,current,played)
            next_feat = state2features(suit_order,trump,player_hand,played,current,player_idx)

            next_qval = model.predict(next_feat.reshape(1,input_layer_nb), batch_size=1)

            max_next_qval = np.max(next_qval)

        # update the output with our updated value
        yindex = yindex_of_card(card,suit_order)
        y = np.zeros([1,len(qval[0])])
        y = qval[0]
        update = y[yindex] + alpha * (reward + (gamma * max_next_qval) - y[yindex])
        logger.info('update qval %f -> %f (%s)' % (y[yindex], update, win_result))
        y[yindex] = update
        #print('y is %s' % y)

        #print("Shape of feat: %s" % feat.shape)
        model.fit(feat.reshape(1,input_layer_nb), y.reshape(1,36), batch_size=1, nb_epoch=1, verbose=0)

        i += 1

    # reset temp_memory and round_wins
    temp_memory[:] = []
    round_wins[:] = []


def save_model(model, stats, comment):
    now = now_as_string()

    json_string = model.to_json()
    text_file = open("data/%s-model.json" % now, "w")
    text_file.write(json_string)
    text_file.close()

    model.save_weights("data/%s-weights.json" % now)

    text_file = open("data/%s-comment.txt" % now, "w")
    text_file.write("%s\n" % comment)
    text_file.write("gamma: %f, alpha: %f\n" % (gamma, alpha))
    text_file.close()

    stats.to_csv("data/%s-stats.csv" % now)


def train_and_test(model, state, nb_simulations, nb_epochs, nb_test_runs, epsilon):
    chrono = Chrono()
    stats = pd.DataFrame({'rounds won':[], 'games won':[]})
    for s in range(nb_simulations):

        temp_memory = []
        round_wins = []
        print('*** learning phase %i / %i (eps: %.3f)' % (s + 1, nb_simulations, epsilon))
        chrono.start()
        for epoch in range(nb_epochs):
            new_game(state)
            logger.info("*** start of a new game")

            print_state(state)

            # we first define the suit order used later for the algorithm, for each player
            # (depending on his hand and the current trump suit)
            suit_orders = []
            for p in range(4):
                player_hand = state['p%i_hand' % p]
                trump = state['trump']
                suit_order = order_suits(player_hand, trump)
                suit_orders.append(suit_order)

            for j in range(9):
                for i in range(4):
                    suit_order = suit_orders[i]

                    #chrono.start()
                    result = play_once(state, lambda (possible_cards): \
                              choose(suit_order,state,possible_cards,model,temp_memory,epsilon))
                    #chrono.stop("choose")

                    assert i < 3 and result is None or \
                           i == 3 and result is not None

                    if i == 3: # at the end of one round

                        round_wins.append(result)

                        if len(state['played']) == 36:

                            # the game is finished, we can update the model
                            #chrono.start()
                            update_model_game_end(model,temp_memory,round_wins,epsilon)
                            #chrono.stop()

        chrono.stop('learning time')

        game_wins = pd.DataFrame({'game won':[]})
        round_wins = pd.DataFrame({'round won':[]})

        print('*** testing phase %i / %i' % (s + 1, nb_simulations))
        game_wins_idx = 0
        round_wins_idx = 0
        chrono.start()
        for epoch in range(nb_test_runs):
            new_game(state)
            logger.info("*** start of a new game")
            print_state(state)

            suit_orders = []
            for p in range(4):
                player_hand = state['p%i_hand' % p]
                trump = state['trump']
                suit_order = order_suits(player_hand, trump)
                suit_orders.append(suit_order)

            for j in range(9):
                for i in range(4):
                    suit_order = suit_orders[i]

                    result = play_once(state, lambda (possible_cards): \
                              choose_for_test(suit_order,state,possible_cards,model))

                    if result is not None:
                        team = result['team']
                        is_final = result['final']

                        if is_final:
                            ratio = result['ratio']
                            df = pd.DataFrame({'game won':[1 if team == 0 else 0], 'ratio': [ratio]}, index=[game_wins_idx])
                            game_wins = game_wins.append(df)
                            game_wins_idx += 1
                        else:
                            df = pd.DataFrame({'round won':[1 if team == 0 else 0]}, index=[round_wins_idx])
                            round_wins = round_wins.append(df)
                            round_wins_idx += 1
        chrono.stop('testing time')

        games_won = 1.0 * len(game_wins[game_wins['game won'] == 1]) / nb_test_runs
        ratio = game_wins['ratio'].mean()

        # we have 8 rounds, excepted the final one that we don't count here
        rounds_won = 1.0 * len(round_wins[round_wins['round won'] == 1]) / (nb_test_runs * 8)

        df = pd.DataFrame({'rounds won':[rounds_won], 'games won':[games_won], 'ratio':[ratio]},index=[s])
        dp.display(df)
        stats = stats.append(df)

        if epsilon > 0.1:
            epsilon -= (1.0 / nb_simulations)

    return stats