import IPython.display as dp
import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

from Config import *
from GameLogic import *

#pd.set_option('display.max_columns', None)


def order_suits(hand, trump):
    return [trump] + [s for s in suits if s != trump]

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

def state2features(trump,player_hand,played,current,player_idx):
    ordered_suits = order_suits(player_hand, trump)

    feat = np.zeros(36 * 3 + 2)

    cards2features(feat, 0*36, player_hand, ordered_suits)
    cards2features(feat, 1*36, played,      ordered_suits)
    cards2features(feat, 2*36, current,     ordered_suits)

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


def choose(state,possible_cards,model,temp_memory,epsilon):
    player_idx = state['player_idx']
    trump = state['trump']
    player_hand = state['p%i_hand' % player_idx]
    played = state['played']
    current = state['current']

    # get features for this state
    feat = state2features(trump,player_hand,played,current,player_idx)

    team_idx = player_idx % 2
    ordered_suits = order_suits(player_hand, trump)

    # make a prediction
    qval = model.predict(feat.reshape(1,input_layer_nb), batch_size=1)

    if (random.random() < epsilon): #choose random action
        card = random.choice(possible_cards)

        # list + [] in order to copy the list
        temp_memory.append((team_idx,feat,qval,card,ordered_suits,trump,player_hand+[],played+[],current+[],player_idx))

        get_logger().info('random card choice: %s' % card)

    else: #choose best action from Q(s,a) values
        # first we get a feature vect corresponding to the possible move

        mask_qval = np.zeros(36)
        cards2features(mask_qval, 0, possible_cards, ordered_suits)

        # now we can mask the qval we predicted with the possible moves
        # todo: proper solution
        moves = qval  + (mask_qval * 1000)
        idx = (np.argmax(moves))

        # now we have a prediction, we reconstruct the card out of it
        suit_idx = idx / 4
        rank_idx = idx % 9
        card = '%s_%s' % (ordered_suits[suit_idx], ranks[rank_idx])

        # list + [] in order to copy the list
        temp_memory.append((team_idx,feat,qval,card,ordered_suits,trump,player_hand+[],played+[],current+[],player_idx))

        get_logger().info('ml card choice: %s' % card)

    return card

def choose_for_test(state,possible_cards,model,temp_memory,epsilon):
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
        feat = state2features(trump,player_hand,played,current,player_idx)

        team_idx = player_idx % 2
        ordered_suits = order_suits(player_hand, trump)

        # make a prediction
        qval = model.predict(feat.reshape(1,input_layer_nb), batch_size=1)[0]

        mask_qval = np.zeros(36)
        cards2features(mask_qval, 0, possible_cards, ordered_suits)

        # now we can mask the qval we predicted with the possible moves
        # todo: proper solution
        moves = qval + (mask_qval * 1000)
        idx = (np.argmax(moves))
        #print('ordered_suits: %s, mask_qval: %s, moves: %s, idx: %i' % (ordered_suits, mask_qval, moves, idx))

        # now we have a prediction, we reconstruct the card out of it
        suit_idx = idx / 9
        rank_idx = idx % 9
        card = '%s_%s' % (ordered_suits[suit_idx], ranks[rank_idx])
        #print('possible cards: %s, chosen card: %s' % (possible_cards, card))

    return card



def get_reward(winning_team_idx,team_idx,final_round):
    if winning_team_idx == team_idx:
        return 10 if final_round else 1
    else:
        return -10 if final_round else -1

def play_card(card,player_hand,current,played):
    player_hand.remove(card)
    current.append(card)
    if len(current) == 4:
        played += current
        current[:] = []

def yindex_of_card(card, ordered_suits):
    suit = card_suit(card)
    suit_idx = ordered_suits.index(suit)
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
    for (team_idx,feat,qval,card,ordered_suits,trump,player_hand,played,current,player_idx) \
          in temp_memory:
        win_result = round_wins[i % 4]
        winning_team_idx = win_result['team']
        is_final = win_result['final']

        reward = get_reward(winning_team_idx,team_idx,is_final)

        if is_final:
            assert reward in [-10,10]

            # if final round, we only use reward for the target qval update
            update = reward

        else:
            assert reward in [-1,1]

            # update state with the move + feature of next state
            play_card(card,player_hand,current,played)
            next_feat = state2features(trump,player_hand,played,current,player_idx)

            next_qval = model.predict(next_feat.reshape(1,input_layer_nb), batch_size=1)

            max_next_qval = np.max(next_qval)

            # if not final round, we use the q-learning algo
            update = reward + (gamma * max_next_qval)

        # update the output with our updated value
        yindex = yindex_of_card(card,ordered_suits)
        y = np.zeros([1,len(qval[0])])
        y = qval[0]
        get_logger().info('update is %s' % update)
        y[yindex] = update
        #print('y is %s' % y)

        #print("Shape of feat: %s" % feat.shape)
        model.fit(feat.reshape(1,input_layer_nb), y.reshape(1,36), batch_size=1, nb_epoch=1, verbose=0)

        i += 1

    # reset temp_memory and round_wins
    temp_memory[:] = []
    round_wins[:] = []
