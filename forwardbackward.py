import sys
import math

# parse the index_to_word/tag file to a dictionary
# which keys are words/tags and vals are index
def parse_dict(i_to_sth):
    idx_dict = dict()
    with open(i_to_sth, "rt") as dictF:
        dict_list = (dictF.read()).splitlines()
        L = len(dict_list)
        for i in range(L):
            idx_dict[dict_list[i]] = i
    return idx_dict, L, dict_list


# parse the original file
def get_text(train_in_file, i2w_d, i2t_d):
    texts = []
    count_words = 0
    with open(train_in_file) as tF:
        sents = (tF.read()).splitlines()
        for sent in sents:
            sentence = []
            text = sent.split(" ")
            count_words += len(text)
            for word_tag in text:
                word = (word_tag.split("_"))[0]
                word_i = i2w_d[word]
                tag = (word_tag.split("_"))[1]
                tag_i = i2t_d[tag]
                sentence.append((word_i, tag_i))
            texts.append(sentence)
    #print(texts)
    return texts, sents, count_words

# parse the info from learning
# in the form of a vector consists of J elements
def get_prior(hmm_prior):
    with open(hmm_prior, "rt") as pF:
        text = (pF.read()).splitlines()
        J = len(text)
        pi_vec = [-1 for i in range(J)] 
        for i in range(J):
            pi_vec[i] = float(text[i])
    return pi_vec, J

# in the form of a matrix J * W
def get_emmit(hmm_emmit): 
    with open(hmm_emmit, "rt") as eF:
        text = (eF.read()).splitlines()
        J = len(text)
        emmi = []
        for i in range(J):
            value = (text[i].split(" "))
            W = len(value)
            cur = [0 for i in range(W)]
            for j in range(W):
                cur[j] = float(value[j])
            emmi.append(cur)
    return emmi

# in the form of a matrix
def get_tran(hmm_trans):
    with open(hmm_trans, "rt") as tF:
        text = (tF.read()).splitlines()
        J = len(text)
        trans = [[-1 for i in range(J)] for j in range(J)]
        for i in range(J):
            # -1 for exluding the last space char
            values = (text[i].split(" "))
            for j in range(J):
                trans[i][j] = float(values[j])
    return trans

# compute the quantities needed for for-back
def forward(word_seq, J, prior, emmi, trans):
    T = len(word_seq)
    alpha = [[-1.0 for i in range(J)] for j in range(T)]
    # get alpha_1:
    #init_weighted_sum = 0.0
    for tag in range(J):
        alpha[0][tag] = prior[tag] * emmi[tag][word_seq[0][0]]
        #init_weighted_sum += alpha[0][tag]

    '''
    for j in range(J):
        alpha[0][j] /= init_weighted_sum
    '''

    for i in range(1,T):
        for j in range(J):
            alpha[i][j] = get_alpha(word_seq, J, alpha, i, j, emmi, trans)
        '''
        # normalization
        if i != T - 1:
            weighted_sum = 0.0
            for j in range(J):
                weighted_sum += alpha[i][j]
            for j in range(J):
                alpha[i][j] /= weighted_sum
        '''
    return alpha

# updating alpha, t > 0
def get_alpha(word_seq, J, alpha, t, j, emmi, trans):
    if alpha[t][j] != -1.0:
        return alpha[t][j]
    word = word_seq[t][0]
    cur_sum = 0
    for tag in range(J):
        cur_sum += get_alpha(word_seq, J, \
                alpha, t-1, tag, emmi, trans) * trans[tag][j]
    return emmi[j][word] * cur_sum

def backward(word_seq, J, emmi, trans):
    T = len(word_seq)
    beta = [[-1.0 for i in range(J)] for j in range(T)]
    # the last row
    beta[-1] = [1.0 for i in range(J)]
    for i in range(T - 1):
        for j in range(J):
            beta[i][j] = get_beta(word_seq, J, beta, i, j, emmi, trans)
    return beta

def get_beta(word_seq, J, beta, t, j, emmi, trans):
    if beta[t][j] != -1.0:
        return beta[t][j]
    cur_sum = 0
    word = word_seq[t + 1][0]
    for tag in range(J):
        cur_sum += emmi[tag][word] * trans[j][tag] * get_beta(word_seq, J, beta, t + 1, tag, emmi, trans)
    beta[t][j] = cur_sum
    return cur_sum

def predict(alpha, beta, J):
    res = []
    for i in range(len(alpha)):
        Pt = []
        for j in range(J):
            Pt.append(alpha[i][j] * beta[i][j])
        max_idx = 0
        max_val = 0
        for k in range(J):
            if Pt[k] > max_val:
                max_val = Pt[k]
                max_idx = k
        res.append(max_idx)
    return res

if __name__ == "__main__":
    t_in = sys.argv[1]
    i_to_word = sys.argv[2]
    i_to_tag = sys.argv[3]
    hmm_prior = sys.argv[4]
    hmm_emmit = sys.argv[5]
    hmm_trans = sys.argv[6]
    p_out = sys.argv[7]
    m_out = sys.argv[8]
    i2w, tw, w2i = parse_dict(i_to_word)
    i2t, tj, t2i = parse_dict(i_to_tag)
    texts, sents, count_words = get_text(t_in, i2w, i2t)
    h_prior, J = get_prior(hmm_prior)
    h_emmi = get_emmit(hmm_emmit)
    h_trans = get_tran(hmm_trans)
    # print the predected labels
    count_accurate = 0.0
    log_sum = 0.0
    with open(p_out, "wt") as pF:
        for i in range(len(texts)):
            sent = sents[i]
            alpha = forward(texts[i],J,h_prior, h_emmi, h_trans)
            print(alpha)
            like = 0.0
            for tag in range(J):
                like += alpha[-1][tag]
            log_sum += math.log(like)
            beta = backward(texts[i], J, h_emmi, h_trans)
            print(beta)
            best = predict(alpha, beta, J)
            words = sent.split(" ")
            for j in range(len(words)):
                if best[j] == texts[i][j][1]:
                    count_accurate += 1
                word = (words[j].split("_"))[0]
                if j != len(words) - 1:
                    pF.write(word + "_" + t2i[best[j]] + " ")
                else:
                    pF.write(word + "_" + t2i[best[j]] + "\n")
            
    with open(m_out, "wt") as mF:
        avg_l = log_sum / len(sents)
        acc = count_accurate / count_words
        mF.write("Average Log-Likelihood: ")
        mF.write(str(avg_l) + "\n")
        mF.write("Accuracy: ")
        mF.write(str(acc))
