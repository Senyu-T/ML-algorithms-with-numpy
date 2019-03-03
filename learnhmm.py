import sys

# parse the index_to_word/tag file to a dictionary
# which keys are words/tags and vals are index
def parse_dict(i_to_sth):
    idx_dict = dict()
    with open(i_to_sth, "rt") as dictF:
        dict_list = (dictF.read()).splitlines()
        L = len(dict_list)
        for i in range(L):
            idx_dict[dict_list[i]] = i
    return idx_dict, L

# parse the original file
def get_text(train_in_file, i2w_d, i2t_d):
    texts = []
    with open(train_in_file) as tF:
        sents = (tF.read()).splitlines()
        for sent in sents:
            sentence = []
            text = sent.split(" ")
            for word_tag in text:
                word = (word_tag.split("_"))[0]
                word_i = i2w_d[word]
                tag = (word_tag.split("_"))[1]
                tag_i = i2t_d[tag]
                sentence.append((word_i, tag_i))
            texts.append(sentence)
    #print(texts)
    return texts

# get the initial prob vector
# J is the total number of tags
def get_init(texts, J, write_on):
    counter = [1 for i in range(J)]
    res = [1.0 for i in range(J)]
    for sent in texts:
        first_tag = sent[0][1]
        counter[first_tag] += 1
    get_sum = 0
    for i in range(J):
        get_sum += float(counter[i])
    with open(write_on, "wt") as wF:
        for i in range(J):
            cur = counter[i] / get_sum
            wF.write("%.16e\n" % cur)

# get the transition matrix
def get_trans(trainF, J, write_on):
    counter = [[1.0 for i in range(J)] for j in range(J)]
    for sent in texts:
        for i in range(len(sent) - 1):
            from_tag = sent[i][1]
            to_tag = sent[i + 1][1]
            counter[from_tag][to_tag] += 1
    with open(write_on, "wt") as wF:
        for row in counter:
            count_sum = 0
            for ele in row:
                count_sum += ele
            for i in range(len(row)):
                cur = row[i] / count_sum
                if i != len(row) - 1:
                    wF.write("%.16e " % cur)
                else:
                    wF.write("%.16e\n" % cur)

# get the emission matrix
def get_emit(trainF, W, J, write_on):
    counter = [[1.0 for i in range(W)] for j in range(J)]
    for sent in texts:
        for ele in sent:
            word = ele[0]
            tag = ele[1]
            counter[tag][word] += 1
    with open(write_on, "wt") as wF:
        for row in counter:
            count_sum = 0
            for ele in row:
                count_sum += ele
            for i in range(len(row)):
                cur = row[i] / count_sum
                if i != (len(row) - 1):
                    wF.write("%.16e " % cur)
                else:
                    wF.write("%.16e\n" % cur)

if __name__ == "__main__":
    t_in = sys.argv[1]
    i_to_word = sys.argv[2]
    i_to_tag = sys.argv[3]
    hmm_prior = sys.argv[4]
    hmm_emmit = sys.argv[5]
    hmm_tran = sys.argv[6]
    i2w, W = parse_dict(i_to_word)
    i2t, J = parse_dict(i_to_tag)
    texts = get_text(t_in, i2w, i2t)
    get_init(texts, J, hmm_prior)
    get_trans(texts, J, hmm_tran)
    get_emit(texts, W, J, hmm_emmit)
