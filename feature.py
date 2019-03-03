import sys

def read_dict(dic_file):
    voc = {}
    with open(dic_file, "rt") as dic:
        in_string = dic.read()
        entry = in_string.splitlines()
        for ele in entry:
            l = ele.split()
            voc[l[0]] = l[1]
    return voc
        
def format_file(in_file, out_file, dic_file, flag):
    voc = read_dict(dic_file)
    with open(in_file, "rt") as inF:
        with open(out_file, "wt") as outF:
            in_string = inF.read()
            lines = in_string.splitlines()
            for line in lines:
                sent = {}
                list_w = line.split("\t")
                out_string = list_w[0] + "\t"
                sentence = list_w[1]
                if flag == 1:
                    for word in sentence.split():
                        if word in voc:
                            if word not in sent:
                                out_string += voc[word] 
                                out_string += ":1\t"
                                sent[word] = 1
                    out_string = out_string[:-1]
                    out_string += "\n"
                    outF.write(out_string)
                else:
                    for word in sentence.split():
                        if word in voc:
                            if voc[word] in sent:
                                sent[voc[word]] += 1
                            else:
                                sent[voc[word]] = 1
                    for word in sent:
                        if sent[word] < 4:
                            out_string += word 
                            out_string += ":1\t"
                    out_string = out_string[:-1]
                    out_string += "\n"
                    outF.write(out_string)

if __name__ == '__main__':
    i_tr = sys.argv[1]
    i_v = sys.argv[2]
    i_te = sys.argv[3]
    dic = sys.argv[4]
    f_tr = sys.argv[5]
    f_v = sys.argv[6]
    f_te = sys.argv[7]
    f_flag = (int)(sys.argv[8])
    format_file(i_tr, f_tr, dic, f_flag)
    format_file(i_v, f_v, dic, f_flag)
    format_file(i_te, f_te, dic, f_flag)
