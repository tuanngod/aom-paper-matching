
from py_stringmatching.similarity_measure.affine import Affine
from py_stringmatching.similarity_measure.tfidf import TfIdf


def author_sim_affine(text1, text2):
    # default: (gap_start) gs = 1; (gap_continuation) gc = 0.5
    aff = Affine(gap_start=2, gap_continuation=0.5)
    js = text1.split('|')
    cs = text2.split('|')
    if len(js) < len(cs):
        denominator = len(js)
        left_set = js
        right_set = cs
    else:
        denominator = len(cs)
        left_set = cs
        right_set = js
    numerator = 0
    for i in left_set:
        score_max = -1
        for j in right_set:
            if score_max < aff.get_raw_score(j, i):
                score_max = aff.get_raw_score(j, i)
        numerator += score_max
    return numerator/denominator

def authors_similarity(ltuple, rtuple):
    j = str(ltuple["authors_first_last"])
    c = str(rtuple["authors_first_last"])
    return author_sim_affine(j, c)

def first_author_similarity(ltuple, rtuple):
    c_first_author_first_last = str(rtuple['first_author_first']) + "_" +\
                                str(rtuple['first_author_last'])
    jauthors = str(ltuple["authors"])
    return author_sim_affine(c_first_author_first_last, jauthors)

def bow_tfidf(ltuple, rtuple):
    df = pd.read_csv('{}/corpus.csv'.format(path_to_csv_dir))
    tfidf = TfIdf(corpus_list=list(df.corpus))
    tfidf.set_dampen(True)
    bag1 = str(ltuple["bow"]).split(",")
    bag2 = str(rtuple["bow"]).split(",")
    return tfidf.get_sim_score(bag1, bag2)

def year_difference(ltuple, rtuple):
    return ltuple["year"] - rtuple["year"]

