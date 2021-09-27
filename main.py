# -*- coding: utf-8 -*-
import os
import re
import sys
import csv
import rich
import logging
import argparse
import numpy as np
import pandas as pd
import py_entitymatching as em

from matching import *


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S"
)

def main(args):
    conf_fn = os.path.join(args.data_dir, args.conf_fn)
    jour_fn = os.path.join(args.data_dir, args.jour_fn)
    train_fn = os.path.join(args.data_dir, args.train_fn)

    CONF_INDEX = 'id'
    CONF = ['id', 'year', 'bow', 'authors', 'authors_first_last', 'first_author_first', 'first_author_last']
    CONF = em.read_csv_metadata(conf_fn, key=CONF_INDEX)

    JOUR_INDEX = 'id'
    JOUR = ['id', 'year', 'bow', 'authors', 'authors_first_last']
    JOUR = em.read_csv_metadata(jour_fn, key=JOUR_INDEX)

    train_set = em.read_csv_metadata(
        train_fn, 
        key='_id',
        ltable=JOUR, 
        rtable=CONF,
        fk_ltable='ltable_'+JOUR_INDEX, 
        fk_rtable='rtable_'+CONF_INDEX
    )

    feature_meta_data = em.get_features_for_matching(JOUR, CONF)
    em.add_blackbox_feature(feature_meta_data, 'authors_similarity', authors_similarity)
    em.add_blackbox_feature(feature_meta_data, 'first_author_similarity', first_author_similarity)
    em.add_blackbox_feature(feature_meta_data, 'year_difference', year_difference)
    em.add_blackbox_feature(feature_meta_data, 'bow_tfidf', bow_tfidf)
    logger.info('list of features: %s' % feature_meta_data.feature_name)

    features_fn = os.path.join(args.data_dir, args.features_fn)
    if os.path.exists(features_fn):
        feature_vectors = pd.read_csv(features_fn)
    else:
        feature_vectors = em.extract_feature_vecs(
            train_set, feature_table=feature_meta_data, attrs_after='gold_label')
        feature_vectors.to_csv(features_fn, index=False)

    em.set_key(feature_vectors, '_id') # key of the metadata
    em.set_fk_ltable(feature_vectors, 'ltable_id') #foreign key to left table
    em.set_fk_rtable(feature_vectors, 'rtable_id') #foreign key to right table
    em.set_ltable(feature_vectors, JOUR) #Sets the ltable for a DataFrame in the catalog
    em.set_rtable(feature_vectors, CONF) #Sets the rtable for a DataFrame in the catalog

    redudant_attrs = ['_id', 'ltable_'+JOUR_INDEX, 'rtable_'+CONF_INDEX, 'gold_label']
    feature_vectors = em.impute_table(
        feature_vectors, 
        exclude_attrs=redudant_attrs, 
        missing_val=np.nan, 
        strategy='mean'
    )

    dt = em.DTMatcher(name='DecisionTree', random_state=0)
    svm = em.SVMMatcher(name='SVM', random_state=0)
    nb = em.SVMMatcher(name='NaiveBayes', random_state=0)
    rf = em.RFMatcher(name='RF', random_state=0)
    lg = em.LogRegMatcher(name='LogReg', random_state=0)
    ln = em.LinRegMatcher(name='LinReg')
    algos = [dt, rf, svm, nb, ln, lg]

    result = em.select_matcher(
        algos, 
        table=feature_vectors, 
        exclude_attrs=redudant_attrs, 
        k=5, target_attr='gold_label', random_state=0)
    logger.info('matcher results:')
    print(result['cv_stats'])

    rf.fit(table=feature_vectors, exclude_attrs=['_id', 'gold_label'], target_attr='gold_label')
    rf.save(args.output_path)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help="Path to dataset directory", required=True)
    parser.add_argument('--conf_fn', type=str, help="Conference metadata filename", required=True)
    parser.add_argument('--jour_fn', type=str, help="Journal metadata filename", required=True)
    parser.add_argument('--train_fn', type=str, help="Name of the train file", required=True)
    parser.add_argument('--features_fn', type=str, help="Name of the features file", required=True)
    parser.add_argument('--output_path', type=str, help="Path to save matching model", required=True)
    args = parser.parse_args()
    main(args)

