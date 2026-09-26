SEED = 42
SUMMARIZATION = r'CarperAI/openai_summarize_tldr'
PUBRATING1 = r'codyburker/yelp_review_sampled'
PUBRATING2 = r'goosmanlei/amazon_reviews_multi'
SENTIMENT1 = r'stanfordnlp/sst2'
SENTIMENT2 = r'stanfordnlp/imdb'
MEDDIA = r'lavita/ChatDoctor-HealthCareMagic-100k'
TOPCLA1 = r'fancyzhx/dbpedia_14'
#TOPCLA2 = r'community-datasets/yahoo_answers_topics'不用
TOPCLA3 = r'fancyzhx/ag_news'
TRANSLATION = r'wmt/wmt19'
REPEAT = r'SetFit/qqp'
OUTDIR = r'../datas'

LABELED_DATASETS = [
    PUBRATING1, PUBRATING2, SENTIMENT1, SENTIMENT2,
    TOPCLA1, TOPCLA3, REPEAT,
]

NON_LABEL_DATASETS = [
    SUMMARIZATION, MEDDIA, TRANSLATION
]