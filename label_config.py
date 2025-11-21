
ENTITY_LABELS = [
    'COMMODITY', 'CURRENCY', 'DATE', 'DOCUMENT_SECTION', 'EVENT',
    'EXCHANGE', 'FILING_TYPE', 'FINANCIAL_TERM', 'FISCAL_PERIOD',
    'INDEX', 'INSTRUMENT', 'LAW', 'LOCATION', 'METRIC', 'MONEY',
    'NUMBER', 'ORG', 'PERCENT', 'PERSON', 'POSITION', 'PRODUCT',
    'SECTOR', 'SENTIMENT', 'SOURCE', 'TABLE_TYPE', 'TICKER'
]

label2id = {
    "O": 0,
    "B-COMMODITY": 1,
    "I-COMMODITY": 2,
    "B-CURRENCY": 3,
    "I-CURRENCY": 4,
    "B-DATE": 5,
    "I-DATE": 6,
    "B-DOCUMENT_SECTION": 7,
    "I-DOCUMENT_SECTION": 8,
    "B-EVENT": 9,
    "I-EVENT": 10,
    "B-EXCHANGE": 11,
    "I-EXCHANGE": 12,
    "B-FILING_TYPE": 13,
    "I-FILING_TYPE": 14,
    "B-FINANCIAL_TERM": 15,
    "I-FINANCIAL_TERM": 16,
    "B-FISCAL_PERIOD": 17,
    "I-FISCAL_PERIOD": 18,
    "B-INDEX": 19,
    "I-INDEX": 20,
    "B-INSTRUMENT": 21,
    "I-INSTRUMENT": 22,
    "B-LAW": 23,
    "I-LAW": 24,
    "B-LOCATION": 25,
    "I-LOCATION": 26,
    "B-METRIC": 27,
    "I-METRIC": 28,
    "B-MONEY": 29,
    "I-MONEY": 30,
    "B-NUMBER": 31,
    "I-NUMBER": 32,
    "B-ORG": 33,
    "I-ORG": 34,
    "B-PERCENT": 35,
    "I-PERCENT": 36,
    "B-PERSON": 37,
    "I-PERSON": 38,
    "B-POSITION": 39,
    "I-POSITION": 40,
    "B-PRODUCT": 41,
    "I-PRODUCT": 42,
    "B-SECTOR": 43,
    "I-SECTOR": 44,
    "B-SENTIMENT": 45,
    "I-SENTIMENT": 46,
    "B-SOURCE": 47,
    "I-SOURCE": 48,
    "B-TABLE_TYPE": 49,
    "I-TABLE_TYPE": 50,
    "B-TICKER": 51,
    "I-TICKER": 52
}


id2label = {v: k for k, v in label2id.items()}


NUM_LABELS = len(label2id)


