import re
import os
import pandas as pd
import numpy as np
from pathlib import Path
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy
import json
from tqdm import tqdm
from datetime import datetime
import random


for resource in ['punkt', 'punkt_tab', 'stopwords', 'averaged_perceptron_tagger']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else 
                      f'corpora/{resource}' if resource == 'stopwords' else 
                      f'taggers/{resource}')
    except LookupError:
        nltk.download(resource)

class MasterFinancialPreprocessor:
    def __init__(self, base_dir='data'):
        self.base_dir = Path(base_dir)
        
        
        self.raw_dir = self.base_dir / 'raw'
        self.sec_dir = self.raw_dir / 'sec_filings'
        self.news_dir = self.raw_dir / 'financial_news'
        self.earnings_dir = self.raw_dir / 'earnings_call'
        
        
        self.processed_dir = self.base_dir / 'processed'
        self.cleaned_dir = self.processed_dir / 'cleaned'
        self.tokenized_dir = self.processed_dir / 'tokenized'
        self.augmented_dir = self.processed_dir / 'augmented'
        
        
        for directory in [self.cleaned_dir, self.tokenized_dir, self.augmented_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            print("Installing spaCy model...")
            os.system('python -m spacy download en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
        
        
        self.financial_terms = self.load_financial_terms()
        
        
        self.stopwords = set(stopwords.words('english'))
        self.financial_stopwords_to_keep = {
            'profit', 'loss', 'revenue', 'growth', 'debt', 'equity',
            'income', 'expense', 'asset', 'liability', 'cash', 'flow'
        }
        self.stopwords = self.stopwords - self.financial_stopwords_to_keep
    
    def load_financial_terms(self):
        
        return {
            'EBITDA': 'Earnings Before Interest Taxes Depreciation Amortization',
            'EPS': 'Earnings Per Share',
            'ROE': 'Return On Equity',
            'ROI': 'Return On Investment',
            'P/E': 'Price to Earnings Ratio',
            'IPO': 'Initial Public Offering',
            'M&A': 'Mergers and Acquisitions',
            'YoY': 'Year over Year',
            'QoQ': 'Quarter over Quarter',
            'FCF': 'Free Cash Flow',
            'CAGR': 'Compound Annual Growth Rate',
            'CAPEX': 'Capital Expenditure',
            'OPEX': 'Operating Expenditure',
             'EBIT' : 'Earnings Before Interest and Taxes',
             'P/B' :'Price-to-Book Ratio',
             'ROA' :'Return on Assets',
            'CAGR' :'Compound Annual Growth Rate',
            'DCF':'Discounted Cash Flow',
            'WACC':'Weighted Average Cost of Capital',
            'COGS':'Cost of Goods Sold',
              "EV": "Enterprise Value",
  "EV/EBITDA": "Enterprise Value to EBITDA",
  "ETF": "Exchange-Traded Fund",
  "GAAP": "Generally Accepted Accounting Principles",
  "GDP": "Gross Domestic Product",
  "CPI": "Consumer Price Index"

        }
    
    
    
    def clean_html(self, text):
        
        soup = BeautifulSoup(text, 'html.parser')
        
        
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text()
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def remove_sec_boilerplate(self, text):
        
        patterns = [
            r'UNITED STATES\s+SECURITIES AND EXCHANGE COMMISSION.*?Washington,?\s*D\.?C\.?\s*20549',
            r'Table of Contents',
            r'Page \d+ of \d+',
            r'XBRL.*?Viewer',
            r'Please enable JavaScript',
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)
        
        return text
    
    def preserve_financial_patterns(self, text):
        
        text = re.sub(r'\$(\d)', r'$ \1', text)
        text = re.sub(r'€(\d)', r'€ \1', text)
        text = re.sub(r'(\d+),(\d+)', r'\1\2', text)
        
        return text
    
    def extract_financial_sentences(self, text, max_sentences=100):

        sentences = sent_tokenize(text)
        
        financial_keywords = [
            'revenue', 'income', 'earnings', 'profit', 'loss',
            'billion', 'million', 'percent', '%', '$',
            'quarter', 'fiscal', 'year', 'shares', 'stock',
            'ebitda', 'eps', 'margin', 'growth', 'debt'
        ]
        
        relevant = []
        for sentence in sentences:
            if any(kw in sentence.lower() for kw in financial_keywords):
                if 30 < len(sentence) < 500:
                    relevant.append(sentence.strip())
                    if len(relevant) >= max_sentences:
                        break
        
        return relevant
    
    def process_sec_filings(self):
        
        print("\n" + "="*70)
        print("PROCESSING SEC FILINGS")
        print("="*70)
        
        html_files = list(self.sec_dir.glob('*.html'))
        
        if not html_files:
            print(f"No HTML files found in {self.sec_dir}")
            return []
        
        print(f"Found {len(html_files)} SEC filing(s)")
        
        all_sentences = []
        
        for file in tqdm(html_files, desc="Processing SEC filings"):
            try:
                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                    raw_text = f.read()
                
                
                if 'XBRL Viewer' in raw_text[:1000] and 'Please enable JavaScript' in raw_text[:1000]:
                    continue
                
                if len(raw_text) < 5000:
                    continue
                
                
                cleaned = self.clean_html(raw_text)
                cleaned = self.remove_sec_boilerplate(cleaned)
                cleaned = self.preserve_financial_patterns(cleaned)
                
               
                sentences = self.extract_financial_sentences(cleaned)
                all_sentences.extend(sentences)
                
                
                cleaned_file = self.cleaned_dir / f"sec_{file.stem}_cleaned.txt"
                with open(cleaned_file, 'w', encoding='utf-8') as f:
                    f.write(cleaned)
                
            except Exception as e:
                print(f"\nError processing {file.name}: {e}")
                continue
        
       
        sec_sentences_file = self.cleaned_dir / 'all_sec_sentences.txt'
        with open(sec_sentences_file, 'w', encoding='utf-8') as f:
            for sentence in all_sentences:
                f.write(sentence + '\n\n')
        
        print(f"Processed {len(html_files)} SEC filings")
        print(f"Extracted {len(all_sentences)} sentences")
        
        return all_sentences
 
    def process_reuters_news(self):
        
        print("\n" + "="*70)
        print("PROCESSING REUTERS NEWS")
        print("="*70)
        
        
        csv_files = list(self.news_dir.glob('*.csv'))
        
        if not csv_files:
            print(f"No CSV files found in {self.news_dir}")
            return []
        
        csv_file = csv_files[0]
        print(f"Loading: {csv_file.name}")
        
        df = pd.read_csv(csv_file)
        print(f"Total rows: {len(df)}")
        print(f"Columns: {df.columns.tolist()}")
        
        
        text_column = None
        for col in ['headline', 'title', 'text', 'content', 'description']:
            if col in df.columns:
                text_column = col
                break
        
        if not text_column:
            text_column = df.columns[0]  
        
        print(f"Using column: {text_column}")
        
        
        financial_keywords = [
            'revenue', 'earnings', 'profit', 'loss', 'billion', 'million',
            'stock', 'shares', 'merger', 'acquisition', 'ipo', 
            'quarter', 'fiscal', 'market', 'dividend', 'percent', '%', '$'
        ]
        
        pattern = '|'.join(financial_keywords)
        mask = df[text_column].str.lower().str.contains(pattern, na=False)
        filtered_df = df[mask]
        
        print(f"Filtered to {len(filtered_df)} financial headlines")
        
       
        filtered_df = filtered_df.drop_duplicates(subset=[text_column])
        print(f"After removing duplicates: {len(filtered_df)}")
        
       
        headlines = filtered_df[text_column].tolist()
        
        
        news_file = self.cleaned_dir / 'reuters_cleaned.txt'
        with open(news_file, 'w', encoding='utf-8') as f:
            for headline in headlines:
                f.write(str(headline) + '\n\n')
        
        print(f"✓ Saved {len(headlines)} headlines")
        
        return headlines
  
    def clean_earnings_transcript(self, text):
       
        text = re.sub(r'^[A-Z][a-zA-Z\s]+,\s+[A-Z]+:', '', text, flags=re.MULTILINE)
        
        
        text = re.sub(r'Operator:.*?(?=\n\n|\Z)', '', text, flags=re.DOTALL)
        
        
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def process_earnings_calls(self):
       
        print("\n" + "="*70)
        print("PROCESSING EARNINGS CALLS")
        print("="*70)
        
        transcript_files = list(self.earnings_dir.glob('*.txt'))
        
        if not transcript_files:
            print(f"No .txt files found in {self.earnings_dir}")
            return []
        
        print(f"Found {len(transcript_files)} transcript(s)")
        
        all_sentences = []
        
        for file in tqdm(transcript_files, desc="Processing earnings calls"):
            try:
                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                    raw_text = f.read()
                
               
                cleaned = self.clean_earnings_transcript(raw_text)
                
                
                sentences = self.extract_financial_sentences(cleaned)
                all_sentences.extend(sentences)
                
                
                cleaned_file = self.cleaned_dir / f"earnings_{file.stem}_cleaned.txt"
                with open(cleaned_file, 'w', encoding='utf-8') as f:
                    f.write(cleaned)
                
            except Exception as e:
                print(f"\nError processing {file.name}: {e}")
                continue
        
        
        earnings_file = self.cleaned_dir / 'all_earnings_sentences.txt'
        with open(earnings_file, 'w', encoding='utf-8') as f:
            for sentence in all_sentences:
                f.write(sentence + '\n\n')
        
        print(f"✓ Processed {len(transcript_files)} transcripts")
        print(f"✓ Extracted {len(all_sentences)} sentences")
        
        return all_sentences
    
 
    def tokenize_text(self, text):
        if len(text) > 1000000:
            text = text[:1000000]
        
        doc = self.nlp(text)
        tokens = [token.text.lower() for token in doc if not token.is_space]
        
        return tokens
    
    def tokenize_all_sentences(self, sentences, source_name):
    
        print(f"\nTokenizing {source_name}...")
        
        tokenized_data = []
        
        for sentence in tqdm(sentences[:1000], desc=f"Tokenizing {source_name}"):  # Limit for speed
            tokens = self.tokenize_text(sentence)
            tokenized_data.append({
                'text': sentence,
                'tokens': tokens,
                'num_tokens': len(tokens)
            })
        
        
        output_file = self.tokenized_dir / f'{source_name}_tokenized.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(tokenized_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved tokenized data: {output_file}")
        
        return tokenized_data

    
    def augment_sentence(self, sentence):

        augmented = []
        
        
        augmented.append(sentence)
        
        
        replacements = {
            'reported': ['announced', 'stated', 'disclosed'],
            'increased': ['rose', 'grew', 'climbed'],
            'decreased': ['fell', 'dropped', 'declined'],
            'company': ['firm', 'corporation', 'business'],
        }
        
        for original, synonyms in replacements.items():
            if original in sentence.lower():
                for synonym in synonyms:
                    new_sentence = re.sub(r'\b' + original + r'\b', synonym, sentence, flags=re.IGNORECASE)
                    if new_sentence != sentence:
                        augmented.append(new_sentence)
                        break
        
        return augmented
    
    def augment_sentences(self, sentences, source_name, num_samples=500):
        
        print(f"\nAugmenting {source_name}...")
        
        
        sample_sentences = random.sample(sentences, min(num_samples, len(sentences)))
        
        augmented_data = []
        
        for sentence in tqdm(sample_sentences, desc=f"Augmenting {source_name}"):
            variations = self.augment_sentence(sentence)
            for variation in variations:
                augmented_data.append({
                    'original': sentence,
                    'augmented': variation,
                    'source': source_name
                })
        
        
        output_file = self.augmented_dir / f'{source_name}_augmented.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(augmented_data, f, indent=2, ensure_ascii=False)
        
        print(f"Created {len(augmented_data)} augmented samples")
        
        return augmented_data
 
    def combine_all_sentences(self, sec_sentences, news_sentences, earnings_sentences):
        
        print("\n" + "="*70)
        print("COMBINING ALL SENTENCES")
        print("="*70)
        
        all_sentences = []
        
        
        for sentence in sec_sentences:
            all_sentences.append({'text': sentence, 'source': 'SEC'})
        
        for sentence in news_sentences:
            all_sentences.append({'text': sentence, 'source': 'News'})
        
        for sentence in earnings_sentences:
            all_sentences.append({'text': sentence, 'source': 'Earnings'})
        
        
        seen = set()
        unique_sentences = []
        for item in all_sentences:
            if item['text'] not in seen:
                seen.add(item['text'])
                unique_sentences.append(item)
        
        print(f"Total unique sentences: {len(unique_sentences)}")
        
        
        good_sentences = [s for s in unique_sentences if 30 < len(s['text']) < 300]
        
        print(f"Sentences with good length: {len(good_sentences)}")
        
        
        if len(good_sentences) > 2000:
            annotation_sentences = random.sample(good_sentences, 2000)
        else:
            annotation_sentences = good_sentences
        
        
        output_file = self.cleaned_dir / 'sentences_for_annotation.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, item in enumerate(annotation_sentences, 1):
                f.write(f"{i}. [{item['source']}] {item['text']}\n\n")
        
        
        json_file = self.cleaned_dir / 'sentences_for_annotation.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(annotation_sentences, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(annotation_sentences)} sentences for annotation")
        print(f"Text file: {output_file}")
        print(f"JSON file: {json_file}")
        
        return annotation_sentences
    
  
    
    def generate_statistics(self, sec_sentences, news_sentences, earnings_sentences):
    
        print("PREPROCESSING STATISTICS")
     
        
        stats = {
            'SEC Filings': {
                'count': len(list(self.sec_dir.glob('*.html'))),
                'sentences': len(sec_sentences)
            },
            'News Articles': {
                'count': 1,
                'sentences': len(news_sentences)
            },
            'Earnings Calls': {
                'count': len(list(self.earnings_dir.glob('*.txt'))),
                'sentences': len(earnings_sentences)
            },
            'Total': {
                'documents': (len(list(self.sec_dir.glob('*.html'))) + 
                             len(list(self.earnings_dir.glob('*.txt'))) + 1),
                'sentences': len(sec_sentences) + len(news_sentences) + len(earnings_sentences)
            }
        }
        
        for source, data in stats.items():
            print(f"\n{source}:")
            for key, value in data.items():
                print(f"  {key}: {value:,}")
        
        
        stats_file = self.processed_dir / f'preprocessing_stats_{datetime.now().strftime("%Y%m%d")}.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats
    

    def process_all(self):
  
        print("MASTER FINANCIAL DATA PREPROCESSOR")
   
        sec_sentences = self.process_sec_filings()
        
        news_sentences = self.process_reuters_news()
        
        earnings_sentences = self.process_earnings_calls()
        
        if sec_sentences:
            self.tokenize_all_sentences(sec_sentences, 'sec')
        if news_sentences:
            self.tokenize_all_sentences(news_sentences, 'news')
        if earnings_sentences:
            self.tokenize_all_sentences(earnings_sentences, 'earnings')
        
        if sec_sentences:
            self.augment_sentences(sec_sentences, 'sec', num_samples=200)
        if news_sentences:
            self.augment_sentences(news_sentences, 'news', num_samples=200)
        if earnings_sentences:
            self.augment_sentences(earnings_sentences, 'earnings', num_samples=200)
        
        annotation_data = self.combine_all_sentences(sec_sentences, news_sentences, earnings_sentences)
        
        self.generate_statistics(sec_sentences, news_sentences, earnings_sentences)
        
        print("PREPROCESSING COMPLETE!")

        print("\nOutput locations:")
        print(f"  Cleaned: {self.cleaned_dir}")
        print(f"  Tokenized: {self.tokenized_dir}")
        print(f"  Augmented: {self.augmented_dir}")
        print("\nNext step: Annotate sentences in:")
        print(f"  {self.cleaned_dir / 'sentences_for_annotation.txt'}")

def main():
    preprocessor = MasterFinancialPreprocessor(base_dir='data')
    preprocessor.process_all()

if __name__ == "__main__":
    main()