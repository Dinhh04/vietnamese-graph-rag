"""
Generator cho 2 notebook Kaggle đã sửa lỗi.
Chạy:  python build_notebooks.py
Sinh ra: notebooks/kaggle_part1_embedding_ner.ipynb, notebooks/kaggle_part2_graph_rag.ipynb
"""
import json, os

def md(s):   return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}

def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "accelerator": "GPU",
        },
        "nbformat": 4, "nbformat_minor": 5,
    }

# ----------------------------------------------------------------------------
# Khối dùng chung: bảng ánh xạ keyword tiếng Việt -> aspect, gazetteer thương hiệu
# ----------------------------------------------------------------------------
KW_BLOCK = r'''ASPECTS = ['SCREEN','CAMERA','BATTERY','PERFORMANCE','STORAGE','DESIGN','PRICE','GENERAL','FEATURES','SER&ACC']

# FIX #3: bang anh xa tu-khoa tieng Viet -> aspect code (kich hoat graph context cho cau hoi tieng Viet)
ASPECT_KEYWORDS = {
    'SCREEN':      ['man hinh','màn hình','hien thi','hiển thị','cam ung','cảm ứng','tan so quet','do phan giai','độ phân giải'],
    'CAMERA':      ['camera','chup','chụp','anh','ảnh','quay','selfie','chup dem','chụp đêm','zoom','ong kinh','ống kính'],
    'BATTERY':     ['pin','sac','sạc','dung luong pin','trau','trâu','tut pin','tụt pin','chai pin'],
    'PERFORMANCE': ['hieu nang','hiệu năng','muot','mượt','lag','giat','giật','chip','ram','cau hinh','cấu hình','xu ly','chay','chạy','nhanh'],
    'STORAGE':     ['bo nho','bộ nhớ','luu tru','lưu trữ','dung luong','dung lượng','rom','gb','day bo nho'],
    'DESIGN':      ['thiet ke','thiết kế','kieu dang','kiểu dáng','dep','đẹp','mong','mỏng','cam','cầm','chat lieu','mau sac'],
    'PRICE':       ['gia','giá','tien','tiền','re','rẻ','dat','đắt','mac','mắc','hop ly','hợp lý','tam gia','gia thanh'],
    'FEATURES':    ['tinh nang','tính năng','loa','am thanh','âm thanh','wifi','song','sóng','bluetooth','van tay','vân tay','bao mat','nfc','cam bien','cảm biến'],
    'SER&ACC':     ['dich vu','dịch vụ','bao hanh','bảo hành','phu kien','phụ kiện','nhan vien','nhân viên','tu van','tư vấn','giao hang','giao hàng','shop','cua hang','cửa hàng','dong goi','đóng gói'],
    'GENERAL':     ['san pham','sản phẩm','may','máy','dien thoai','điện thoại','tong the','tổng thể','noi chung','on dinh','ổn'],
}

def aspects_from_text(text):
    """Aspect QUAN SAT duoc trong noi dung (tin hieu doc lap voi nhan vang)."""
    t = str(text).lower()
    return {a for a, kws in ASPECT_KEYWORDS.items() if any(k in t for k in kws)}

def aspect_from_query(query):
    """Aspect chinh ma cau hoi nham toi (nhieu keyword khop nhat)."""
    t = str(query).lower(); best, bn = None, 0
    for a, kws in ASPECT_KEYWORDS.items():
        n = sum(1 for k in kws if k in t)
        if n > bn: best, bn = a, n
    return best

BRAND_GAZETTEER = {
    'iphone':'Apple','apple':'Apple','samsung':'Samsung','galaxy':'Samsung','xiaomi':'Xiaomi','redmi':'Xiaomi',
    'poco':'Xiaomi','oppo':'OPPO','vivo':'Vivo','realme':'Realme','huawei':'Huawei','nokia':'Nokia',
    'vsmart':'VSmart','asus':'Asus','sony':'Sony','oneplus':'OnePlus',
}

def detect_brand(text):
    low = str(text).lower()
    for kw, b in BRAND_GAZETTEER.items():
        if kw in low: return b
    return 'Unknown'
'''

LOADER_BLOCK = r'''import os, re, json, warnings, urllib.request, zipfile, glob
import numpy as np
import pandas as pd
DATA_DIR = 'data'; os.makedirs(DATA_DIR, exist_ok=True)

def _find_csv(name):
    hits = glob.glob('/kaggle/input/**/' + name, recursive=True)
    if hits: return hits[0]
    for p in [DATA_DIR + '/' + name, DATA_DIR + '/raw/' + name, name]:
        if os.path.exists(p): return p
    return None

def load_split(name):
    p = _find_csv(name)
    if p is None:
        zp = DATA_DIR + '/UIT-ViSFD.zip'
        if not os.path.exists(zp):
            try:
                print('Downloading UIT-ViSFD.zip ...')
                urllib.request.urlretrieve('https://github.com/LuongPhan/UIT-ViSFD/raw/main/UIT-ViSFD.zip', zp)
            except Exception as e:
                print('zip download failed:', e)
        if os.path.exists(zp):
            try:
                with zipfile.ZipFile(zp) as z: z.extractall(DATA_DIR)
            except Exception as e:
                print('extract failed:', e)
        p = _find_csv(name)
    if p is None:
        p = DATA_DIR + '/' + name
        urllib.request.urlretrieve('https://raw.githubusercontent.com/kimkim00/UIT-ViSFD/main/' + name, p)
    return pd.read_csv(p)
'''

PARSE_BLOCK = r'''# FIX #1: regex giu ky tu '&' -> KHONG mat aspect SER&ACC (1995 lan trong train)
ASPECT_PATTERN = re.compile(r'\{([\w&]+)#(\w+)\}')

def parse_label(label_str):
    """'{CAMERA#Positive};{SER&ACC#Negative};{OTHERS};' -> [('CAMERA','Positive'),('SER&ACC','Negative')]"""
    if pd.isna(label_str):
        return []
    return [(m.group(1), m.group(2)) for m in ASPECT_PATTERN.finditer(str(label_str))]
'''

# ============================================================================
# PART 1
# ============================================================================
p1 = [
md(r'''# 🛒 Vietnamese Graph RAG — Part 1
## Data · Preprocessing · NER · So sánh Embedding (TF-IDF / Word2Vec / GloVe-SVD / PhoBERT)

**Đầu ra:** `data/train_processed.csv`, `results_part1.json`, các hình PNG.

> ⚙️ Bật **Internet** + **GPU (T4)** trong Settings của Kaggle trước khi chạy.
> So sánh embedding bằng **Precision@k & MRR** (nhãn aspect là ground-truth) — không dùng "avg cosine".'''),

md(r'''## 0. Cài đặt thư viện'''),
code(r'''!pip install -q underthesea gensim networkx
!pip install -q transformers'''),

code(LOADER_BLOCK + r'''
import matplotlib.pyplot as plt
from collections import Counter
warnings.filterwarnings('ignore')
try: plt.style.use('seaborn-v0_8-whitegrid')
except Exception: pass
np.random.seed(42)
print('Libraries loaded!')'''),

md(r'''## 1. Load dữ liệu UIT-ViSFD'''),
code(r'''train_df = load_split('Train.csv')
dev_df   = load_split('Dev.csv')
test_df  = load_split('Test.csv')
print(f'Train: {len(train_df)} | Dev: {len(dev_df)} | Test: {len(test_df)}')
print('Columns:', list(train_df.columns))
train_df.head(3)'''),

code(PARSE_BLOCK + r'''
for df in (train_df, dev_df, test_df):
    df['parsed_labels'] = df['label'].apply(parse_label)
    df['aspects']    = df['parsed_labels'].apply(lambda x: [a for a, s in x])
    df['sentiments'] = df['parsed_labels'].apply(lambda x: [s for a, s in x])

aspect_counts = Counter(a for lst in train_df['aspects'] for a in lst)
print('Phan bo aspect (train) — SER&ACC gio da xuat hien:')
for a, c in aspect_counts.most_common():
    print(f'  {a:12s}: {c}')
assert 'SER&ACC' in aspect_counts, "SER&ACC van bi mat - kiem tra regex!"
'''),

code(r'''fig, axes = plt.subplots(1, 2, figsize=(14, 5))
asp_df = pd.DataFrame(aspect_counts.most_common(), columns=['Aspect', 'Count'])
axes[0].barh(asp_df['Aspect'][::-1], asp_df['Count'][::-1], color='teal')
axes[0].set_title('Phan bo Aspect (train)')
if 'n_star' in train_df.columns:
    train_df['n_star'].value_counts().sort_index().plot(kind='bar', ax=axes[1], color='coral')
    axes[1].set_title('Phan bo so sao'); axes[1].set_xlabel('Stars')
plt.tight_layout(); plt.savefig('fig_aspect_distribution.png', dpi=150, bbox_inches='tight'); plt.show()'''),

md(r'''## 2. Tiền xử lý tiếng Việt'''),
code(r'''from underthesea import word_tokenize
STOPWORDS = set('va cua la co cho duoc trong voi khong nay cac mot nhung da khi de tu cung nhu nhung hay hoac vi nen thi ma rat lai bi do neu ve theo tai den con se dang ra vao len toi minh a'.split())
STOPWORDS |= set('và của là có cho được trong với không này các một những đã khi để từ cũng như nhưng hay hoặc vì nên thì mà rất lại bị do nếu về theo tại đến còn sẽ đang ra vào lên tôi mình ạ'.split())

URL_RE = re.compile(r'http\S+|www\S+'); NONWORD_RE = re.compile(r'[^\w\s]')
NUM_RE = re.compile(r'\d+');           SPACE_RE = re.compile(r'\s+')

def preprocess_vietnamese(text):
    if pd.isna(text): return ''
    t = URL_RE.sub(' ', str(text).lower().strip())
    t = NONWORD_RE.sub(' ', t); t = NUM_RE.sub(' ', t); t = SPACE_RE.sub(' ', t).strip()
    toks = word_tokenize(t, format='text').split()
    return ' '.join(w for w in toks if w not in STOPWORDS and len(w) > 1)

# PhoBERT nen nhan input da TACH TU (giu dau, giu so) -> seg() rieng
def seg(text):
    return word_tokenize(str(text)[:256], format='text')

for df in (train_df, dev_df, test_df):
    df['clean_text'] = df['comment'].apply(preprocess_vietnamese)
print('Preprocess xong.')
print('VD :', str(train_df.iloc[0]['comment'])[:80])
print(' ->', train_df.iloc[0]['clean_text'][:80])'''),

md(r'''## 3. Bảng ánh xạ từ-khóa tiếng Việt → aspect (dùng lại ở Part 2)'''),
code(KW_BLOCK + r'''
train_df['brand'] = train_df['comment'].apply(detect_brand)
print('Brand distribution:', dict(Counter(train_df['brand']).most_common()))
for q in ['camera chụp đêm có tốt không','pin dùng được bao lâu','giá có đắt quá không','nhân viên tư vấn nhiệt tình']:
    print(f'  {q!r:45s} -> {aspect_from_query(q)}')'''),

md(r'''## 4. NER + Gazetteer (trung thực với domain)

`underthesea.ner` là NER tiếng Việt tổng quát (PER/LOC/ORG) → bắt tốt **thương hiệu/cửa hàng**.
Thực thể e-commerce (dòng máy, sản phẩm) không nằm trong nhãn NER tổng quát → bổ sung **gazetteer**.
`PhoNER_COVID19` là NER domain COVID, **không phù hợp** review điện thoại nên không dùng để gán nhãn sản phẩm.'''),
code(r'''from underthesea import ner as uts_ner
from tqdm.auto import tqdm

def extract_entities(text):
    text = str(text); ents = []
    try:
        for tok in uts_ner(text[:400]):
            word, pos, chunk_tag, ner_tag = tok
            if ner_tag != 'O':
                ents.append({'text': word, 'type': ner_tag.split('-')[-1], 'source': 'ner'})
    except Exception:
        pass
    low = text.lower()
    for kw, brand in BRAND_GAZETTEER.items():
        if kw in low:
            ents.append({'text': brand, 'type': 'BRAND', 'source': 'gazetteer'})
    return ents

demo = 'iPhone 15 Pro Max của Apple chụp ảnh đẹp, mua tại Thế Giới Di Động giá 30 triệu'
print('Demo:', demo)
for e in extract_entities(demo): print('  ', e)

sample = train_df.head(800)
ents_all = [extract_entities(c) for c in tqdm(sample['comment'], desc='NER(sample)')]
ent_types = Counter(e['type'] for es in ents_all for e in es)
print('\nEntity types (800 mau):', dict(ent_types))'''),

md(r'''## 5. Embedding — xây corpus căn chỉnh với nhãn vàng'''),
code(r'''mask = train_df['clean_text'].str.len() > 5
corpus_df = train_df[mask].reset_index(drop=True)
corpus         = corpus_df['clean_text'].tolist()
corpus_raw     = [seg(c) for c in tqdm(corpus_df['comment'], desc='seg(PhoBERT)')]
corpus_aspects = [set(a) for a in corpus_df['aspects']]   # ground-truth / document
print(f'Corpus: {len(corpus)} documents (can chinh voi gold aspects)')'''),

md(r'''### 5.1 TF-IDF (Week 1 — Baseline)'''),
code(r'''from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
tfidf = TfidfVectorizer(max_features=10000)
tfidf_matrix = tfidf.fit_transform(corpus)
print('TF-IDF:', tfidf_matrix.shape)
def rank_tfidf(query, k):
    qv = tfidf.transform([preprocess_vietnamese(query)])
    s = cosine_similarity(qv, tfidf_matrix).flatten()
    return s.argsort()[::-1][:k]'''),

md(r'''### 5.2 Word2Vec (Week 2)'''),
code(r'''from gensim.models import Word2Vec
tok_corpus = [d.split() for d in corpus]
w2v = Word2Vec(tok_corpus, vector_size=100, window=5, min_count=2, workers=4, sg=1, epochs=20)
print('W2V vocab:', len(w2v.wv))
def doc2vec_w2v(doc):
    ws = [w for w in doc.split() if w in w2v.wv]
    return np.mean([w2v.wv[w] for w in ws], axis=0) if ws else np.zeros(w2v.vector_size)
w2v_vecs = np.vstack([doc2vec_w2v(d) for d in corpus]).astype('float32')
def rank_w2v(query, k):
    qv = doc2vec_w2v(preprocess_vietnamese(query)).reshape(1, -1)
    s = cosine_similarity(qv, w2v_vecs).flatten()
    return s.argsort()[::-1][:k]
for w in ['camera','pin','màn_hình','giá']:
    if w in w2v.wv: print(f'  {w} -> {[x for x,_ in w2v.wv.most_similar(w, topn=5)]}')'''),

md(r'''### 5.3 GloVe-SVD (Week 3 — count-based)

> ⚠️ Trung thực: đây là **xấp xỉ** kiểu GloVe — SVD trên ma trận đồng xuất hiện ở **mức tài liệu**,
> không phải GloVe cửa sổ ngữ cảnh gốc. Vẫn minh hoạ embedding count-based/global.'''),
code(r'''from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
cm = cv.fit_transform(corpus)
cooc = (cm.T @ cm).astype(float); cooc.setdiag(0); cooc.eliminate_zeros()
cooc.data = np.log1p(cooc.data)
svd = TruncatedSVD(n_components=100, random_state=42)
glove_emb = svd.fit_transform(cooc)
vocab = cv.get_feature_names_out(); w2i = {w: i for i, w in enumerate(vocab)}
def doc2vec_glove(doc):
    ids = [w2i[w] for w in doc.split() if w in w2i]
    return np.mean(glove_emb[ids], axis=0) if ids else np.zeros(100)
glove_vecs = np.vstack([doc2vec_glove(d) for d in corpus]).astype('float32')
def rank_glove(query, k):
    qv = doc2vec_glove(preprocess_vietnamese(query)).reshape(1, -1)
    s = cosine_similarity(qv, glove_vecs).flatten()
    return s.argsort()[::-1][:k]
print('GloVe-SVD vectors:', glove_vecs.shape)'''),

md(r'''### 5.4 PhoBERT (Week 5 — Transformer, SOTA)'''),
code(r'''import torch
from transformers import AutoModel, AutoTokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)
tok = AutoTokenizer.from_pretrained('vinai/phobert-base-v2')
phobert = AutoModel.from_pretrained('vinai/phobert-base-v2').to(device).eval()

@torch.no_grad()
def encode_phobert(texts, bs=32):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i+bs], padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        h = phobert(**enc).last_hidden_state
        m = enc['attention_mask'].unsqueeze(-1)
        out.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    return np.vstack(out).astype('float32')

phobert_vecs = encode_phobert(corpus_raw)
print('PhoBERT vectors:', phobert_vecs.shape)
def rank_phobert(query, k):
    qv = encode_phobert([seg(query)])
    s = cosine_similarity(qv, phobert_vecs).flatten()
    return s.argsort()[::-1][:k]'''),

md(r'''### 5.5 ⭐ So sánh công bằng: Precision@k & MRR

> FIX #5: bỏ "avg cosine" (không so sánh được giữa các không gian embedding khác nhau).
> Ground-truth: review *liên quan* tới truy vấn về aspect X nếu **nhãn vàng** của review chứa X.'''),
code(r'''EVAL_QUERIES = [
    ('camera chụp ảnh có đẹp không', 'CAMERA'),
    ('pin trâu dùng được bao lâu', 'BATTERY'),
    ('màn hình hiển thị có sắc nét không', 'SCREEN'),
    ('máy chạy có mượt hiệu năng mạnh không', 'PERFORMANCE'),
    ('giá cả có hợp lý hay đắt không', 'PRICE'),
    ('thiết kế đẹp cầm có thoải mái không', 'DESIGN'),
    ('bộ nhớ lưu trữ dung lượng bao nhiêu', 'STORAGE'),
    ('loa âm thanh và các tính năng', 'FEATURES'),
    ('nhân viên tư vấn dịch vụ bảo hành', 'SER&ACC'),
]
RANKERS = {'TF-IDF': rank_tfidf, 'Word2Vec': rank_w2v, 'GloVe-SVD': rank_glove, 'PhoBERT': rank_phobert}

def precision_at_k(ranked, asp, k): return sum(1 for i in ranked[:k] if asp in corpus_aspects[i]) / k
def mrr(ranked, asp):
    for r, i in enumerate(ranked, 1):
        if asp in corpus_aspects[i]: return 1.0 / r
    return 0.0

rows = []
for name, fn in RANKERS.items():
    p5 = p10 = mr = 0.0
    for q, asp in EVAL_QUERIES:
        ranked = list(fn(q, 50))
        p5 += precision_at_k(ranked, asp, 5); p10 += precision_at_k(ranked, asp, 10); mr += mrr(ranked, asp)
    n = len(EVAL_QUERIES)
    rows.append({'Method': name, 'P@5': p5/n, 'P@10': p10/n, 'MRR': mr/n})
metrics_df = pd.DataFrame(rows).set_index('Method').round(4)
print(metrics_df)'''),

code(r'''ax = metrics_df.plot(kind='bar', figsize=(10, 5))
ax.set_title('So sanh Embedding — Precision@k & MRR (nhan aspect la ground-truth)')
ax.set_ylabel('Score'); plt.xticks(rotation=0); plt.tight_layout()
plt.savefig('fig_embedding_comparison.png', dpi=150, bbox_inches='tight'); plt.show()

print('\n% ===== LaTeX (dan vao bao cao) =====')
print(r'\begin{tabular}{lrrr}\toprule')
print(r'Phuong phap & P@5 & P@10 & MRR \\ \midrule')
for m, r in metrics_df.iterrows():
    print(f"{m} & {r['P@5']:.3f} & {r['P@10']:.3f} & {r['MRR']:.3f} " + r'\\')
print(r'\bottomrule\end{tabular}')'''),

md(r'''## 6. Lưu kết quả cho Part 2 & báo cáo'''),
code(r'''keep = [c for c in ['index','comment','n_star','label','parsed_labels','aspects','sentiments','clean_text','brand'] if c in train_df.columns]
train_df[keep].to_csv('data/train_processed.csv', index=False)
results = {
    'n_train': int(len(train_df)),
    'aspect_counts': dict(aspect_counts),
    'embedding_metrics': metrics_df.reset_index().to_dict(orient='records'),
    'ner_entity_types': dict(ent_types),
}
with open('results_part1.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print('Saved: data/train_processed.csv + results_part1.json')
print(json.dumps(results['embedding_metrics'], ensure_ascii=False, indent=2))'''),
]

# ============================================================================
# PART 2
# ============================================================================
p2 = [
md(r'''# 🛒 Vietnamese Graph RAG — Part 2
## Knowledge Graph · Attention Re-ranking · Graph RAG (OpenAI)

> ⚙️ Bật **Internet** + **GPU**. OpenAI API key: Add-ons → Secrets → `OPENAI_API_KEY`
> (lấy tại https://platform.openai.com/api-keys). Không có key vẫn chạy được phần Retrieval + Graph.

Notebook **tự chứa** (tự tải & tính lại), không cần output của Part 1.'''),

md(r'''## 0. Cài đặt'''),
code(r'''!pip install -q underthesea gensim networkx openai
!pip install -q transformers'''),

code(LOADER_BLOCK + r'''
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
from underthesea import word_tokenize
warnings.filterwarnings('ignore')
np.random.seed(42)
print('Libraries loaded!')'''),

md(r'''## 1. Load dữ liệu + helpers (self-contained)'''),
code(PARSE_BLOCK + r'''
train_df = load_split('Train.csv')
train_df['parsed_labels'] = train_df['label'].apply(parse_label)
train_df['aspects'] = train_df['parsed_labels'].apply(lambda x: [a for a, s in x])
print('Loaded', len(train_df), 'reviews')

def seg(text): return word_tokenize(str(text)[:256], format='text')'''),

code(KW_BLOCK + r'''
train_df['brand'] = train_df['comment'].apply(detect_brand)
print('Brand dist:', dict(Counter(train_df['brand']).most_common()))'''),

md(r'''## 2. Knowledge Graph: brand → aspect → sentiment (đã gồm SER&ACC)'''),
code(r'''G = nx.DiGraph()
for a in ASPECTS: G.add_node(a, type='aspect', color='lightblue')
SENT_COLOR = {'Positive':'lightgreen','Negative':'lightcoral','Neutral':'khaki'}
for _, row in train_df.iterrows():
    brand = row['brand']
    if brand != 'Unknown' and not G.has_node(brand):
        G.add_node(brand, type='brand', color='gold')
    for asp, sent in row['parsed_labels']:
        if asp not in ASPECTS: continue
        sn = f'{asp}#{sent}'
        if not G.has_node(sn):
            G.add_node(sn, type='sentiment', sentiment=sent, color=SENT_COLOR.get(sent, 'lightgray'))
        if G.has_edge(asp, sn): G[asp][sn]['weight'] += 1
        else: G.add_edge(asp, sn, relation='has_sentiment', weight=1)
        if brand != 'Unknown':
            if G.has_edge(brand, asp): G[brand][asp]['weight'] += 1
            else: G.add_edge(brand, asp, relation='reviewed_on', weight=1)
print(f'KG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')
print('Node types:', dict(Counter(nx.get_node_attributes(G, 'type').values())))
assert any(str(n).startswith('SER&ACC') for n in G.nodes), "thieu SER&ACC trong KG!"
'''),

code(r'''fig, ax = plt.subplots(figsize=(15, 11))
pos = nx.spring_layout(G, k=2, seed=42)
colors = [G.nodes[n].get('color', 'white') for n in G.nodes()]
sizes  = [300 + G.degree(n) * 40 for n in G.nodes()]
ew     = [0.3 + G[u][v]['weight'] * 0.2 for u, v in G.edges()]
nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, alpha=0.85, ax=ax)
nx.draw_networkx_edges(G, pos, width=ew, alpha=0.35, edge_color='gray', arrows=True, arrowsize=12, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
ax.set_title('Knowledge Graph — Vietnamese E-commerce Reviews'); ax.axis('off')
plt.tight_layout(); plt.savefig('fig_knowledge_graph.png', dpi=150, bbox_inches='tight'); plt.show()'''),

code(r'''def graph_query(aspect):
    out = {}
    if aspect in G:
        for nb in G.successors(aspect):
            d = G.nodes[nb]
            if d.get('type') == 'sentiment':
                out[nb] = {'sentiment': d['sentiment'], 'count': G[aspect][nb]['weight']}
    return out
for a in ['CAMERA', 'BATTERY', 'SER&ACC']:
    print(a, '->', {k: v['count'] for k, v in graph_query(a).items()})'''),

md(r'''## 3. PhoBERT embedding (bi-encoder) + token-level cho attention'''),
code(r'''import torch
from transformers import AutoModel, AutoTokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tk = AutoTokenizer.from_pretrained('vinai/phobert-base-v2')
ph = AutoModel.from_pretrained('vinai/phobert-base-v2').to(device).eval()

mask_ok = train_df['comment'].astype(str).str.len() > 5
corpus_df = train_df[mask_ok].reset_index(drop=True)
from tqdm.auto import tqdm
corpus_raw     = [seg(c) for c in tqdm(corpus_df['comment'], desc='seg')]
corpus_aspects = [set(a) for a in corpus_df['aspects']]

@torch.no_grad()
def encode_mean(texts, bs=32):
    out = []
    for i in range(0, len(texts), bs):
        enc = tk(texts[i:i+bs], padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        h = ph(**enc).last_hidden_state; m = enc['attention_mask'].unsqueeze(-1)
        out.append(((h * m).sum(1) / m.sum(1)).cpu())
    return torch.cat(out)

print('Encoding corpus (mean-pool)...')
doc_vecs = encode_mean(corpus_raw)
doc_vecs_n = torch.nn.functional.normalize(doc_vecs, dim=1)
print('doc_vecs:', tuple(doc_vecs.shape))

@torch.no_grad()
def encode_tokens(text):
    enc = tk([text], padding=True, truncation=True, max_length=64, return_tensors='pt').to(device)
    h = ph(**enc).last_hidden_state[0]; m = enc['attention_mask'][0].bool()
    return torch.nn.functional.normalize(h[m], dim=1).cpu()'''),

md(r'''## 4. Attention Re-ranking (Week 4) — late-interaction MaxSim

> FIX #2: bỏ reranker `nn.Linear` **khởi tạo ngẫu nhiên / chưa train** (chỉ là nhiễu).
> Thay bằng **late-interaction MaxSim** (kiểu ColBERT): tương tác token-query × token-doc qua
> dot-product có chuẩn hoá (= cosine). Không tham số ngẫu nhiên, không cần huấn luyện, có ý nghĩa.'''),
code(r'''def maxsim(q_tok, d_tok):
    sim = q_tok @ d_tok.T                 # (Lq, Ld) — cosine vi da normalize
    return sim.max(dim=1).values.sum().item() / q_tok.shape[0]

def _nz(x):
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 1e-9 else x * 0.0

def hybrid_retrieve(query, top_k=5, n_cand=50, w_bi=0.4, w_attn=0.4, w_graph=0.2):
    qv = torch.nn.functional.normalize(encode_mean([seg(query)]), dim=1)
    bi = (doc_vecs_n @ qv.T).squeeze(1).numpy()
    cand = bi.argsort()[::-1][:n_cand]
    q_tok = encode_tokens(seg(query))
    attn = np.array([maxsim(q_tok, encode_tokens(corpus_raw[i])) for i in cand])
    # FIX #4 + chong ro ri: graph boost theo aspect QUAN SAT trong NOI DUNG review (khong dung nhan vang)
    q_asp = aspect_from_query(query)
    graph = np.array([1.0 if (q_asp and q_asp in aspects_from_text(corpus_raw[i])) else 0.0 for i in cand])
    combined = w_bi * _nz(bi[cand]) + w_attn * _nz(attn) + w_graph * graph
    order = combined.argsort()[::-1][:top_k]
    return [{'idx': int(cand[o]), 'text': corpus_raw[cand[o]],
             'aspects': list(corpus_aspects[cand[o]]), 'score': float(combined[o])} for o in order]

for r in hybrid_retrieve('camera chụp đêm có tốt không', top_k=3):
    print(f"[{r['score']:.3f}] {r['text'][:90]} | {r['aspects']}")'''),

md(r'''## 5. Graph RAG — sinh câu trả lời bằng OpenAI'''),
code(r'''from openai import OpenAI
OPENAI_KEY = None
try:
    from kaggle_secrets import UserSecretsClient
    OPENAI_KEY = UserSecretsClient().get_secret('OPENAI_API_KEY')
except Exception:
    OPENAI_KEY = os.environ.get('OPENAI_API_KEY')

client = None
LLM_MODEL = 'gpt-4o-mini'   # doi sang 'gpt-4o' / 'gpt-4.1-mini' / 'gpt-3.5-turbo' tuy key cua ban
if OPENAI_KEY:
    try:
        client = OpenAI(api_key=OPENAI_KEY)
        _ = client.chat.completions.create(model=LLM_MODEL,
                messages=[{'role': 'user', 'content': 'ping'}], max_tokens=5)
        print('LLM ready:', LLM_MODEL)
        try:
            gpts = sorted(m.id for m in client.models.list().data if 'gpt' in m.id)
            print('Model GPT kha dung cho key nay:', gpts[:15])
        except Exception:
            pass
    except Exception as e:
        print('OpenAI loi:', str(e)[:100]); client = None
if client is None:
    print('⚠️ Chua co OPENAI_API_KEY hop le -> bo qua Generation, van chay Retrieval + Graph.')'''),

code(r'''def graph_rag_answer(question, top_k=5):
    retrieved = hybrid_retrieve(question, top_k=top_k)
    asp = aspect_from_query(question)
    gctx = ''
    if asp:
        sd = graph_query(asp); tot = sum(v['count'] for v in sd.values())
        for n, inf in sorted(sd.items(), key=lambda x: -x[1]['count']):
            pct = 100 * inf['count'] / tot if tot else 0
            gctx += f"- {asp} / {inf['sentiment']}: {inf['count']} review ({pct:.0f}%)\n"
    ctx = '\n'.join(f"Review {i+1}: {r['text'][:160]}" for i, r in enumerate(retrieved))
    prompt = f"""Ban la tro ly tu van mua smartphone. CHI dua tren du lieu duoi day, tra loi ngan gon bang tieng Viet.

=== DANH GIA NGUOI DUNG ===
{ctx}

=== THONG KE TU KNOWLEDGE GRAPH ===
{gctx or 'Khong co thong ke cho khia canh nay.'}

=== CAU HOI ===
{question}

Tra loi:"""
    ans = '(LLM chua cau hinh — xem cac review truy xuat o duoi)'
    if client is not None:
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{'role': 'system', 'content': 'Ban la tro ly tu van mua smartphone, tra loi bang tieng Viet.'},
                          {'role': 'user', 'content': prompt}],
                temperature=0.3, max_tokens=400)
            ans = resp.choices[0].message.content
        except Exception as e:
            ans = f'(Loi LLM: {e})'
    return {'question': question, 'answer': ans, 'graph_context': gctx, 'retrieved': retrieved}

for q in ['Camera chụp đêm có tốt không?', 'Pin dùng được bao lâu?', 'Nhân viên tư vấn và bảo hành thế nào?']:
    r = graph_rag_answer(q)
    print('=' * 70); print('Q:', q); print('A:', r['answer'][:400])
    if r['graph_context']: print('Graph:', r['graph_context'].replace(chr(10), ' | '))'''),

md(r'''## 6. Đánh giá ablation (không rò rỉ metric)

> FIX #3: graph boost dùng aspect **quan sát trong nội dung** (độc lập), còn relevance chấm điểm dùng
> **nhãn vàng** → không còn vòng tròn. So sánh đóng góp từng thành phần trên cùng tập candidate.'''),
code(r'''def retrieve_components(query, n_cand=50):
    qv = torch.nn.functional.normalize(encode_mean([seg(query)]), dim=1)
    bi = (doc_vecs_n @ qv.T).squeeze(1).numpy()
    cand = bi.argsort()[::-1][:n_cand]
    q_tok = encode_tokens(seg(query))
    attn = np.array([maxsim(q_tok, encode_tokens(corpus_raw[i])) for i in cand])
    q_asp = aspect_from_query(query)
    graph = np.array([1.0 if (q_asp and q_asp in aspects_from_text(corpus_raw[i])) else 0.0 for i in cand])
    return cand, bi[cand], attn, graph

CONFIGS = {
    'Bi-encoder (PhoBERT)': (1.0, 0.0, 0.0),
    '+ Attention rerank':   (0.5, 0.5, 0.0),
    '+ Graph (Graph RAG)':  (0.4, 0.4, 0.2),
}
EVAL = [('camera chụp ảnh đẹp không','CAMERA'), ('pin trâu dùng lâu không','BATTERY'),
        ('màn hình hiển thị sắc nét','SCREEN'), ('máy chạy mượt hiệu năng','PERFORMANCE'),
        ('giá hợp lý hay đắt','PRICE'), ('thiết kế đẹp cầm thoải mái','DESIGN'),
        ('loa âm thanh tính năng','FEATURES'), ('nhân viên tư vấn bảo hành dịch vụ','SER&ACC')]

def p_at_k(order, asp, k): return sum(1 for i in order[:k] if asp in corpus_aspects[i]) / k
def mrr(order, asp):
    for r, i in enumerate(order, 1):
        if asp in corpus_aspects[i]: return 1.0 / r
    return 0.0

cache = {q: retrieve_components(q) for q, _ in EVAL}
rows = []
for name, (wb, wa, wg) in CONFIGS.items():
    p5 = p10 = mr = 0.0
    for q, asp in EVAL:
        cand, bi, attn, graph = cache[q]
        comb = wb * _nz(bi) + wa * _nz(attn) + wg * graph
        order = cand[comb.argsort()[::-1]]
        p5 += p_at_k(order, asp, 5); p10 += p_at_k(order, asp, 10); mr += mrr(order, asp)
    n = len(EVAL); rows.append({'Config': name, 'P@5': p5/n, 'P@10': p10/n, 'MRR': mr/n})
eval_df = pd.DataFrame(rows).set_index('Config').round(4)
print(eval_df)'''),

code(r'''ax = eval_df.plot(kind='bar', figsize=(10, 5))
ax.set_title('Ablation: dong gop cua Attention va Graph (ground-truth = nhan vang)')
ax.set_ylabel('Score'); plt.xticks(rotation=10, ha='right'); plt.tight_layout()
plt.savefig('fig_rag_vs_graphrag.png', dpi=150, bbox_inches='tight'); plt.show()

print('\n% ===== LaTeX =====')
print(r'\begin{tabular}{lrrr}\toprule')
print(r'Cau hinh & P@5 & P@10 & MRR \\ \midrule')
for m, r in eval_df.iterrows():
    print(f"{m} & {r['P@5']:.3f} & {r['P@10']:.3f} & {r['MRR']:.3f} " + r'\\')
print(r'\bottomrule\end{tabular}')

with open('results_part2.json', 'w', encoding='utf-8') as f:
    json.dump({'kg_nodes': G.number_of_nodes(), 'kg_edges': G.number_of_edges(),
               'ablation': eval_df.reset_index().to_dict(orient='records')}, f, ensure_ascii=False, indent=2)
print('Saved results_part2.json')'''),
]

os.makedirs('notebooks', exist_ok=True)
with open('notebooks/kaggle_part1_embedding_ner.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb(p1), f, ensure_ascii=False, indent=1)
with open('notebooks/kaggle_part2_graph_rag.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb(p2), f, ensure_ascii=False, indent=1)
print('OK: wrote part1 (%d cells), part2 (%d cells)' % (len(p1), len(p2)))
