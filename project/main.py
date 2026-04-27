# Импорт библиотек
import os
import json
import re
import pickle
import logging
import time
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from implicit.als import AlternatingLeastSquares
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from collections import Counter

os.environ["OPENBLAS_NUM_THREADS"] = "1"

# --- Настройка путей ---
DATA_PATH = "./src/data"
ARTIFACTS_PATH = "./artifacts"
SRC_MODELS_PATH = "./src/models"  # Путь к лучшим моделям
os.makedirs(ARTIFACTS_PATH, exist_ok=True)
os.makedirs(SRC_MODELS_PATH, exist_ok=True)

# ID модели Qwen (Можно заменить на другую, если система позволяет. Иначе будет очень долго генерировать ответ)
QWEN_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# --- Настройка логирования ---
def setup_logging():
    """Настройка структурированного логирования"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Консольный вывод
            logging.FileHandler('artifacts/app.log', encoding='utf-8')  # Файловый вывод
        ]
    )
    
    # Создаем logger для приложения
    logger = logging.getLogger('book_recommendation')
    logger.setLevel(logging.INFO)
    
    return logger

logger = setup_logging()


app_state = {
    "ratings_df": None,
    "books_df": None,
    "user_id_map": None,
    "book_id_map": None,
    "user_id_inv": None,
    "book_id_inv": None,
    # SVD
    "svd_user_factors": None,
    "svd_item_factors": None,
    # ALS
    "als_user_factors": None,
    "als_item_factors": None,
    "als_sparse_matrix": None, 
    # Qwen
    "qwen_model": None,
    "qwen_tokenizer": None,
}

class RecommendationRequest(BaseModel):
    user_id: int
    n_recommendations: int = 5

class BookRecommendation(BaseModel):
    book_id: int
    title: str
    authors: str
    score: Optional[float] = None
    average_rating: Optional[float] = None
    reason: Optional[str] = None 

class RecommendationResponse(BaseModel):
    user_id: int
    algorithm: str
    recommendations: List[BookRecommendation]

# --- Вспомогательные функции для тегов ---
def enrich_books_with_tags(books_df: pd.DataFrame, data_path: str) -> pd.DataFrame:
    """
    Функция пытается загрузить теги и объединить их с датафреймом книг.
    Если файлов нет, возвращает книги с пустым столбцом top_tags.
    """
    try:
        tags_path = os.path.join(data_path, "tags.csv")
        book_tags_path = os.path.join(data_path, "book_tags.csv")
        
        if os.path.exists(tags_path) and os.path.exists(book_tags_path):
            logger.info("Загрузка и обработка тегов...")
            tags = pd.read_csv(tags_path)
            book_tags = pd.read_csv(book_tags_path)
            
            # Сортировка, группировка и выдача топ-5 тегов
            book_tags_merged = book_tags.merge(tags, on="tag_id", how="left")
            top_tags = (
                book_tags_merged.sort_values(["goodreads_book_id", "count"], ascending=[True, False])
                .groupby("goodreads_book_id")["tag_name"]
                .apply(lambda x: ", ".join(x.dropna().astype(str).unique()[:5]))
                .reset_index()
                .rename(columns={"tag_name": "top_tags"})
            )
            
            # Объединяем с книгами (в books.csv должен быть book_id, связываем с goodreads_book_id)
            if 'goodreads_book_id' in books_df.columns:
                books_df = books_df.merge(top_tags, on="goodreads_book_id", how="left")
                books_df["top_tags"] = books_df["top_tags"].fillna("")
                # Удаляем дублирующую колонку после объединения
                if "goodreads_book_id" in books_df.columns:
                     books_df.drop(columns=["goodreads_book_id"], inplace=True)
            else:
                # Если goodreads_book_id нет, пробуем по book_id
                 books_df["top_tags"] = ""
        else:
            logger.warning("Файлы тегов не найдены, Qwen рекомендации будут работать без учета тегов.")
            books_df["top_tags"] = ""
    except Exception as e:
        logger.error(f"Ошибка при обработке тегов: {e}")
        books_df["top_tags"] = ""
        
    return books_df

def load_data():
    logger.info("Загрузка данных...")
    try:
        ratings = pd.read_csv(os.path.join(DATA_PATH, "ratings.csv"))
        books = pd.read_csv(os.path.join(DATA_PATH, "books.csv"))
        
        # Базовая очистка
        ratings = ratings.drop_duplicates(subset=['user_id', 'book_id'])
        books = books.drop_duplicates(subset='book_id')
        ratings = ratings.dropna(subset=['user_id', 'book_id', 'rating'])
        books = books.dropna(subset=['book_id', 'title', 'authors'])
        
        ratings["user_id"] = ratings["user_id"].astype(int)
        ratings["book_id"] = ratings["book_id"].astype(int)
        ratings["rating"] = ratings["rating"].astype(float)
        books["book_id"] = books["book_id"].astype(int)
        
        # --- Интеграция тегов для Qwen ---
        books = enrich_books_with_tags(books, DATA_PATH)
        
        return ratings, books
    except Exception as e:
        raise RuntimeError(f"Ошибка загрузки данных: {e}")

def create_sparse_matrix(ratings, user_id_map, book_id_map):
    rows = ratings['user_id'].map(user_id_map)
    cols = ratings['book_id'].map(book_id_map)
    data_vals = ratings['rating'].values.astype(np.float32)
    return csr_matrix((data_vals, (rows, cols)), 
                      shape=(len(user_id_map), len(book_id_map)))

def train_and_save_models(ratings: pd.DataFrame, books: pd.DataFrame):
    logger.info("Обучение моделей с оптимальными параметрами...")
    user_ids = ratings['user_id'].unique()
    book_ids = ratings['book_id'].unique()

    user_id_map = {uid: i for i, uid in enumerate(user_ids)}
    book_id_map = {bid: i for i, bid in enumerate(book_ids)}
    user_id_inv = {i: uid for uid, i in user_id_map.items()}
    book_id_inv = {i: bid for bid, i in book_id_map.items()}

    sparse_matrix = create_sparse_matrix(ratings, user_id_map, book_id_map)

    # --- SVD (лучшие параметры: n_components=64, n_iter=20) ---
    logger.info("Обучение SVD (n_components=64, n_iter=20)...")
    svd = TruncatedSVD(n_components=64, random_state=42, n_iter=20)
    svd_user_factors = svd.fit_transform(sparse_matrix)
    svd_item_factors = svd.components_.T

    svd_user_factors = normalize(svd_user_factors)
    svd_item_factors = normalize(svd_item_factors)

    # --- ALS (лучшие параметры: factors=64, regularization=0.05, iterations=50, alpha=2.0) ---
    logger.info("Обучение ALS (factors=64, regularization=0.05, iterations=50, alpha=2.0)...")
    als_model = AlternatingLeastSquares(
        factors=64, 
        regularization=0.05, 
        iterations=50,
        alpha=2.0
    )
    als_model.fit(sparse_matrix)

    als_user_factors = als_model.user_factors
    als_item_factors = als_model.item_factors

    # --- Сохранение артефактов ---
    np.save(os.path.join(SRC_MODELS_PATH, "svd_user_factors.npy"), svd_user_factors)
    np.save(os.path.join(SRC_MODELS_PATH, "svd_item_factors.npy"), svd_item_factors)
    np.save(os.path.join(SRC_MODELS_PATH, "als_user_factors.npy"), als_user_factors)
    np.save(os.path.join(SRC_MODELS_PATH, "als_item_factors.npy"), als_item_factors)

    # Сохраняем отдельные маппинги для SVD и ALS
    with open(os.path.join(SRC_MODELS_PATH, "svd_id_mappings.pkl"), 'wb') as f:
        pickle.dump({
            'user_id_map': user_id_map,
            'book_id_map': book_id_map,
            'user_id_inv': user_id_inv,
            'book_id_inv': book_id_inv
         }, f)
    
    with open(os.path.join(SRC_MODELS_PATH, "als_id_mappings.pkl"), 'wb') as f:
        pickle.dump({
            'user_id_map': user_id_map,
            'book_id_map': book_id_map,
            'user_id_inv': user_id_inv,
            'book_id_inv': book_id_inv
         }, f)
    
    # Также сохраняем в artifacts для обратной совместимости
    np.save(os.path.join(ARTIFACTS_PATH, "svd_user_factors.npy"), svd_user_factors)
    np.save(os.path.join(ARTIFACTS_PATH, "svd_item_factors.npy"), svd_item_factors)
    np.save(os.path.join(ARTIFACTS_PATH, "als_user_factors.npy"), als_user_factors)
    np.save(os.path.join(ARTIFACTS_PATH, "als_item_factors.npy"), als_item_factors)

    with open(os.path.join(ARTIFACTS_PATH, "id_mappings.pkl"), 'wb') as f:
        pickle.dump({
            'user_id_map': user_id_map,
            'book_id_map': book_id_map,
            'user_id_inv': user_id_inv,
            'book_id_inv': book_id_inv
         }, f)

    logger.info("Модели обучены и сохранены.")
    return (svd_user_factors, svd_item_factors, 
            als_user_factors, als_item_factors, 
            sparse_matrix,
            user_id_map, book_id_map, user_id_inv, book_id_inv)

def load_saved_models(ratings: pd.DataFrame):
    """Загрузка моделей из src/models"""
    logger.info("Поиск сохраненных моделей...")
    
    # Проверяем наличие моделей в src/models
    svd_user_path = os.path.join(SRC_MODELS_PATH, "svd_user_factors.npy")
    svd_item_path = os.path.join(SRC_MODELS_PATH, "svd_item_factors.npy")
    als_user_path = os.path.join(SRC_MODELS_PATH, "als_user_factors.npy")
    als_item_path = os.path.join(SRC_MODELS_PATH, "als_item_factors.npy")
    svd_mappings_path = os.path.join(SRC_MODELS_PATH, "svd_id_mappings.pkl")
    als_mappings_path = os.path.join(SRC_MODELS_PATH, "als_id_mappings.pkl")
    
    # Если модели есть в src/models, загружаем оттуда
    if all(os.path.exists(p) for p in [svd_user_path, svd_item_path, als_user_path, als_item_path]):
        logger.info("Загрузка моделей из src/models...")
        
        # Загружаем SVD
        with open(svd_mappings_path, 'rb') as f:
            svd_mappings = pickle.load(f)
        
        svd_user_factors = np.load(svd_user_path)
        svd_item_factors = np.load(svd_item_path)
        
        # Загружаем ALS
        with open(als_mappings_path, 'rb') as f:
            als_mappings = pickle.load(f)
        
        als_user_factors = np.load(als_user_path)
        als_item_factors = np.load(als_item_path)
        
        # Используем маппинги от SVD (они должны быть одинаковыми)
        user_id_map = svd_mappings['user_id_map']
        book_id_map = svd_mappings['book_id_map']
        user_id_inv = svd_mappings['user_id_inv']
        book_id_inv = svd_mappings['book_id_inv']
        
    # Иначе пробуем загрузить из artifacts (например пользователь сам туда загрузил модели)
    elif os.path.exists(os.path.join(ARTIFACTS_PATH, "svd_user_factors.npy")):
        logger.info("Загрузка моделей из artifacts...")
        with open(os.path.join(ARTIFACTS_PATH, "id_mappings.pkl"), 'rb') as f:
            mappings = pickle.load(f)
        
        svd_user_factors = np.load(os.path.join(ARTIFACTS_PATH, "svd_user_factors.npy"))
        svd_item_factors = np.load(os.path.join(ARTIFACTS_PATH, "svd_item_factors.npy"))
        als_user_factors = np.load(os.path.join(ARTIFACTS_PATH, "als_user_factors.npy"))
        als_item_factors = np.load(os.path.join(ARTIFACTS_PATH, "als_item_factors.npy"))
        
        user_id_map = mappings['user_id_map']
        book_id_map = mappings['book_id_map']
        user_id_inv = mappings['user_id_inv']
        book_id_inv = mappings['book_id_inv']
    else:
        return None

    sparse_mat = create_sparse_matrix(ratings, user_id_map, book_id_map)

    return (svd_user_factors, svd_item_factors, 
            als_user_factors, als_item_factors, 
            sparse_mat,
            user_id_map, book_id_map, 
            user_id_inv, book_id_inv)

# --- Логика для LLM ---
def build_user_profile(user_id: int, ratings_df: pd.DataFrame, books_df: pd.DataFrame) -> dict:
    """Функция создает профиль пользователя на основе его истории."""
    user_ratings = ratings_df[ratings_df['user_id'] == int(user_id)]
    if user_ratings.empty:
        return None

    # Объединяем с информацией о книгах (включая теги)
    history_books = user_ratings.merge(books_df[['book_id', 'title', 'authors', 'top_tags']], on="book_id", how="left")
    
    liked = history_books[history_books["rating"] >= 4.0].copy()
    disliked = history_books[history_books["rating"] <= 2.0].copy()
    
    def split_tags(series):
        out = []
        for x in series.fillna(""):
            if x.strip():
                out.extend([t.strip() for t in x.split(",") if t.strip()])
        return out

    liked_tags = split_tags(liked.get("top_tags", pd.Series([])))
    
    mean_rating = user_ratings['rating'].mean()
    
    return {
     "user_id": int(user_id),
     "mean_rating": mean_rating if not np.isnan(mean_rating) else 0.0,
     "liked_books": liked[['book_id', 'title', 'authors', 'rating']].head(10).to_dict("records"),
     "disliked_books": disliked[['book_id', 'title', 'authors', 'rating']].head(10).to_dict("records"), 
     "favorite_tags": dict(Counter(liked_tags).most_common(8)),
}

def build_candidates(profile: dict, books_df: pd.DataFrame, ratings_df: pd.DataFrame, n_candidates: int = 10) -> pd.DataFrame:
    """Функция отбирает кандидатов на основе популярности и пересечения тегов."""
    # Получаем ID книг, которые пользователь уже оценил
    rated_books = set(ratings_df[ratings_df['user_id'] == profile['user_id']]['book_id'].tolist())
    
    # Берем пул книг, которые пользователь не читал
    pool = books_df[~books_df["book_id"].isin(rated_books)].copy()
    
    fav_tags = list(profile["favorite_tags"].keys())
    fav_tag_set = set(fav_tags)
    
    # Считаем пересечение тегов
    if fav_tags and "top_tags" in pool.columns:
        pool["tag_overlap"] = pool["top_tags"].apply(
            lambda x: len(fav_tag_set.intersection({t.strip() for t in x.split(",") if t.strip()})) if pd.notna(x) else 0
        )
    else:
        pool["tag_overlap"] = 0

    # Нормализуем популярность 
    if "ratings_count" in pool.columns and pool["ratings_count"].max() > 0:
        pool["pop_norm"] = np.log1p(pool["ratings_count"])
        pool["pop_norm"] = (pool["pop_norm"] - pool["pop_norm"].min()) / (pool["pop_norm"].max() - pool["pop_norm"].min() + 1e-9)
    else:
        pool["pop_norm"] = 0.0
        
    if pool["tag_overlap"].max() > 0:
        pool["tag_norm"] = pool["tag_overlap"] / pool["tag_overlap"].max()
    else:
        pool["tag_norm"] = 0.0

    # Формула скоринга 
    pool["retrieval_score"] = (0.45 * pool["tag_norm"]) + (0.35 * pool["pop_norm"])
    
    # Возвращаем топ кандидатов
    cols = ["book_id", "title", "authors", "average_rating", "top_tags", "retrieval_score"]
    # Убедимся, что колонки существуют
    final_cols = [c for c in cols if c in pool.columns]
    return pool.sort_values("retrieval_score", ascending=False).head(n_candidates)[final_cols]

def make_prompt(profile: dict, candidates: pd.DataFrame, n: int = 5) -> list:
    # Формируем блок любимых книг с контекстом
    liked_books_txt = "\n".join(
        [f"- \"{b['title']}\" by {b['authors']} | Rating: {b['rating']:.1f}"
         for b in profile["liked_books"]]
    ) or "- нет данных"

    # Формируем блок нелюбимых книг
    disliked_books_txt = "\n".join(
        [f"- \"{b['title']}\" by {b['authors']} | Rating: {b['rating']:.1f}"
         for b in profile["disliked_books"]]
    ) or "- нет данных"

    fav_tags_list = list(profile["favorite_tags"].items())[:8]
    fav_tags_txt = ", ".join([f"{t} ({c})" for t, c in fav_tags_list]) or "- нет данных"

    authors_list = [b['authors'] for b in profile["liked_books"] if b['authors']]
    unique_authors = list(set(authors_list))[:5] # Берем до 5 уникальных авторов
    fav_authors_str = ", ".join(unique_authors) if unique_authors else "Нет данных"

    # Формируем список кандидатов
    cand_lines = []
    for _, row in candidates.iterrows():
        cand_lines.append(
            f"ID:{int(row['book_id'])}. \"{row['title']}\" by {row['authors']} | "
            f"AvgRating:{row['average_rating']:.2f} | Tags:{row['top_tags']}"
        )

    prompt = f"""
### Профиль пользователя
- Средний рейтинг: {profile["mean_rating"]:.2f}
- Любимые авторы (из истории): {fav_authors_str}
- Ключевые теги интересов: {fav_tags_txt}

### История высоких оценок (Что нравится)
{liked_books_txt}

### История низких оценок (Что НЕ нравится)
{disliked_books_txt}

### Список кандидатов для рекомендации
{chr(10).join(cand_lines)}

### Задача
Выбери top {n} книг из кандидатов.

### Правила выбора
1. **Автор:** Если среди кандидатов есть книги авторов, которые пользователь уже высоко оценивал, отдавай им приоритет.
2. **Жанр/Теги:** Сравнивай теги кандидатов с "Ключевыми тегами интересов" и тегами из "Истории высоких оценок". Ищи пересечения.
3. **Стиль:** Если пользователь любит сложные миры (теги: magic-system, epic, world-building), избегай простых романов.
4. **Исключения:** Не рекомендуй книги, которые пользователь уже читал (их нет в списке кандидатов, но будь внимателен).
5. **Reason:** Напиши причину на русском языке (1-2 предложения). Объясни, почему эта книга подходит, ссылаясь на конкретного автора, похожий жанр или тег из истории пользователя.
6. **Самопроверка:** Проверь, насколько рекомендуемая книга действительно подходит пользователю. Убедись, что похожие книги серии/автора нравятся читателю (оценка не ниже 4). Удели особое внимание тегу favorites.

### Формат ответа (строго JSON)
{{
  "user_id": {profile["user_id"]},
  "recommendations": [
    {{"book_id": 123, "title": "...", "authors": "...", "reason": "..."}},
    ...
  ]
}}
""".strip()

    return prompt

def extract_json_block(text: str) -> dict:
    """Парсит JSON из ответа модели, удаляя возможный markdown."""
    if not text: return {}
    clean_text = re.sub(r'```json\s*|\s*```', '', text, flags=re.IGNORECASE).strip()
    match = re.search(r'\{.*\}', clean_text, flags=re.DOTALL)
    if match:
        try: return json.loads(match.group(0))
        except: return {}
    return {}

async def load_qwen_model():
    """Асинхронная загрузка модели Qwen в состояние приложения."""
    logger.info(f"Загрузка модели {QWEN_MODEL_ID}...")
    try:
        # Конфигурация для 4-битного квантования
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_ID,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
        model.eval()
        
        app_state["qwen_model"] = model
        app_state["qwen_tokenizer"] = tokenizer
        logger.info("Модель Qwen загружена успешно.")
    except Exception as e:
        logger.error(f"Ошибка загрузки Qwen: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    ratings, books = load_data()
    app_state["ratings_df"] = ratings
    app_state["books_df"] = books
    
    # Пытаемся загрузить сохраненные модели
    models = load_saved_models(ratings)
    
    if models is not None:
        (svd_uf, svd_if, als_uf, als_if, sparse_mat, u_map, b_map, u_inv, b_inv) = models
        logger.info("Модели успешно загружены.")
    else:
        logger.info("Модели не найдены. Начинаем обучение...")
        (svd_uf, svd_if, als_uf, als_if, sparse_mat, u_map, b_map, u_inv, b_inv) = train_and_save_models(ratings, books)

    app_state.update({
        "svd_user_factors": svd_uf,
        "svd_item_factors": svd_if,
        "als_user_factors": als_uf,
        "als_item_factors": als_if,
        "als_sparse_matrix": sparse_mat,
        "user_id_map": u_map,
        "book_id_map": b_map,
        "user_id_inv": u_inv,
        "book_id_inv": b_inv
    })
    
    # Загружаем Qwen
    await load_qwen_model()

    yield
    logger.info("Отключение...")

app = FastAPI(title="Book Recommendation API", lifespan=lifespan)

# --- Middleware для логирования запросов ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Логирование всех HTTP запросов и ответов"""
    start_time = time.time()
    
    # Логируем запрос
    logger.info(f"Request: {request.method} {request.url.path} - Client: {request.client.host if request.client else 'unknown'}")
    
    try:
        response = await call_next(request)
        
        # Логируем ответ
        process_time = time.time() - start_time
        logger.info(f"Response: {response.status_code} - Time: {process_time:.3f}s - Path: {request.url.path}")
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Request failed: {request.method} {request.url.path} - Error: {str(e)} - Time: {process_time:.3f}s")
        raise

# --- Эндпоинты ---
@app.post("/recommend/svd", response_model=RecommendationResponse)
def get_svd_recommendations(req: RecommendationRequest):
    """Рекомендации с SVD (Singular Value Decomposition)."""
    start_time = time.time()
    logger.info(f"SVD recommendation request - User: {req.user_id}, Count: {req.n_recommendations}")
    
    try:
        user_id = req.user_id
        n_rec = req.n_recommendations
        u_map = app_state["user_id_map"]
        b_inv = app_state["book_id_inv"]
        svd_if = app_state["svd_item_factors"]
        svd_uf = app_state["svd_user_factors"]
        books_df = app_state["books_df"]
        ratings_df = app_state["ratings_df"]

        if user_id not in u_map:
            logger.warning(f"User {user_id} not found in SVD training data")
            raise HTTPException(status_code=404, detail="User not found in training data")

        user_idx = u_map[user_id]
        user_vec = svd_uf[user_idx]
        scores = svd_if @ user_vec

        top_indices = np.argsort(scores)[::-1]
        rated_books = set(ratings_df[ratings_df['user_id'] == user_id]['book_id'])

        recs = []
        for idx in top_indices:
            if len(recs) >= n_rec:
                break
            book_id = b_inv[idx]
            if book_id not in rated_books:
                details = get_book_details(book_id, books_df)
                if details:
                    recs.append(BookRecommendation(**details, score=float(scores[idx])))

        process_time = time.time() - start_time
        logger.info(f"SVD recommendation completed - User: {user_id}, Recommendations: {len(recs)}, Time: {process_time:.3f}s")
        return RecommendationResponse(user_id=user_id, algorithm="SVD", recommendations=recs)
        
    except HTTPException:
        raise
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"SVD recommendation failed - User: {req.user_id}, Error: {str(e)}, Time: {process_time:.3f}s")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/recommend/als", response_model=RecommendationResponse)
def get_als_recommendations(req: RecommendationRequest):
    """Рекомендации с ALS (Alternating Least Squares)."""
    start_time = time.time()
    logger.info(f"ALS recommendation request - User: {req.user_id}, Count: {req.n_recommendations}")
    
    try:
        user_id = req.user_id
        n_rec = req.n_recommendations
        u_map = app_state["user_id_map"]
        b_inv = app_state["book_id_inv"]
        als_uf = app_state["als_user_factors"]
        als_if = app_state["als_item_factors"]
        books_df = app_state["books_df"]
        ratings_df = app_state["ratings_df"]

        if user_id not in u_map:
            logger.warning(f"User {user_id} not found in ALS training data")
            raise HTTPException(status_code=404, detail="User not found in training data")

        user_idx = u_map[user_id]
        user_vec = als_uf[user_idx]
        scores = als_if @ user_vec

        top_indices = np.argsort(scores)[::-1]
        rated_books = set(ratings_df[ratings_df['user_id'] == user_id]['book_id'])

        recs = []
        for idx in top_indices:
            if len(recs) >= n_rec:
                break
            book_id = b_inv[idx]
            if book_id not in rated_books:
                details = get_book_details(book_id, books_df)
                if details:
                    recs.append(BookRecommendation(**details, score=float(scores[idx])))

        process_time = time.time() - start_time
        logger.info(f"ALS recommendation completed - User: {user_id}, Recommendations: {len(recs)}, Time: {process_time:.3f}s")
        return RecommendationResponse(user_id=user_id, algorithm="ALS", recommendations=recs)
        
    except HTTPException:
        raise
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"ALS recommendation failed - User: {req.user_id}, Error: {str(e)}, Time: {process_time:.3f}s")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/recommend/qwen", response_model=RecommendationResponse)
def get_qwen_recommendations(req: RecommendationRequest):
    """Рекомендации с использованием LLM (Qwen) с обоснованием."""
    start_time = time.time()
    logger.info(f"Qwen recommendation request - User: {req.user_id}, Count: {req.n_recommendations}")
    
    try:
        user_id = req.user_id
        n_rec = req.n_recommendations
        books_df = app_state["books_df"]
        ratings_df = app_state["ratings_df"]
        
        # Проверка наличия модели
        if app_state["qwen_model"] is None or app_state["qwen_tokenizer"] is None:
            logger.error("Qwen model not loaded for recommendation request")
            raise HTTPException(status_code=503, detail="Модель Qwen не загружена")

        # Проверка пользователя
        user_ratings = ratings_df[ratings_df['user_id'] == user_id]
        if user_ratings.empty:
            logger.warning(f"User {user_id} has no rating history")
            raise HTTPException(status_code=404, detail="User history is empty or user not found")

        # Профиль и кандидаты
        profile = build_user_profile(user_id, ratings_df, books_df)
        if profile is None:
            logger.error(f"Failed to build profile for user {user_id}")
            raise HTTPException(status_code=400, detail="Не удалось построить профиль")
        
        logger.info(f"Built profile for user {user_id} - Mean rating: {profile['mean_rating']:.2f}, Liked books: {len(profile['liked_books'])}")
        
        candidates = build_candidates(profile, books_df, ratings_df, n_candidates=n_rec * 3)
        logger.info(f"Generated {len(candidates)} candidates for user {user_id}")
        
        # Промпт
        messages = [
            {
                "role": "system",
                "content": (
                    "Ты — интеллектуальная система рекомендаций книг. "
                    "Твоя задача — выбрать из списка кандидатов книги, которые максимально соответствуют вкусам пользователя. "
                    "Анализируй авторов, жанры (теги) и стиль предыдущих высоких оценок пользователя. "
                    "Избегай книг, которые стилистически противоречат вкусу пользователя. "
                    "Верни строго JSON без пояснений."
                ),
            },
            {"role": "user", "content": make_prompt(profile, candidates, n=n_rec)}
        ]
        
        # Генерация
        tokenizer = app_state["qwen_tokenizer"]
        model = app_state["qwen_model"]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generation_start = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.1
            )
        generation_time = time.time() - generation_start

        generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        logger.info(f"Qwen generation completed in {generation_time:.3f}s for user {user_id}")
        
        # Парсинг ответа
        data = extract_json_block(generated)
        if not data:
            logger.error(f"Failed to parse Qwen response for user {user_id}")
            raise HTTPException(status_code=500, detail="Failed to parse model response")
        
        # Формирование ответа
        result_recs = []
        rec_data = data.get("recommendations", [])
        
        books_map = books_df.set_index('book_id').to_dict('index')
        
        for item in rec_data:
            try:
                bid = int(item.get("book_id"))
                book_info = books_df[books_df['book_id'] == bid]
                
                if not book_info.empty:
                    row = book_info.iloc[0]
                    result_recs.append(BookRecommendation(
                        book_id=bid,
                        title=str(row['title']),
                        authors=str(row['authors']),
                        score=None,
                        average_rating=float(row['average_rating']) if 'average_rating' in row and pd.notna(row['average_rating']) else None,
                        reason=item.get("reason", "Подходит по вашим интересам")
                    ))
                else:
                    logger.warning(f"Qwen recommended non-existent book_id: {bid}")
            except Exception as e:
                logger.warning(f"Error processing recommendation item: {e}")
                continue

        process_time = time.time() - start_time
        logger.info(f"Qwen recommendation completed - User: {user_id}, Recommendations: {len(result_recs)}, Total time: {process_time:.3f}s")
        
        return RecommendationResponse(user_id=user_id, algorithm="Qwen-LLM", recommendations=result_recs[:n_rec])
        
    except HTTPException:
        raise
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Qwen recommendation failed - User: {req.user_id}, Error: {str(e)}, Time: {process_time:.3f}s")
        raise HTTPException(status_code=500, detail="Internal server error")

def get_book_details(book_id: int, books_df: pd.DataFrame) -> dict:
    book_info = books_df[books_df['book_id'] == book_id]
    if book_info.empty:
        return None
    row = book_info.iloc[0]
    return {
        "book_id": int(book_id),
        "title": str(row['title']),
        "authors": str(row['authors']),
        "average_rating": float(row['average_rating']) if 'average_rating' in row and pd.notna(row['average_rating']) else None
    }

@app.get("/health")
def health_check():
    """Детальная проверка состояния системы"""
    try:
        status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "components": {}
        }
        
        # Проверка данных
        data_status = "loaded"
        if app_state.get("ratings_df") is None:
            data_status = "not_loaded"
        elif app_state["ratings_df"].empty:
            data_status = "empty"
            
        status["components"]["data"] = {
            "status": data_status,
            "ratings_count": len(app_state.get("ratings_df", pd.DataFrame())),
            "books_count": len(app_state.get("books_df", pd.DataFrame()))
        }
        
        # Проверка SVD модели
        svd_status = "loaded"
        if app_state.get("svd_user_factors") is None or app_state.get("svd_item_factors") is None:
            svd_status = "not_loaded"
        status["components"]["svd_model"] = {"status": svd_status}
        
        # Проверка ALS модели
        als_status = "loaded"
        if app_state.get("als_user_factors") is None or app_state.get("als_item_factors") is None:
            als_status = "not_loaded"
        status["components"]["als_model"] = {"status": als_status}
        
        # Проверка Qwen модели
        qwen_status = "loaded"
        if app_state.get("qwen_model") is None or app_state.get("qwen_tokenizer") is None:
            qwen_status = "not_loaded"
            status["status"] = "degraded"  # Система работает, но не полноценно
        status["components"]["qwen_model"] = {"status": qwen_status}
        
        # Проверка маппингов
        mappings_status = "loaded"
        if app_state.get("user_id_map") is None or app_state.get("book_id_map") is None:
            mappings_status = "not_loaded"
            status["status"] = "unhealthy"
        status["components"]["mappings"] = {"status": mappings_status}
        
        # Информация о системе
        status["system"] = {
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            "torch_version": torch.__version__ if torch else "not_installed",
            "device": str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
        }
        
        # Если есть критические проблемы, меняем статус
        if data_status == "not_loaded" or mappings_status == "not_loaded":
            status["status"] = "unhealthy"
            
        logger.info(f"Health check completed: {status['status']}")
        return status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)