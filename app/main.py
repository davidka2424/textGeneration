from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import gc
import logging

# Создаем приложение
app = FastAPI()

# Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Подключение статических файлов
app.mount("/static", StaticFiles(directory="static"), name="static")

# Подключение шаблонов
templates = Jinja2Templates(directory="templates")

# Глобальные переменные для модели
model = None
tokenizer = None

# Конфигурация
MAX_TOKENS_LIMIT = 300  # Лимит для слабого сервера

# Загрузка модели при старте
@app.on_event("startup")
async def load_model():
    global model, tokenizer
    try:
        logger.info("Загрузка модели...")
        tokenizer = GPT2Tokenizer.from_pretrained("./saved_model", padding_side='left')
        model = GPT2LMHeadModel.from_pretrained(
            "./saved_model",
            torch_dtype=torch.float32,  # Переключение на float32
            low_cpu_mem_usage=True
        )
        model.eval()
        logger.info("Модель успешно загружена!")
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}")
        raise e

# Проверка здоровья сервера
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Маршрут для главной страницы
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Модель запроса для генерации текста
class GenerateRequest(BaseModel):
    max_new_tokens: int = 100  # Лимит по умолчанию

# Генерация текста
@app.post("/generate")
async def generate_text(req: GenerateRequest):
    try:
        # Очистка памяти
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Настройка лимита токенов
        max_tokens = min(req.max_new_tokens, MAX_TOKENS_LIMIT)
        prompt = "В одном далёком королевстве жил мудрый король, который любил природу."
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Генерация текста
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_k=30,
                top_p=0.9,
                num_beams=1,  # Можно увеличить для поиска с бимом
            )

        # Постобработка результата
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        generated_text = generated_text.replace("�", "").strip()

        if '.' in generated_text:
            generated_text = generated_text[:generated_text.rfind('.') + 1]

        return {"result": generated_text}

    except Exception as e:
        logger.error(f"Ошибка генерации текста: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Точка входа для сервера
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
