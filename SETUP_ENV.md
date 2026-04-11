# Окружение и зависимости — полная инструкция

Путь к проекту (подставь свой, если другой):
```
c:\Users\1\OneDrive\Desktop\image_description-main
```
Дальше везде этот путь обозначен как `ПРОЕКТ`.

---

## 1. Открыть терминал в папке проекта

**Вариант A — через проводник**
1. Открой папку `ПРОЕКТ` в проводнике.
2. В адресной строке введи `cmd` и Enter — откроется cmd в этой папке.

**Вариант B — через cmd**
```cmd
cd /d "c:\Users\1\OneDrive\Desktop\image_description-main"
```

**Вариант C — через PowerShell**
```powershell
cd "c:\Users\1\OneDrive\Desktop\image_description-main"
```

---

## 2. Создать виртуальное окружение (если ещё нет)

```cmd
cd /d "c:\Users\1\OneDrive\Desktop\image_description-main"
python -m venv venv
```

Проверка: должна появиться папка `venv` с подпапками `Scripts`, `Lib` и т.д.

---

## 3. Активировать окружение

**CMD:**
```cmd
cd /d "c:\Users\1\OneDrive\Desktop\image_description-main"
venv\Scripts\activate.bat
```
В начале строки должно появиться `(venv)`.

**PowerShell:**
```powershell
cd "c:\Users\1\OneDrive\Desktop\image_description-main"
.\venv\Scripts\Activate.ps1
```

---

## 4. Поставить зависимости (обязательные)

Убедись, что окружение активировано (видно `(venv)`), затем:

```cmd
pip install -r requirements.txt
```

Если нужен PyTorch с CUDA (для дообучения на видеокарте):

```cmd
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
(Для CUDA 11.8 замени `cu121` на `cu118`.)

Потом дообучение/Unsloth (если нужно):

```cmd
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --no-deps
pip install unsloth_zoo trl peft accelerate bitsandbytes
```

---

## 5. Запуск приложения

**Через батник (рекомендуется):**
```cmd
cd /d "c:\Users\1\OneDrive\Desktop\image_description-main"
run.bat
```
Батник сам активирует venv, ставит зависимости из requirements.txt и запускает приложение.

**Вручную:**
```cmd
cd /d "c:\Users\1\OneDrive\Desktop\image_description-main"
venv\Scripts\activate.bat
python app.py
```

---

## 6. Опционально: flash-linear-attention (быстрый путь для Qwen3.5 MoE)

Нужен **CUDA Toolkit** с компилятором **nvcc** в PATH (не только драйвер).

1. Установи [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (версия совместимая с твоим драйвером).
2. Перезапусти терминал, проверь: `nvcc --version`.
3. Активируй окружение проекта и выполни:

```cmd
cd /d "c:\Users\1\OneDrive\Desktop\image_description-main"
venv\Scripts\activate.bat
pip install "flash-linear-attention[conv1d]"
```

Если сборка упадёт (часто на Windows) — ничего страшного, обучение работает и без этого, просто чуть медленнее.

---

## Краткая шпаргалка (всё по порядку)

```cmd
cd /d "c:\Users\1\OneDrive\Desktop\image_description-main"
venv\Scripts\activate.bat
pip install -r requirements.txt
python app.py
```

Или просто запусти **run.bat** из папки проекта.
