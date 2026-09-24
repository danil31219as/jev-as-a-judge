# Судья по шкале оценок на Qwen и Halo

Это **классификация с переменным числом вариантов оценки**. Каждый вариант
представлен отдельным вызовом `score`, а его логит берётся из скрытого состояния
токена `</tool_call>`. Одна скалярная голова `nn.Linear(hidden_size, 1)` считает
логит на каждом токене. Для логитов кандидатов считается cross entropy с
распределением оценок ассесоров, а `predict_proba` возвращает softmax. При шкале 0–2 получаются
три класса. При другой рубрике число вызовов
меняется, а отсутствующие варианты маскируются при батчинге.

Это адаптация идеи [Laya](https://github.com/NandhaKishorM/laya): один маркер и
один логит на вариант. По умолчанию Qwen обучается по soft-target cross entropy;
опция `--laya-rl` добавляет к ней вариант RLCD из Laya. Шкала содержится только в аргументах вызовов `score`;
в сообщении пользователя остаются критерий и, если есть, эталонный ответ.
У причинной модели логит раннего варианта не видит описания следующих
вариантов. В обучении порядок вызовов перемешивается детерминированно, а
целевой вектор вероятностей переставляется в том же порядке.

## Данные

По умолчанию `train_judge.py` берёт ровно 100 строк из открытого
[POLLUX](https://huggingface.co/datasets/ai-forever/POLLUX). У источника есть
только split `test`; скрипт создаёт собственные **train (80)** и **test (20)**.
В test попадают только восемь типов из `TEST_TASK_TYPES` по полю `task_type`,
по умолчанию не менее двух строк каждого типа. Эти типы полностью исключены из train.
Строки читаются потоково и перемешиваются с фиксированным seed. Рубрика каждой
строки определяет число и описания кандидатов. Цель —
эмпирическое распределение оценок из `annotations`: число голосов за каждую
оценку делится на число допустимых голосов. Например, `[0, 1, 1]` даёт
`[1/3, 2/3, 0]` для шкалы 0–2. Значения вне шкалы (включая `-1`) и некорректные
оценки исключаются; строка пропускается, только если допустимых голосов нет,
рубрика неясна или токенизированная последовательность длиннее **4096 токенов**.
Последовательности не обрезаются. При ничьей строка
сохраняется. После перемешивания вызовов вероятности переставляются в том же
порядке; отсутствующие варианты дополняются нулями до `max_options`.
`criteria_score` не используется как цель. Ни один вызов `score` не
считается уже выбранным ответом: это только варианты. В test вызовы идут по
возрастанию оценки, чтобы столбец логитов совпадал с числовой оценкой.

Опция `--full-dataset` проходит по **всему** исходному split POLLUX. После
проверки рубрик, голосов и ограничения 4096 токенов все пригодные строки восьми
типов из `TEST_TASK_TYPES` составляют test, а все остальные пригодные строки —
train. Лимиты `--samples` и `--test-samples` в этом режиме не используются.
Скрипт сохраняет токенизированные строки в `prepared/train.jsonl` и
`prepared/test.jsonl`, а числа и состав выборок — в `selection.json`. Подготовку
нужно выполнить **одним процессом** до запуска на двух GPU; при обучении оба
процесса читают готовые файлы. Параметр `--prepared-data-dir` позволяет
использовать это разбиение повторно с другим `--output-dir`, например для
отдельного запуска RLCD.

После каждой эпохи тренер считает на test **MAE**, **RMSE** и **macro-F1**
и записывает `eval_mae`, `eval_rmse`, `eval_f1_macro` в логи. После обучения
последние значения сохраняются в `test_metrics.json` как `test_*`:
все три метрики сравнивают оценку `argmax` модели с оценкой, за которую
проголосовало больше всего ассесоров. При равенстве голосов или логитов
выбирается меньшая оценка.
Также логируются `eval_predicted_unique_scores` (сколько разных оценок
предсказано через argmax) и `eval_predicted_score_N_count` (сколько раз
предсказана каждая оценка N, включая нулевые значения). Они тоже сохраняются
в `test_metrics.json` с префиксом `test_`.
Тест используется только для оценки, не для обновления весов.

Опция `--laya-rl` добавляет к CE алгоритм RLCD из
[Laya](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb): четыре
гауссовых возмущения логитов на пример, награда из log score, spherical score
и ordinal RPS, групповое центрирование преимущества и policy gradient.
Стандартное отклонение шума снижается от 0.4 до 0.1 по эпохам. Для RPS
варианты упорядочиваются по числовой оценке, независимо от порядка вызовов.
В `configs/full_h100_rlcd_hard.toml` задано `hard_labels = true`: и CE, и
RLCD используют one-hot цель из оценки большинства ассесоров. При равенстве
голосов берётся меньшая числовая оценка. Подготовленные данные не изменяются;
преобразование меток выполняется при расчёте loss. Метрики всегда сравнивают
argmax модели с оценкой большинства.

Авторы POLLUX [не рекомендуют обучаться на нём](https://huggingface.co/datasets/ai-forever/POLLUX/blob/main/README.md#out-of-scope-use),
чтобы сохранить независимость бенчмарка. Этот запуск выполняет ваш запрос как
учебный эксперимент. Результат на POLLUX после такого обучения нельзя считать
независимой оценкой качества.

## Конфигурация и запуск

Параметры данных, обучения, RLCD и ClearML находятся в TOML-файлах каталога
`configs/`. Программа читает их через `--config`; `--prepare-only` запускает
только подготовку данных. `configs/sample.toml` и `configs/sample_rlcd.toml`
задают пробный запуск на 100 строках. Для всего датасета и двух H100 служат
`configs/full_h100.toml`, `configs/full_h100_rlcd.toml` и
`configs/full_h100_rlcd_hard.toml`. Для Qwen3.5-4B есть конфиги
`configs/full_h100_4b.toml` и `configs/full_h100_4b_rlcd_hard.toml`.
Изменяйте гиперпараметры
в этих файлах. Например, при нехватке памяти установите
`per_device_train_batch_size = 1` и `gradient_accumulation_steps = 16`.

При первом запуске три Parquet-файла POLLUX (около 400 МБ) загружаются в
`source_dir = "checkpoints/pollux-source"` и проверяются по SHA-256. Пути
закреплены за конкретной ревизией датасета, поэтому подготовка не запрашивает
список файлов через API Hugging Face, который может отвечать HTTP 429.
Повторный запуск использует проверенные файлы из кэша. Если сервер ограничит
даже прямую загрузку, скопируйте эти три файла из той же ревизии в `source_dir`.

Скрипт использует `ClassificationTrainer` из [Halo](https://github.com/whitecircle/halo)
напрямую: стандартная команда `halo launch classification` не извлекает логиты
на `</tool_call>`.
Judge наследует полную архитектуру `Qwen3_5ForConditionalGeneration`, чтобы
языковые веса из исходного мультимодального чекпоинта загружались по тем же
именам (`model.language_model.*`). Обработка примеров остаётся текстовой.
При загрузке скрипт останавливается, если веса backbone отсутствуют или не
соответствуют чекпоинту. Новый `score`-слой инициализируется отдельно.

### Установка без Docker на Linux-сервере с H100

Нужны Python 3.12, драйвер NVIDIA с поддержкой CUDA 13 и достаточно места
для кэша датасета и чекпоинтов. `nvcc` не требуется: Qwen3.5 использует
[PyTorch-реализацию свёртки](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen3_5/modeling_qwen3_5.py),
если расширение `causal-conv1d` не установлено. Этот путь может быть медленнее.
Halo официально
[поставляется в контейнере](https://github.com/whitecircle/halo/blob/main/human-docs/installation.md);
ниже — установка его исходников и зависимостей в отдельное Python-окружение
для этой плотной модели и параллельного обучения. Клон Halo должен лежать рядом
с этим репозиторием. Скрипт сам добавляет его корень в путь импорта, потому что
`src` в Halo импортируется как пакет от корня клона. Если клон находится в другом
месте, задайте `HALO_REPO_DIR=/путь/к/halo` в окружении перед запуском.

Halo 1.0.0 объявляет `gram-newton-schulz` обязательной зависимостью. Она
подтягивает CUDA 12.9 bindings, несовместимые с `torch==2.11.0+cu130`.
В этом проекте используется AdamW, а не Muon: пакет `gram-newton-schulz`
нужен Halo только для импорта модуля оптимизаторов. Поэтому основные зависимости
устанавливаются из `requirements.txt`, затем `gram-newton-schulz` и Halo
устанавливаются с `--no-deps`. CUDA-библиотеки Muon при этом не требуются.

```bash
git clone https://github.com/whitecircle/halo.git
git -C halo checkout v1.0.0
git clone https://github.com/danil31219as/jev-as-a-judge.git
cd jev-as-a-judge
nvidia-smi -L

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel hatchling
python -m pip install 'torch==2.11.0+cu130' 'torchvision==0.26.0+cu130' \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install --no-build-isolation -r requirements.txt
python -m pip install --no-build-isolation --no-deps \
  'gram-newton-schulz==0.1.6' -e ../halo

python -c 'import torch, train_judge; train_judge.ensure_halo_source_path(); from src.trainers.reward.classification import ClassificationTrainer; print(torch.__version__, torch.cuda.device_count())'
python -m unittest discover -s tests -v
```

Если используете `uv` в уже активированном Python 3.12 окружении, после
установки CUDA PyTorch выполните аналогичные команды:

```bash
uv pip install --upgrade setuptools wheel hatchling
uv pip install --no-build-isolation -r requirements.txt
uv pip install --no-build-isolation --no-deps \
  'gram-newton-schulz==0.1.6' -e ../halo
python -c 'import torch, train_judge; train_judge.ensure_halo_source_path(); from src.trainers.reward.classification import ClassificationTrainer; print(torch.__version__, torch.cuda.device_count())'
```

В `configs/full_h100.toml` включён ClearML. Настройте подключение один раз
командой `clearml-init`; если оно не нужно, задайте `clearml = false` в обоих
полных конфигах и пропустите `clearml-init`. Режим `--prepare-only` не создаёт
задачу ClearML. При обучении туда отправляются только текстовые логи и числовые
метрики тренера (включая loss и learning rate), а после каждой эпохи — тестовые
MAE/RMSE/macro-F1. Файлы данных, `selection.json`, TOML-конфиг,
`test_metrics.json` и чекпоинты остаются локально.

### Полный POLLUX на двух H100

Подготовка выполняется один раз одним процессом. Обе тренировки используют
одно и то же разбиение в `checkpoints/pollux-full-data`:

Во время подготовки сообщение `Scanned N rows: train=X, test=Y` появляется
каждые 5000 просмотренных строк исходного датасета. `X` и `Y` — количество
строк, прошедших фильтры и добавленных в соответствующие выборки.

```bash
clearml-init
mkdir -p checkpoints/hf-cache checkpoints/tmp
export HF_HOME="$PWD/checkpoints/hf-cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TMPDIR="$PWD/checkpoints/tmp"
export CUDA_VISIBLE_DEVICES=0,1

python train_judge.py --config configs/full_h100.toml --prepare-only
torchrun --standalone --nproc_per_node=2 train_judge.py \
  --config configs/full_h100.toml
cat checkpoints/pollux-full-ce/test_metrics.json
```

Отдельный запуск с RLCD на тех же подготовленных данных:

```bash
torchrun --standalone --nproc_per_node=2 train_judge.py \
  --config configs/full_h100_rlcd.toml
cat checkpoints/pollux-full-rlcd/test_metrics.json
```

RLCD с one-hot меткой по большинству ассесоров на тех же данных:

```bash
torchrun --standalone --nproc_per_node=2 train_judge.py \
  --config configs/full_h100_rlcd_hard.toml
cat checkpoints/pollux-full-rlcd-hard/test_metrics.json
```

### Qwen3.5-4B на двух H100

Конфиг `configs/full_h100_4b_rlcd_hard.toml` запускает CE и RLCD с one-hot
оценкой большинства ассесоров. Конфиг `configs/full_h100_4b.toml` оставлен
для CE с распределением голосов. Оба сохраняют ту же схему train/test и используют отдельную
подготовку в `checkpoints/pollux-full-4b-data`: токены строятся шаблоном и
токенизатором 4B, а сохранённый набор проверяется по имени модели. Исходные
Parquet-файлы в `checkpoints/pollux-source` повторно скачивать не нужно.
В конфиге batch на GPU равен 1, накопление градиентов — 16, общий эффективный
batch на двух GPU — 32. Оценка test и сохранение чекпоинтов выполняются после
каждой эпохи.

```bash
python train_judge.py --config configs/full_h100_4b_rlcd_hard.toml --prepare-only
torchrun --standalone --nproc_per_node=2 train_judge.py \
  --config configs/full_h100_4b_rlcd_hard.toml
cat checkpoints/pollux-full-4b-rlcd-hard/test_metrics.json
```

Для пробного запуска на 100 строках используйте:

```bash
python train_judge.py --config configs/sample.toml --prepare-only
python train_judge.py --config configs/sample.toml
python train_judge.py --config configs/sample_rlcd.toml
```

Чтобы сохранить отдельную случайную строку POLLUX и посмотреть точный вход
Qwen после токенизации и обратного декодирования, выполните:

```bash
python inspect_random_example.py --fetch
python inspect_random_example.py
```

Результаты записываются в `examples/pollux_random_example.json`,
`examples/pollux_decoded_input.txt` и `examples/pollux_inspection.json`.

Для инференса одного примера через Transformers:

```bash
python infer_one.py \
  --checkpoint checkpoints/pollux-full-ce/final \
  --example examples/pollux_random_example.json
```

`--example` принимает сохранённую строку POLLUX либо JSON-объект с полями
`instruction`, `answer`, `criteria_name`, `criteria_description`, `rubrics` и,
если есть, `reference_answer`. Скрипт использует тот же `make_messages` и chat
template; `annotations` для инференса не нужны. Вывод содержит выбранную оценку
и вероятности каждого варианта.
В `examples/pollux_test_score_2.json` лежит короткий пример из test:
тип задачи «Написать художественный текст», три оценки ассесоров равны 2.
Чтобы проверить его, передайте этот путь в `--example`.

Источники реализации: [шаблон и токенизатор Qwen](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/tokenizer_config.json),
[классификационный тренер Halo](https://github.com/whitecircle/halo/blob/main/src/trainers/reward/classification.py),
[модельная основа Qwen](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modular_qwen3_5.py).
