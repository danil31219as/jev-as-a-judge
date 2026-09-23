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

После обучения `test_metrics.json` содержит **MAE** и **RMSE** между ожидаемыми
числовыми оценками модели и ассесоров, а также **macro-F1** между наиболее
вероятными оценками. При ничьей по голосам для F1 выбирается меньшая оценка.
Тест используется только для итоговой оценки, не для обновления весов.

Опция `--laya-rl` добавляет к soft-target CE алгоритм RLCD из
[Laya](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb): четыре
гауссовых возмущения логитов на пример, награда из log score, spherical score
и ordinal RPS, групповое центрирование преимущества и policy gradient.
Стандартное отклонение шума снижается от 0.4 до 0.1 по эпохам. Для RPS
варианты упорядочиваются по числовой оценке, независимо от порядка вызовов.

Авторы POLLUX [не рекомендуют обучаться на нём](https://huggingface.co/datasets/ai-forever/POLLUX/blob/main/README.md#out-of-scope-use),
чтобы сохранить независимость бенчмарка. Этот запуск выполняет ваш запрос как
учебный эксперимент. Результат на POLLUX после такого обучения нельзя считать
независимой оценкой качества.

## Запуск

Нужна среда [Halo](https://github.com/whitecircle/halo) 1.0 с совместимыми
PyTorch, Transformers, Datasets и GPU с поддержкой BF16. Скрипт использует
`ClassificationTrainer` из Halo напрямую: стандартная команда
`halo launch classification` использует представление конца последовательности
и не извлекает логиты на `</tool_call>`. Положите корень клона Halo в
`PYTHONPATH`, затем запустите:

```bash
export PYTHONPATH=/path/to/halo:$PYTHONPATH
python train_judge.py --prepare-only
python train_judge.py --output-dir checkpoints/pollux-tool-call-judge
python train_judge.py --laya-rl --output-dir checkpoints/pollux-tool-call-judge-rlcd
```

Для логирования в ClearML установите и настройте SDK, затем добавьте флаг:

```bash
pip install clearml
clearml-init
python train_judge.py --clearml --clearml-project jev-as-a-judge
```

Каждый запуск создаёт отдельную задачу ClearML. В неё записываются параметры
запуска, loss и learning rate по шагам, итоговые тестовые MAE/RMSE/macro-F1,
размеры выборок, а также файлы `selection.json` и `test_metrics.json`.
Имя задачи можно задать через `--clearml-task-name`. По умолчанию большие
чекпоинты остаются локально; `CLEARML_LOG_MODEL=TRUE` включает их загрузку
через штатную интеграцию Transformers.

Первый запуск проверяет разбиение 80/20, chat template, сохранность всех
закрывающих маркеров и записывает `selection.json`, не загружая веса Qwen.
Второй обучает на 80 примерах, считает метрики на 20 тестовых и сохраняет
модель с токенизатором в `checkpoints/pollux-tool-call-judge/final`.
Число тестовых строк меняется через `--test-samples` (не менее восьми), а
`--samples` задаёт общее число. Проверка данных без GPU:

```bash
python -m unittest discover -s tests -v
```

### Полный POLLUX на двух H100

Команды ниже рассчитаны на Linux-сервер с Docker, NVIDIA Container Toolkit и
двумя H100. Образ Halo для Hopper уже содержит совместимые PyTorch,
Transformers и исходники Halo в `/workspace`; репозиторий судьи монтируется
отдельно в `/judge`.

На сервере:

```bash
git clone https://github.com/danil31219as/jev-as-a-judge.git
cd jev-as-a-judge
nvidia-smi -L
mkdir -p checkpoints/hf-cache checkpoints/tmp
docker pull public.ecr.aws/whitecircle/halo:hopper-1.0.0
docker run --rm -it --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=128g \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e HF_HOME=/judge/checkpoints/hf-cache \
  -e HF_DATASETS_CACHE=/judge/checkpoints/hf-cache/datasets \
  -e TMPDIR=/judge/checkpoints/tmp \
  -e PYTHONPATH=/workspace:/judge \
  -v "$PWD:/judge" -w /judge \
  public.ecr.aws/whitecircle/halo:hopper-1.0.0 bash
```

В контейнере:

```bash
python -m unittest discover -s tests -v
python train_judge.py --full-dataset --prepare-only \
  --output-dir checkpoints/pollux-full-data

# Если нужен ClearML, настройте его в этой же сессии контейнера:
python -m pip install clearml
clearml-init

torchrun --standalone --nproc_per_node=2 train_judge.py \
  --full-dataset --prepared-data-dir checkpoints/pollux-full-data \
  --output-dir checkpoints/pollux-full-ce \
  --per-device-train-batch-size 2 --per-device-eval-batch-size 2 \
  --gradient-accumulation-steps 8 --dataloader-num-workers 4 \
  --epochs 3 --learning-rate 1e-5 \
  --clearml --clearml-project jev-as-a-judge

cat checkpoints/pollux-full-ce/test_metrics.json
```

Чтобы обучить отдельную модель с RLCD на том же разбиении, запустите в
контейнере ещё одну команду:

```bash
torchrun --standalone --nproc_per_node=2 train_judge.py \
  --full-dataset --prepared-data-dir checkpoints/pollux-full-data \
  --output-dir checkpoints/pollux-full-rlcd --laya-rl \
  --per-device-train-batch-size 2 --per-device-eval-batch-size 2 \
  --gradient-accumulation-steps 8 --dataloader-num-workers 4 \
  --epochs 3 --learning-rate 1e-5 \
  --clearml --clearml-project jev-as-a-judge
```

Если ClearML не настроен, пропустите две команды его установки и настройки и
уберите `--clearml --clearml-project jev-as-a-judge` из запусков обучения.
Эффективный размер батча при указанных параметрах: 2 GPU × 2 примера × 8
шагов накопления = 32. Если памяти не хватит, уменьшите
`--per-device-train-batch-size` до 1 и увеличьте
`--gradient-accumulation-steps` до 16.

Чтобы сохранить отдельную случайную строку POLLUX и посмотреть точный вход
Qwen после токенизации и обратного декодирования, выполните:

```bash
python inspect_random_example.py --fetch
python inspect_random_example.py
```

Результаты записываются в `examples/pollux_random_example.json`,
`examples/pollux_decoded_input.txt` и `examples/pollux_inspection.json`.

После обучения загружайте `ToolCallJudge.from_pretrained(...)` из
`judge_model.py`, применяйте **тот же** `make_messages` и chat template, а
вероятности переводите из позиций логитов обратно в числовые оценки через
`option_values`. В режиме 100 примеров эта связь находится в строках
`selection.json`, а в полном режиме — в каждой записи `prepared/*.jsonl`.

Источники реализации: [шаблон и токенизатор Qwen](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/tokenizer_config.json),
[классификационный тренер Halo](https://github.com/whitecircle/halo/blob/main/src/trainers/reward/classification.py),
[модельная основа Qwen](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modular_qwen3_5.py).
