# mediawiki-autocategorization

Проект по автокатегоризации статей из [MediaWiki](0x1tv.ru) с использованием вероятностных тематических моделей

### Как запускать?

```bash
bash install-env39.sh
```

- training (in separate windows)
```
ollama run gemma2
python train.py
```
- inference
```
python inference.py
```

### Пример результатов
![Двухуровневая тематическая модель](mediawiki-clusterization.png "Двухуровневая тематическая модель")
