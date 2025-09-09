import json
from openai import OpenAI
from typing import List, Dict
import ast
import re
import numpy as np

# ------------------------------------------------------------
# 0. Configure NVIDIA NIM client
# ------------------------------------------------------------
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="NVIDIA API KEY: https://build.nvidia.com/models"
)

# ------------------------------------------------------------
# 1. Brique 1 – Base de connaissances (static KB)
# Rules (expandable from GitHub CAST / cnumr best practices, all turned to Python for consistency reasons)
# ------------------------------------------------------------
knowledge_base = [
  {
    "id": "rule_001",
    "langage": "Python",
    "contexte": "Algorithmes et boucles",
    "anti_pattern": "Boucles Python inefficaces pour transformer des données",
    "advice": "Utiliser des compréhensions de liste ou des générateurs au lieu de for-loops manuels lorsque c'est possible.",
    "exemple_avant": "res = []\nfor x in data:\n    res.append(x * x)",
    "exemple_apres": "res = [x * x for x in data]",
    "gain_cpu": "-30%",
    "gain_ram": "-10%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.10g CO₂ économisés pour 1M opérations"
  },
  {
    "id": "rule_002",
    "langage": "Python",
    "contexte": "Structures de données",
    "anti_pattern": "Recherche linéaire dans une liste à l'intérieur d'une boucle",
    "advice": "Utiliser un set ou un dictionnaire pour des tests d'appartenance en O(1) au lieu de scans en O(n).",
    "exemple_avant": "hits = 0\nfor x in items:\n    if x in big_list:\n        hits += 1",
    "exemple_apres": "big_set = set(big_list)\nhits = sum(1 for x in items if x in big_set)",
    "gain_cpu": "-60%",
    "gain_ram": "-20%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.22g CO₂ économisés pour 1M opérations"
  },
  {
    "id": "rule_003",
    "langage": "Python",
    "contexte": "Chaînes de caractères",
    "anti_pattern": "Concaténation de chaînes dans une boucle",
    "advice": "Construire une liste puis utiliser ''.join(list) pour assembler la chaîne.",
    "exemple_avant": "s = \"\"\nfor part in parts:\n    s += part",
    "exemple_apres": "s = \"\".join(parts)",
    "gain_cpu": "-70%",
    "gain_ram": "-20%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.24g CO₂ économisés pour 1M opérations"
  },
  {
    "id": "rule_004",
    "langage": "Python",
    "contexte": "Algorithmes",
    "anti_pattern": "Tri répété dans une boucle",
    "advice": "Trier une seule fois ou utiliser heapq pour des extractions partielles.",
    "exemple_avant": "for x in batches:\n    data.sort()\n    use(data[0])",
    "exemple_apres": "import heapq\nheap = data[:]\nheapq.heapify(heap)\nfor x in batches:\n    use(heapq.heappop(heap))",
    "gain_cpu": "-60%",
    "gain_ram": "-20%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.22g CO₂ économisés pour 1M opérations"
  },
  {
    "id": "rule_005",
    "langage": "Python",
    "contexte": "Fichiers et I/O",
    "anti_pattern": "Ouverture/fermeture de fichiers manuelle sans gestion de contexte",
    "advice": "Utiliser des gestionnaires de contexte (with open(...)) pour une gestion sûre et efficace.",
    "exemple_avant": "f = open(path, \"w\")\nfor row in rows:\n    f.write(row)\nf.close()",
    "exemple_apres": "with open(path, \"w\") as f:\n    for row in rows:\n        f.write(row)",
    "gain_cpu": "-10%",
    "gain_ram": "-10%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.05g CO₂ économisés pour 1M écritures"
  },
  {
    "id": "rule_006",
    "langage": "Python",
    "contexte": "Regex et parsing",
    "anti_pattern": "Compilation répétée d'expressions régulières",
    "advice": "Compiler une fois avec re.compile() et réutiliser le motif.",
    "exemple_avant": "import re\nfor line in lines:\n    if re.search(r\"\\d+\", line):\n        process(line)",
    "exemple_apres": "import re\npat = re.compile(r\"\\d+\")\nfor line in lines:\n    if pat.search(line):\n        process(line)",
    "gain_cpu": "-40%",
    "gain_ram": "-10%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.12g CO₂ économisés pour 1M lignes"
  },
  {
    "id": "rule_007",
    "langage": "Python",
    "contexte": "Qualité de code",
    "anti_pattern": "Variables inutilisées",
    "advice": "Supprimer les variables inutiles pour réduire l'empreinte mémoire et clarifier le code.",
    "exemple_avant": "tmp = heavy_calc()\n_ = 0\nreturn tmp",
    "exemple_apres": "return heavy_calc()",
    "gain_cpu": "-5%",
    "gain_ram": "-5%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.02g CO₂ économisés pour 1M appels"
  },
  {
    "id": "rule_008",
    "langage": "Python",
    "contexte": "Sérialisation JSON",
    "anti_pattern": "Parsing JSON lent avec le module standard pour de gros fichiers",
    "advice": "Utiliser orjson/ujson pour charger rapidement de gros JSON.",
    "exemple_avant": "import json\nobj = [json.loads(s) for s in big_strings]",
    "exemple_apres": "import orjson\nobj = [orjson.loads(s) for s in big_strings]",
    "gain_cpu": "-50%",
    "gain_ram": "-20%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.20g CO₂ économisés pour 1M objets"
  },
  {
    "id": "rule_009",
    "langage": "Python",
    "contexte": "Bases de données",
    "anti_pattern": "Requêtes SQL non groupées dans une boucle",
    "advice": "Utiliser executemany/bulk insert pour grouper les opérations.",
    "exemple_avant": "for row in rows:\n    cur.execute(\"INSERT INTO t VALUES (%s,%s)\", row)",
    "exemple_apres": "cur.executemany(\"INSERT INTO t VALUES (%s,%s)\", rows)",
    "gain_cpu": "-40%",
    "gain_ram": "-25%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.18g CO₂ économisés pour 1M lignes"
  },
  {
    "id": "rule_010",
    "langage": "Python",
    "contexte": "Qualité de code",
    "anti_pattern": "Imports non utilisés",
    "advice": "Supprimer les imports inutiles pour réduire la mémoire et le temps de démarrage.",
    "exemple_avant": "import numpy as np\nimport pandas as pd\nfrom math import sqrt\nprint(\"ok\")",
    "exemple_apres": "print(\"ok\")",
    "gain_cpu": "-5%",
    "gain_ram": "-10%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.03g CO₂ économisés par exécution"
  },
  {
    "id": "rule_011",
    "langage": "Python",
    "contexte": "Journalisation",
    "anti_pattern": "Concaténation de chaînes dans les logs",
    "advice": "Utiliser le formattage paresseux avec %s ou les paramètres du logger.",
    "exemple_avant": "logger.debug(\"value=\" + str(expensive()))",
    "exemple_apres": "logger.debug(\"value=%s\", expensive())",
    "gain_cpu": "-25%",
    "gain_ram": "-5%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.10g CO₂ économisés pour 1M logs"
  },
  {
    "id": "rule_012",
    "langage": "Python",
    "contexte": "Optimisation d'appels",
    "anti_pattern": "Appels répétés de fonctions coûteuses avec mêmes entrées",
    "advice": "Mettre en cache avec functools.lru_cache.",
    "exemple_avant": "def fib(n):\n    return n if n<2 else fib(n-1)+fib(n-2)",
    "exemple_apres": "from functools import lru_cache\n@lru_cache(None)\ndef fib(n):\n    return n if n<2 else fib(n-1)+fib(n-2)",
    "gain_cpu": "-80%",
    "gain_ram": "-20%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.30g CO₂ économisés pour 1M appels"
  },
  {
    "id": "rule_013",
    "langage": "Python",
    "contexte": "Gestion de mémoire",
    "anti_pattern": "Sur-utilisation de deepcopy()",
    "advice": "Utiliser des copies superficielles lorsque c'est sûr.",
    "exemple_avant": "from copy import deepcopy\nb = deepcopy(a)",
    "exemple_apres": "from copy import copy\nb = copy(a)  # ou utiliser une vue",
    "gain_cpu": "-30%",
    "gain_ram": "-45%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.18g CO₂ économisés pour 1M objets"
  },
  {
    "id": "rule_014",
    "langage": "Python",
    "contexte": "Algorithmes et boucles",
    "anti_pattern": "Appel répété à len(list) dans une boucle",
    "advice": "Mémoriser la longueur avant la boucle.",
    "exemple_avant": "for i in range(len(lst)):\n    do(lst[i])",
    "exemple_apres": "n = len(lst)\nfor i in range(n):\n    do(lst[i])",
    "gain_cpu": "-15%",
    "gain_ram": "-0%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.06g CO₂ économisés pour 1M itérations"
  },
  {
    "id": "rule_015",
    "langage": "Python",
    "contexte": "CSV et parsing",
    "anti_pattern": "Lecture CSV manuelle avec split() sans structure",
    "advice": "Utiliser csv.DictReader ou pandas pour un parsing structuré.",
    "exemple_avant": "for line in open(path):\n    cols = line.strip().split(',')\n    use(cols)",
    "exemple_apres": "import csv\nwith open(path) as f:\n    for row in csv.DictReader(f):\n        use(row)",
    "gain_cpu": "-25%",
    "gain_ram": "-25%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.12g CO₂ économisés pour 1M lignes"
  },
  {
    "id": "rule_016",
    "langage": "Python",
    "contexte": "Itérateurs",
    "anti_pattern": "Découpage (slicing) répété de grandes listes",
    "advice": "Utiliser des itérateurs ou itertools.islice pour éviter des copies.",
    "exemple_avant": "for i in range(0, len(lst), 1000):\n    chunk = lst[i:i+1000]\n    process(chunk)",
    "exemple_apres": "from itertools import islice\nit = iter(lst)\nwhile True:\n    chunk = list(islice(it, 1000))\n    if not chunk: break\n    process(chunk)",
    "gain_cpu": "-25%",
    "gain_ram": "-40%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.16g CO₂ économisés pour 1M éléments"
  },
  {
    "id": "rule_017",
    "langage": "Python",
    "contexte": "Gestion d'exceptions",
    "anti_pattern": "Utiliser les exceptions pour le contrôle de flux",
    "advice": "Tester les conditions (in, get, try léger) plutôt que lever/capturer souvent.",
    "exemple_avant": "for k in keys:\n    try:\n        total += d[k]\n    except KeyError:\n        pass",
    "exemple_apres": "for k in keys:\n    total += d.get(k, 0)",
    "gain_cpu": "-50%",
    "gain_ram": "-5%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.18g CO₂ économisés pour 1M itérations"
  },
  {
    "id": "rule_018",
    "langage": "Python",
    "contexte": "Concurrence et ressources",
    "anti_pattern": "Nettoyage manuel des ressources (pools, sockets) sans contexte",
    "advice": "Utiliser des gestionnaires de contexte pour garantir le cleanup.",
    "exemple_avant": "from multiprocessing import Pool\npool = Pool()\nres = pool.map(f, items)\npool.close(); pool.join()",
    "exemple_apres": "from multiprocessing import Pool\nwith Pool() as pool:\n    res = pool.map(f, items)",
    "gain_cpu": "-10%",
    "gain_ram": "-20%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.08g CO₂ économisés pour 1M tâches"
  },
  {
    "id": "rule_019",
    "langage": "Python",
    "contexte": "NumPy",
    "anti_pattern": "Boucles Python sur des tableaux NumPy",
    "advice": "Vectoriser les opérations NumPy.",
    "exemple_avant": "out = np.empty_like(a)\nfor i in range(len(a)):\n    out[i] = a[i] * 2",
    "exemple_apres": "out = a * 2",
    "gain_cpu": "-80%",
    "gain_ram": "-20%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.30g CO₂ économisés pour 1M éléments"
  },
  {
    "id": "rule_020",
    "langage": "Python",
    "contexte": "pandas",
    "anti_pattern": "DataFrame.apply() ligne par ligne",
    "advice": "Utiliser des opérations vectorisées pandas.",
    "exemple_avant": "df[\"y\"] = df[\"x\"].apply(lambda v: v*2)",
    "exemple_apres": "df[\"y\"] = df[\"x\"] * 2",
    "gain_cpu": "-70%",
    "gain_ram": "-20%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.26g CO₂ économisés pour 1M lignes"
  },
  {
    "id": "rule_021",
    "langage": "Python",
    "contexte": "pandas",
    "anti_pattern": "Itération ligne par ligne sur un DataFrame",
    "advice": "Préférer les opérations vectorisées ou itertuples() si nécessaire.",
    "exemple_avant": "for _, row in df.iterrows():\n    total += row[\"x\"]",
    "exemple_apres": "total = df[\"x\"].sum()  # ou for r in df.itertuples(): total += r.x",
    "gain_cpu": "-65%",
    "gain_ram": "-15%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.24g CO₂ économisés pour 1M lignes"
  },
  {
    "id": "rule_022",
    "langage": "Python",
    "contexte": "Structures de données",
    "anti_pattern": "Boucles imbriquées avec tests d'appartenance sur des listes",
    "advice": "Remplacer les listes par des sets/dicts pour les membership tests.",
    "exemple_avant": "count = 0\nfor x in A:\n    if x in B:\n        count += 1",
    "exemple_apres": "B_set = set(B)\ncount = sum(1 for x in A if x in B_set)",
    "gain_cpu": "-70%",
    "gain_ram": "-20%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.26g CO₂ économisés pour 1M comparaisons"
  },
  {
    "id": "rule_023",
    "langage": "Python",
    "contexte": "Chaînes de caractères",
    "anti_pattern": "Concaténation lente pour formatage",
    "advice": "Utiliser des f-strings ou str.format pour lisibilité et performance.",
    "exemple_avant": "msg = \"Hello \" + name + \", id=\" + str(i)",
    "exemple_apres": "msg = f\"Hello {name}, id={i}\"",
    "gain_cpu": "-20%",
    "gain_ram": "-5%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.08g CO₂ économisés pour 1M chaînes"
  },
  {
    "id": "rule_024",
    "langage": "Python",
    "contexte": "Algorithmes et boucles",
    "anti_pattern": "Croissance inefficace d'une liste par append dans de très grandes boucles",
    "advice": "Utiliser des compréhensions ou pré-allouer si possible.",
    "exemple_avant": "res = []\nfor x in data:\n    res.append(f(x))",
    "exemple_apres": "res = [f(x) for x in data]",
    "gain_cpu": "-25%",
    "gain_ram": "-20%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.12g CO₂ économisés pour 1M éléments"
  },
  {
    "id": "rule_025",
    "langage": "Python",
    "contexte": "Sérialisation",
    "anti_pattern": "Utilisation inefficace de pickle pour de gros objets",
    "advice": "Utiliser joblib/orjson/rapids pour objets volumineux ou NumPy.",
    "exemple_avant": "import pickle\npickle.dump(obj, open(\"f.pkl\",\"wb\"))",
    "exemple_apres": "from joblib import dump\ndump(obj, \"f.joblib\")",
    "gain_cpu": "-30%",
    "gain_ram": "-25%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.14g CO₂ économisés pour 1M enregistrements"
  },
  {
    "id": "rule_026",
    "langage": "Python",
    "contexte": "Qualité de code",
    "anti_pattern": "Variables globales persistantes",
    "advice": "Éviter les globales, passer les données en paramètres ou via objets.",
    "exemple_avant": "cache = []\ndef add(x):\n    global cache\n    cache.append(x)",
    "exemple_apres": "def add(cache, x):\n    cache.append(x)",
    "gain_cpu": "-5%",
    "gain_ram": "-20%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.06g CO₂ économisés pour 1M appels"
  },
  {
    "id": "rule_027",
    "langage": "Python",
    "contexte": "Chaînes de caractères",
    "anti_pattern": "Découpage de chaîne manuel avec tranches/regex",
    "advice": "Utiliser str.split quand c'est possible.",
    "exemple_avant": "area = text[:3]; number = text[4:]",
    "exemple_apres": "area, number = text.split('-')",
    "gain_cpu": "-15%",
    "gain_ram": "-5%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.06g CO₂ économisés pour 1M découpages"
  },
  {
    "id": "rule_028",
    "langage": "Python",
    "contexte": "pandas",
    "anti_pattern": "Concaténation répétée de DataFrames dans une boucle",
    "advice": "Accumuler dans une liste puis utiliser pd.concat une seule fois.",
    "exemple_avant": "df = pd.DataFrame()\nfor part in parts:\n    df = pd.concat([df, part])",
    "exemple_apres": "df = pd.concat(parts, ignore_index=True)",
    "gain_cpu": "-65%",
    "gain_ram": "-60%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.28g CO₂ économisés pour 1M lignes"
  },
  {
    "id": "rule_029",
    "langage": "Python",
    "contexte": "CSV et I/O",
    "anti_pattern": "Écriture CSV non tamponnée, ligne par ligne",
    "advice": "Utiliser un buffer ou pandas.to_csv avec chunksize.",
    "exemple_avant": "with open(path, 'w', newline='') as f:\n    w = csv.writer(f)\n    for r in rows:\n        w.writerow(r)",
    "exemple_apres": "with open(path, 'w', newline='') as f:\n    w = csv.writer(f)\n    w.writerows(rows)  # ou pandas.to_csv(chunksize=...)\n",
    "gain_cpu": "-20%",
    "gain_ram": "-15%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.10g CO₂ économisés pour 1M lignes"
  },
  {
    "id": "rule_030",
    "langage": "Python",
    "contexte": "pandas et mémoire",
    "anti_pattern": "Copies de DataFrame non nécessaires via .copy()",
    "advice": "Éviter .copy() inutile, préférer les vues ou opérations en place lorsque c'est sûr.",
    "exemple_avant": "df2 = df.copy()\ndf2[\"x\"] *= 2",
    "exemple_apres": "df[\"x\"] *= 2  # si l'aliasing est accepté",
    "gain_cpu": "-20%",
    "gain_ram": "-50%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.20g CO₂ économisés pour 1M lignes"
   },

    {
    "id": "rule_031",
    "langage": "Python",
    "contexte": "Algorithmes et listes",
    "anti_pattern": "Aplatissement manuel de listes imbriquées avec des boucles",
    "advice": "Utiliser itertools.chain.from_iterable pour aplatir efficacement.",
    "exemple_avant": "flat = []\nfor sub in lists:\n    for x in sub:\n        flat.append(x)",
    "exemple_apres": "from itertools import chain\nflat = list(chain.from_iterable(lists))",
    "gain_cpu": "-30%",
    "gain_ram": "-10%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.12g CO₂ économisés pour 1M éléments"
  },
  {
    "id": "rule_032",
    "langage": "Python",
    "contexte": "Algèbre linéaire",
    "anti_pattern": "Multiplication de matrices avec des boucles Python",
    "advice": "Utiliser NumPy (BLAS optimisé) ou l’opérateur @.",
    "exemple_avant": "C = [[sum(a*b for a,b in zip(row,col)) for col in zip(*B)] for row in A]",
    "exemple_apres": "import numpy as np\nC = A @ B  # A et B sont des tableaux NumPy",
    "gain_cpu": "-85%",
    "gain_ram": "-20%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.34g CO₂ économisés pour 1M produits scalaires"
  },
  {
    "id": "rule_033",
    "langage": "Python",
    "contexte": "Fichiers et I/O",
    "anti_pattern": "readlines() sur de gros fichiers avant itération",
    "advice": "Itérer directement sur l’objet fichier (streaming).",
    "exemple_avant": "lines = open(path).readlines()\nfor line in lines:\n    process(line)",
    "exemple_apres": "with open(path) as f:\n    for line in f:\n        process(line)",
    "gain_cpu": "-10%",
    "gain_ram": "-40%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.10g CO₂ économisés pour 1M lignes"
  },
  {
    "id": "rule_034",
    "langage": "Python",
    "contexte": "Structures de données",
    "anti_pattern": "Initialisation manuelle d’un compteur/dict avec tests",
    "advice": "Utiliser collections.Counter ou defaultdict.",
    "exemple_avant": "counts = {}\nfor k in keys:\n    if k in counts:\n        counts[k] += 1\n    else:\n        counts[k] = 1",
    "exemple_apres": "from collections import Counter\ncounts = Counter(keys)",
    "gain_cpu": "-40%",
    "gain_ram": "-10%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.16g CO₂ économisés pour 1M éléments"
  },
  {
    "id": "rule_035",
    "langage": "Python",
    "contexte": "Optimisation de boucles",
    "anti_pattern": "Conversions de type répétées dans une boucle",
    "advice": "Convertir une fois en amont ou vectoriser.",
    "exemple_avant": "total = 0\nfor x in values:\n    total += int(x)",
    "exemple_apres": "values = list(map(int, values))\ntotal = sum(values)",
    "gain_cpu": "-20%",
    "gain_ram": "-5%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.08g CO₂ économisés pour 1M conversions"
  },
  {
    "id": "rule_036",
    "langage": "Python",
    "contexte": "Dates et parsing",
    "anti_pattern": "datetime.strptime dans une boucle sur de gros volumes",
    "advice": "Utiliser pandas.to_datetime ou ciso8601 pour vectoriser/accélérer.",
    "exemple_avant": "from datetime import datetime\nfor s in dates:\n    d = datetime.strptime(s, '%Y-%m-%d')\n    use(d)",
    "exemple_apres": "import pandas as pd\nfor d in pd.to_datetime(dates, format='%Y-%m-%d'):\n    use(d)",
    "gain_cpu": "-60%",
    "gain_ram": "-10%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.24g CO₂ économisés pour 1M dates"
  },
  {
    "id": "rule_037",
    "langage": "Python",
    "contexte": "Concurrence (threads)",
    "anti_pattern": "Gestion manuelle d’un pool de threads",
    "advice": "Utiliser ThreadPoolExecutor pour mapper les tâches.",
    "exemple_avant": "from threading import Thread\nthreads = []\nfor x in tasks:\n    t = Thread(target=work, args=(x,))\n    t.start(); threads.append(t)\nfor t in threads:\n    t.join()",
    "exemple_apres": "from concurrent.futures import ThreadPoolExecutor\nwith ThreadPoolExecutor() as ex:\n    list(ex.map(work, tasks))",
    "gain_cpu": "-25%",
    "gain_ram": "-10%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.10g CO₂ économisés pour 1M tâches"
  },
  {
    "id": "rule_038",
    "langage": "Python",
    "contexte": "Concurrence (processus)",
    "anti_pattern": "Création de très nombreux processus non réutilisés",
    "advice": "Réutiliser un Pool de processus.",
    "exemple_avant": "from multiprocessing import Process\nfor chunk in chunks:\n    p = Process(target=work, args=(chunk,))\n    p.start(); p.join()",
    "exemple_apres": "from multiprocessing import Pool\nwith Pool() as pool:\n    pool.map(work, chunks)",
    "gain_cpu": "-35%",
    "gain_ram": "-30%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.14g CO₂ économisés pour 1M tâches"
  },
  {
    "id": "rule_039",
    "langage": "Python",
    "contexte": "Qualité de code",
    "anti_pattern": "Embranchements et tests redondants imbriqués",
    "advice": "Aplatir et simplifier les conditions.",
    "exemple_avant": "if a:\n    if b:\n        if not c:\n            do()",
    "exemple_apres": "if a and b and not c:\n    do()",
    "gain_cpu": "-10%",
    "gain_ram": "-0%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.04g CO₂ économisés pour 1M tests"
  },
  {
    "id": "rule_040",
    "langage": "Python",
    "contexte": "Aléatoire et vecteur",
    "anti_pattern": "Génération de nombres aléatoires un par un avec random.random",
    "advice": "Utiliser numpy.random pour une génération vectorisée.",
    "exemple_avant": "import random\nvals = [random.random() for _ in range(n)]",
    "exemple_apres": "import numpy as np\nvals = np.random.random(n)",
    "gain_cpu": "-70%",
    "gain_ram": "-10%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.28g CO₂ économisés pour 1M tirages"
  },
  {
    "id": "rule_041",
    "langage": "Python",
    "contexte": "XML (streaming)",
    "anti_pattern": "Chargement complet de l’arbre XML avec parse()",
    "advice": "Utiliser iterparse et nettoyer les nœuds au fil de l’eau.",
    "exemple_avant": "import xml.etree.ElementTree as ET\nroot = ET.parse(file).getroot()\nfor elem in root.iter('item'):\n    process(elem)",
    "exemple_apres": "import xml.etree.ElementTree as ET\nfor event, elem in ET.iterparse(file, events=('end',)):\n    if elem.tag == 'item':\n        process(elem)\n        elem.clear()",
    "gain_cpu": "-30%",
    "gain_ram": "-50%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.18g CO₂ économisés pour 1M éléments"
  },
  {
    "id": "rule_042",
    "langage": "Python",
    "contexte": "Mémorisation des résultats",
    "anti_pattern": "Recalcul d’un résultat coûteux pour les mêmes entrées",
    "advice": "Utiliser functools.lru_cache pour mettre en cache.",
    "exemple_avant": "def f(x):\n    return expensive(x)\nres = [f(x) for x in xs]",
    "exemple_apres": "from functools import lru_cache\n@lru_cache(None)\ndef f(x):\n    return expensive(x)\nres = [f(x) for x in xs]",
    "gain_cpu": "-60%",
    "gain_ram": "-20%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.24g CO₂ économisés pour 1M appels"
  },
  {
    "id": "rule_043",
    "langage": "Python",
    "contexte": "Chaînes de caractères",
    "anti_pattern": "Recherche simple via regex",
    "advice": "Utiliser str.find/startswith/endswith pour les cas simples.",
    "exemple_avant": "import re\nif re.search(r'^abc', s):\n    ...",
    "exemple_apres": "if s.startswith('abc'):\n    ...",
    "gain_cpu": "-30%",
    "gain_ram": "-5%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.12g CO₂ économisés pour 1M tests"
  },
  {
    "id": "rule_044",
    "langage": "Python",
    "contexte": "CSV volumineux / pandas",
    "anti_pattern": "Lecture d’un gros CSV sans chunks (pic mémoire)",
    "advice": "Lire par morceaux avec chunksize et traiter incrémentalement.",
    "exemple_avant": "import pandas as pd\ndf = pd.read_csv(path)",
    "exemple_apres": "import pandas as pd\nfor chunk in pd.read_csv(path, chunksize=100_000):\n    process(chunk)",
    "gain_cpu": "-20%",
    "gain_ram": "-60%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.16g CO₂ économisés pour 1M lignes"
  },
  {
    "id": "rule_045",
    "langage": "Python",
    "contexte": "Calcul numérique",
    "anti_pattern": "Boucles Python pour des calculs mathématiques élément par élément",
    "advice": "Vectoriser avec NumPy.",
    "exemple_avant": "out = []\nfor x in a:\n    out.append(math.sin(x) + x*x)",
    "exemple_apres": "import numpy as np\nout = np.sin(a) + a*a",
    "gain_cpu": "-80%",
    "gain_ram": "-20%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.30g CO₂ économisés pour 1M éléments"
  },
  {
    "id": "rule_046",
    "langage": "Python",
    "contexte": "Dictionnaires",
    "anti_pattern": "Fusion de dictionnaires par boucles manuelles",
    "advice": "Utiliser l’unpacking {**d1, **d2} ou ChainMap.",
    "exemple_avant": "merged = {}\nfor k, v in d1.items():\n    merged[k] = v\nfor k, v in d2.items():\n    merged[k] = v",
    "exemple_apres": "merged = {**d1, **d2}  # ou: from collections import ChainMap; merged = dict(ChainMap(d2, d1))",
    "gain_cpu": "-15%",
    "gain_ram": "-5%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.06g CO₂ économisés pour 1M clés"
  },
  {
    "id": "rule_047",
    "langage": "Python",
    "contexte": "Collections",
    "anti_pattern": "Comptage manuel d’items dans une boucle",
    "advice": "Utiliser collections.Counter.",
    "exemple_avant": "counts = {}\nfor x in data:\n    counts[x] = counts.get(x, 0) + 1",
    "exemple_apres": "from collections import Counter\ncounts = Counter(data)",
    "gain_cpu": "-30%",
    "gain_ram": "-10%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.12g CO₂ économisés pour 1M éléments"
  },
  {
    "id": "rule_048",
    "langage": "Python",
    "contexte": "Aléatoire",
    "anti_pattern": "Reseeding à chaque itération",
    "advice": "Initialiser la graine une seule fois en amont.",
    "exemple_avant": "import random\nfor _ in range(n):\n    random.seed(42)\n    do()",
    "exemple_apres": "import random\nrandom.seed(42)\nfor _ in range(n):\n    do()",
    "gain_cpu": "-15%",
    "gain_ram": "-0%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.04g CO₂ économisés pour 1M itérations"
  },
  {
    "id": "rule_049",
    "langage": "Python",
    "contexte": "Qualité de code et mémoire",
    "anti_pattern": "Arguments par défaut mutables (fuite mémoire/comportement inattendu)",
    "advice": "Utiliser None et initialiser dans le corps.",
    "exemple_avant": "def add(item, cache=[]):\n    cache.append(item)\n    return cache",
    "exemple_apres": "def add(item, cache=None):\n    if cache is None:\n        cache = []\n    cache.append(item)\n    return cache",
    "gain_cpu": "-0%",
    "gain_ram": "-10%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "Prévention d’anomalies; impact variable"
  },
  {
    "id": "rule_050",
    "langage": "Python",
    "contexte": "Tri et clés",
    "anti_pattern": "Utiliser une comparaison (cmp) coûteuse qui recalcule la clé",
    "advice": "Pré-calculer des clés et utiliser le paramètre key du tri.",
    "exemple_avant": "from functools import cmp_to_key\nsorted(recs, key=cmp_to_key(lambda a,b: expensive(a)-expensive(b)))",
    "exemple_apres": "keys = {r: expensive(r) for r in recs}\nsorted(recs, key=lambda r: keys[r])",
    "gain_cpu": "-50%",
    "gain_ram": "-10%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.20g CO₂ économisés pour 1M comparaisons"
  },
  {
    "id": "rule_051",
    "langage": "Python",
    "contexte": "Lisibilité et mémoire",
    "anti_pattern": "Compréhensions profondément imbriquées créant de gros intermédiaires",
    "advice": "Décomposer ou utiliser un générateur pour être paresseux.",
    "exemple_avant": "res = [[f(x,y) for y in ys] for x in xs]",
    "exemple_apres": "res = (f(x, y) for x in xs for y in ys)  # générateur",
    "gain_cpu": "-15%",
    "gain_ram": "-35%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.12g CO₂ économisés pour 1M éléments"
  },
  {
    "id": "rule_052",
    "langage": "Python",
    "contexte": "Fichiers / OS",
    "anti_pattern": "Appels os.path répétés dans une boucle sans mise en cache",
    "advice": "Mettre en variable les fonctions utilisées fréquemment.",
    "exemple_avant": "import os\nfor p in paths:\n    if os.path.exists(os.path.join(base, p)):\n        handle(p)",
    "exemple_apres": "import os\njoin, exists = os.path.join, os.path.exists\nfor p in paths:\n    if exists(join(base, p)):\n        handle(p)",
    "gain_cpu": "-10%",
    "gain_ram": "-0%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.04g CO₂ économisés pour 1M chemins"
  },
  {
    "id": "rule_053",
    "langage": "Python",
    "contexte": "Système / sous-processus",
    "anti_pattern": "Appels subprocess répétés dans une boucle",
    "advice": "Regrouper en un appel batch quand c’est possible.",
    "exemple_avant": "import subprocess\nfor f in files:\n    subprocess.run(['cmd', f])",
    "exemple_apres": "import subprocess\nsubprocess.run(['cmd'] + files)",
    "gain_cpu": "-40%",
    "gain_ram": "-10%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.22g CO₂ économisés pour 1M fichiers"
  },
  {
    "id": "rule_054",
    "langage": "Python",
    "contexte": "I/O fichiers",
    "anti_pattern": "Écritures non tamponnées sur de gros volumes",
    "advice": "Utiliser io.BufferedWriter ou mmap pour très gros fichiers.",
    "exemple_avant": "with open(path, 'wb', buffering=0) as f:\n    for b in blocks:\n        f.write(b)",
    "exemple_apres": "from io import BufferedWriter\nwith BufferedWriter(open(path, 'wb')) as f:\n    for b in blocks:\n        f.write(b)",
    "gain_cpu": "-20%",
    "gain_ram": "-10%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.10g CO₂ économisés pour 1M blocs"
  },
  {
    "id": "rule_055",
    "langage": "Python",
    "contexte": "pandas (jointure)",
    "anti_pattern": "Jointures faites ligne par ligne",
    "advice": "Utiliser DataFrame.merge pour joindre en vectorisé.",
    "exemple_avant": "for _, r in df1.iterrows():\n    df1.loc[_, 'val'] = df2[df2.id == r.id]['val'].values[0]",
    "exemple_apres": "df = df1.merge(df2[['id','val']], on='id', how='left')",
    "gain_cpu": "-75%",
    "gain_ram": "-50%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.32g CO₂ économisés pour 1M lignes"
  },
  {
    "id": "rule_056",
    "langage": "Python",
    "contexte": "pandas (agrégation)",
    "anti_pattern": "Agrégation manuelle au lieu de groupby vectorisé",
    "advice": "Utiliser groupby + opérations vectorisées.",
    "exemple_avant": "tot = {}\nfor _, r in df.iterrows():\n    tot[r['k']] = tot.get(r['k'], 0) + r['v']",
    "exemple_apres": "res = df.groupby('k')['v'].sum().reset_index()",
    "gain_cpu": "-70%",
    "gain_ram": "-20%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.26g CO₂ économisés pour 1M lignes"
  },
  {
    "id": "rule_057",
    "langage": "Python",
    "contexte": "Qualité de code",
    "anti_pattern": "Constantes en dur dupliquées",
    "advice": "Centraliser les constantes dans des variables/config.",
    "exemple_avant": "price = base * (1 + 0.2)",
    "exemple_apres": "TAX = 0.2\nprice = base * (1 + TAX)",
    "gain_cpu": "-0%",
    "gain_ram": "-5%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "Impact faible mais améliore la maintenabilité"
  },
  {
    "id": "rule_058",
    "langage": "Python",
    "contexte": "Journalisation",
    "anti_pattern": "Journalisation excessive dans les boucles critiques",
    "advice": "Réduire le niveau/volume ou utiliser des handlers asynchrones.",
    "exemple_avant": "import logging\nfor rec in recs:\n    logging.info('rec=%s', rec)",
    "exemple_apres": "import logging\nlogger = logging.getLogger(__name__)\nif logger.isEnabledFor(logging.INFO):\n    logger.info('Processed %d records', len(recs))",
    "gain_cpu": "-30%",
    "gain_ram": "-10%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.18g CO₂ économisés pour 1M logs"
  },
  {
    "id": "rule_059",
    "langage": "Python",
    "contexte": "Gestion d’exceptions",
    "anti_pattern": "Capture trop large couvrant de longues sections",
    "advice": "Encadrer finement et cibler les exceptions pertinentes.",
    "exemple_avant": "try:\n    big_block()\nexcept Exception:\n    handle()",
    "exemple_apres": "try:\n    risky()\nexcept SpecificError:\n    handle()",
    "gain_cpu": "-10%",
    "gain_ram": "-0%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.04g CO₂ économisés pour 1M passages"
  },
  {
    "id": "rule_060",
    "langage": "Python",
    "contexte": "CSV et parsing",
    "anti_pattern": "Parsing CSV manuel avec split au lieu d’un parseur",
    "advice": "Utiliser csv.reader ou pandas pour un parsing robuste.",
    "exemple_avant": "for line in open('f.csv'):\n    cols = line.strip().split(',')\n    use(cols)",
    "exemple_apres": "import csv\nwith open('f.csv') as f:\n    for row in csv.reader(f):\n        use(row)",
    "gain_cpu": "-25%",
    "gain_ram": "-20%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.12g CO₂ économisés pour 1M lignes"
  },
    {
    "id": "rule_061",
    "langage": "Python",
    "contexte": "Itérateurs et collections",
    "anti_pattern": "Conversion inutile d’itérables en liste avec list() sans besoin",
    "advice": "Conserver les générateurs/itérateurs tant qu’une liste matérielle n’est pas requise.",
    "exemple_avant": "items = list(generator())\nfor x in items:\n    use(x)",
    "exemple_apres": "for x in generator():\n    use(x)",
    "gain_cpu": "-15%",
    "gain_ram": "-35%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.10g CO₂ économisés pour 1M éléments"
  },
  {
    "id": "rule_062",
    "langage": "Python",
    "contexte": "XML et sérialisation",
    "anti_pattern": "Construction complète d’un arbre XML en mémoire avant écriture",
    "advice": "Utiliser des écritures incrémentales/streaming (iterwrite) pour limiter l’empreinte mémoire.",
    "exemple_avant": "from xml.etree.ElementTree import Element, SubElement, tostring\nroot = Element('root')\nfor r in rows:\n    e = SubElement(root, 'row')\n    e.text = r\nopen('f.xml','wb').write(tostring(root))",
    "exemple_apres": "from lxml.etree import xmlfile\nwith xmlfile('f.xml', encoding='utf-8') as xf:\n    with xf.element('root'):\n        for r in rows:\n            with xf.element('row'):\n                xf.write(r)",
    "gain_cpu": "-20%",
    "gain_ram": "-60%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.18g CO₂ économisés pour 1M éléments"
  },
  {
    "id": "rule_063",
    "langage": "Python",
    "contexte": "I/O réseau",
    "anti_pattern": "Appels d’API non groupés un par un dans une boucle",
    "advice": "Regrouper les requêtes (batch), pagination ou bulk endpoints.",
    "exemple_avant": "for rec in records:\n    requests.post(url, json=rec)",
    "exemple_apres": "batch = list(chunk(records, 100))\nfor part in batch:\n    requests.post(url+'/bulk', json=part)",
    "gain_cpu": "-10%",
    "gain_ram": "-10%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.14g CO₂ économisés pour 1M requêtes"
  },
  {
    "id": "rule_064",
    "langage": "Python",
    "contexte": "pandas et mémoire",
    "anti_pattern": "deepcopy() de DataFrames sans nécessité",
    "advice": "Éviter deepcopy, préférer les vues ou copies légères et opérations en place.",
    "exemple_avant": "from copy import deepcopy\ndf2 = deepcopy(df)\ndf2['x'] = df2['x'] * 2",
    "exemple_apres": "df['x'] = df['x'] * 2  # si acceptable\n# ou df2 = df.copy(deep=False)",
    "gain_cpu": "-20%",
    "gain_ram": "-50%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.22g CO₂ économisés pour 1M lignes"
  },
  {
    "id": "rule_065",
    "langage": "Python",
    "contexte": "Dates et séries temporelles",
    "anti_pattern": "Itération jour par jour en Python pour générer des dates",
    "advice": "Utiliser pandas.date_range ou numpy.datetime64 pour vectoriser.",
    "exemple_avant": "from datetime import date, timedelta\ncur = start\ndates = []\nwhile cur <= end:\n    dates.append(cur)\n    cur += timedelta(days=1)",
    "exemple_apres": "import pandas as pd\ndates = pd.date_range(start, end, freq='D')",
    "gain_cpu": "-60%",
    "gain_ram": "-15%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.20g CO₂ économisés pour 1M dates"
  },
  {
    "id": "rule_066",
    "langage": "Python",
    "contexte": "Ensembles et opérations ensemblistes",
    "anti_pattern": "Boucles manuelles pour union/intersection/différence",
    "advice": "Utiliser les opérations de set (|, &, -).",
    "exemple_avant": "common = []\nfor x in A:\n    if x in B:\n        common.append(x)",
    "exemple_apres": "common = list(set(A) & set(B))",
    "gain_cpu": "-70%",
    "gain_ram": "-20%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.26g CO₂ économisés pour 1M comparaisons"
  },
  {
    "id": "rule_067",
    "langage": "Python",
    "contexte": "Fichiers compressés",
    "anti_pattern": "Compression/décompression gzip sans buffering",
    "advice": "Utiliser gzip.open avec buffer et mode adapté.",
    "exemple_avant": "import gzip\nwith gzip.open('f.gz','wb', compresslevel=9) as f:\n    for b in blocks:\n        f.write(b)",
    "exemple_apres": "import gzip\nfrom io import BufferedWriter\nwith gzip.open('f.gz','wb', compresslevel=6) as g:\n    with BufferedWriter(g) as f:\n        for b in blocks:\n            f.write(b)",
    "gain_cpu": "-25%",
    "gain_ram": "-10%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.12g CO₂ économisés pour 1M blocs"
  },
  {
    "id": "rule_068",
    "langage": "Python",
    "contexte": "JSON volumineux",
    "anti_pattern": "Chargement complet des gros JSON en mémoire",
    "advice": "Utiliser ijson pour le streaming incrémental.",
    "exemple_avant": "import json\nobj = json.load(open('big.json'))\nfor rec in obj['items']:\n    process(rec)",
    "exemple_apres": "import ijson\nwith open('big.json','rb') as f:\n    for rec in ijson.items(f, 'items.item'):\n        process(rec)",
    "gain_cpu": "-20%",
    "gain_ram": "-70%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.20g CO₂ économisés pour 1M objets"
  },
  {
    "id": "rule_069",
    "langage": "Python",
    "contexte": "Aléatoire",
    "anti_pattern": "Reseeding répété des générateurs aléatoires dans les boucles",
    "advice": "Initialiser les graines une fois au début du run.",
    "exemple_avant": "import random\nfor i in range(n):\n    random.seed(123)\n    do(random.random())",
    "exemple_apres": "import random\nrandom.seed(123)\nfor i in range(n):\n    do(random.random())",
    "gain_cpu": "-15%",
    "gain_ram": "-0%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.04g CO₂ économisés pour 1M itérations"
  },
  {
    "id": "rule_070",
    "langage": "Python",
    "contexte": "Statistiques vectorisées",
    "anti_pattern": "Calculs de stats élément par élément dans des boucles Python",
    "advice": "Utiliser numpy.mean/std ou les méthodes pandas.",
    "exemple_avant": "s = 0\nfor x in arr:\n    s += x\nmean = s / len(arr)",
    "exemple_apres": "import numpy as np\nmean = np.mean(arr)",
    "gain_cpu": "-75%",
    "gain_ram": "-15%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.28g CO₂ économisés pour 1M éléments"
  },
  {
    "id": "rule_071",
    "langage": "Python",
    "contexte": "Parsing HTML",
    "anti_pattern": "Parsing HTML avec regex ou split manuels",
    "advice": "Utiliser lxml ou BeautifulSoup pour parser de manière robuste.",
    "exemple_avant": "import re\nlinks = re.findall(r'<a href=\"(.*?)\"', html)",
    "exemple_apres": "from bs4 import BeautifulSoup\nsoup = BeautifulSoup(html, 'html.parser')\nlinks = [a['href'] for a in soup.find_all('a', href=True)]",
    "gain_cpu": "-20%",
    "gain_ram": "-10%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.10g CO₂ économisés pour 1M balises"
  },
  {
    "id": "rule_072",
    "langage": "Python",
    "contexte": "Itérateurs et zip",
    "anti_pattern": "Création de grosses listes intermédiaires avec list(zip(...))",
    "advice": "Itérer directement sur zip (générateur) ou utiliser itertools.",
    "exemple_avant": "pairs = list(zip(a, b))\nfor x, y in pairs:\n    use(x, y)",
    "exemple_apres": "for x, y in zip(a, b):\n    use(x, y)",
    "gain_cpu": "-10%",
    "gain_ram": "-40%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.12g CO₂ économisés pour 1M paires"
  },
  {
    "id": "rule_073",
    "langage": "Python",
    "contexte": "Dictionnaires",
    "anti_pattern": "try/except KeyError répété pour accéder à des clés manquantes",
    "advice": "Utiliser dict.get avec valeur par défaut.",
    "exemple_avant": "try:\n    v = d[k]\nexcept KeyError:\n    v = 0",
    "exemple_apres": "v = d.get(k, 0)",
    "gain_cpu": "-25%",
    "gain_ram": "-0%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.10g CO₂ économisés pour 1M accès"
  },
  {
    "id": "rule_074",
    "langage": "Python",
    "contexte": "Validation de schémas",
    "anti_pattern": "Reconstruction du schéma à chaque validation",
    "advice": "Mettre en cache/instancier une fois l’objet de schéma et le réutiliser.",
    "exemple_avant": "for doc in docs:\n    schema = fastjsonschema.compile(spec)\n    schema(doc)",
    "exemple_apres": "schema = fastjsonschema.compile(spec)\nfor doc in docs:\n    schema(doc)",
    "gain_cpu": "-55%",
    "gain_ram": "-10%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.22g CO₂ économisés pour 1M documents"
  },
  {
    "id": "rule_075",
    "langage": "Python",
    "contexte": "Boucles de production",
    "anti_pattern": "time.sleep répétés au lieu d’attentes événementielles",
    "advice": "Utiliser des sémaphores, files d’attente ou sélecteurs d’événements.",
    "exemple_avant": "while not ready():\n    time.sleep(0.1)",
    "exemple_apres": "event.wait()  # signalé par le producteur",
    "gain_cpu": "-20%",
    "gain_ram": "-0%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "Réduction d’attente active; impact dépend du cas"
  },
  {
    "id": "rule_076",
    "langage": "Python",
    "contexte": "Multiprocessing et mémoire",
    "anti_pattern": "Transfert de gros objets entre processus (pickling massif)",
    "advice": "Partager la mémoire (multiprocessing.Array/Manager) ou mapper les fichiers.",
    "exemple_avant": "from multiprocessing import Pool\nwith Pool() as p:\n    p.map(work, [big_object]*n)",
    "exemple_apres": "from multiprocessing import Pool, Manager\nwith Manager() as m:\n    shared = m.list(big_object)\n    with Pool() as p:\n        p.map(work_shared, [shared]*n)",
    "gain_cpu": "-30%",
    "gain_ram": "-40%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.18g CO₂ économisés pour 1M items transférés"
  },
  {
    "id": "rule_077",
    "langage": "Python",
    "contexte": "Aléatoire",
    "anti_pattern": "Mélange manuel d’une liste par swaps dans une boucle",
    "advice": "Utiliser random.shuffle ou numpy.random.permutation.",
    "exemple_avant": "for i in range(len(a)):\n    j = random.randint(0, len(a)-1)\n    a[i], a[j] = a[j], a[i]",
    "exemple_apres": "import random\nrandom.shuffle(a)",
    "gain_cpu": "-35%",
    "gain_ram": "-0%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.14g CO₂ économisés pour 1M échanges"
  },
  {
    "id": "rule_078",
    "langage": "Python",
    "contexte": "Gestion d’exceptions",
    "anti_pattern": "Relancer une exception en perdant la trace (raise e)",
    "advice": "Relancer avec raise sans argument pour conserver la stack.",
    "exemple_avant": "try:\n    risky()\nexcept Exception as e:\n    log(e)\n    raise e",
    "exemple_apres": "try:\n    risky()\nexcept Exception:\n    log('failed')\n    raise",
    "gain_cpu": "-0%",
    "gain_ram": "-0%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "Impact surtout sur débogage/maintenabilité"
  },
  {
    "id": "rule_079",
    "langage": "Python",
    "contexte": "Fichiers et encodage",
    "anti_pattern": "Écriture de fichiers texte là où le binaire suffit (surcharge d’encodage)",
    "advice": "Écrire en mode binaire quand c’est possible.",
    "exemple_avant": "with open('f.out','w') as f:\n    f.write(bytes_data.decode('latin1'))",
    "exemple_apres": "with open('f.out','wb') as f:\n    f.write(bytes_data)",
    "gain_cpu": "-20%",
    "gain_ram": "-5%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.10g CO₂ économisés pour 1M écritures"
  },
  {
    "id": "rule_080",
    "langage": "Python",
    "contexte": "JSON (sérialisation)",
    "anti_pattern": "json.dumps lent pour de très gros dumps",
    "advice": "Utiliser orjson/rapidjson pour des dumps rapides.",
    "exemple_avant": "import json\nout = json.dumps(obj)",
    "exemple_apres": "import orjson\nout = orjson.dumps(obj)",
    "gain_cpu": "-50%",
    "gain_ram": "-15%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.22g CO₂ économisés pour 1M objets"
  },
  {
    "id": "rule_081",
    "langage": "Python",
    "contexte": "JSON (options)",
    "anti_pattern": "Tri des clés (sort_keys=True) sans nécessité",
    "advice": "Désactiver sort_keys pour réduire le CPU.",
    "exemple_avant": "json.dumps(obj, sort_keys=True)",
    "exemple_apres": "json.dumps(obj, sort_keys=False)",
    "gain_cpu": "-25%",
    "gain_ram": "-0%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.10g CO₂ économisés pour 1M dumps"
  },
  {
    "id": "rule_082",
    "langage": "Python",
    "contexte": "Hachage et dictionnaires",
    "anti_pattern": "Implémentations personnalisées de hash dans des boucles serrées",
    "advice": "Utiliser les hash intégrés et types immuables (tuple, frozenset).",
    "exemple_avant": "def my_hash(x):\n    return sum(ord(c) for c in x)\nkeys = {my_hash(k): v for k,v in items}",
    "exemple_apres": "keys = {hash(k): v for k,v in items}",
    "gain_cpu": "-40%",
    "gain_ram": "-0%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.16g CO₂ économisés pour 1M hachages"
  },
  {
    "id": "rule_083",
    "langage": "Python",
    "contexte": "I/O réseau",
    "anti_pattern": "Requêtes HTTP synchrones séquentielles dans une boucle",
    "advice": "Utiliser asyncio/aiohttp pour paralléliser I/O non bloquants.",
    "exemple_avant": "import requests\nfor url in urls:\n    data = requests.get(url).text\n    consume(data)",
    "exemple_apres": "import asyncio, aiohttp\nasync def fetch_all(urls):\n    async with aiohttp.ClientSession() as s:\n        tasks = [s.get(u) for u in urls]\n        for r in await asyncio.gather(*tasks):\n            consume(await r.text())\nasyncio.run(fetch_all(urls))",
    "gain_cpu": "-10%",
    "gain_ram": "-10%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "Réduction du temps d’attente réseau; CO₂ ↓ si infra partagée"
  },
  {
    "id": "rule_084",
    "langage": "Python",
    "contexte": "Traitement d’images",
    "anti_pattern": "Boucles pixel-par-pixel en Python pur",
    "advice": "Vectoriser avec OpenCV/Pillow ou NumPy.",
    "exemple_avant": "for i in range(h):\n    for j in range(w):\n        img[i,j] = min(img[i,j]*2, 255)",
    "exemple_apres": "import numpy as np\nimg = np.clip(img * 2, 0, 255)",
    "gain_cpu": "-85%",
    "gain_ram": "-15%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.34g CO₂ économisés pour 1M pixels"
  },
  {
    "id": "rule_085",
    "langage": "Python",
    "contexte": "Cache et mémoire",
    "anti_pattern": "Mise en cache d’objets énormes en mémoire locale",
    "advice": "Limiter la taille du cache, utiliser weakref ou cache externe (Redis).",
    "exemple_avant": "CACHE = {}\nCACHE['big'] = big_object",
    "exemple_apres": "import weakref\nCACHE = weakref.WeakValueDictionary()\nCACHE['big'] = big_object",
    "gain_cpu": "-0%",
    "gain_ram": "-45%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "Réduction des pics mémoire; impact selon taille"
  },
  {
    "id": "rule_086",
    "langage": "Python",
    "contexte": "NumPy et copies",
    "anti_pattern": "deepcopy de tableaux NumPy",
    "advice": "Utiliser numpy.copy() (ou vues) au lieu de deepcopy.",
    "exemple_avant": "from copy import deepcopy\nb = deepcopy(a)",
    "exemple_apres": "b = a.copy()  # ou b = a.view()",
    "gain_cpu": "-25%",
    "gain_ram": "-35%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.16g CO₂ économisés pour 1M éléments"
  },
  {
    "id": "rule_087",
    "langage": "Python",
    "contexte": "Graphes et algorithmes",
    "anti_pattern": "DFS/BFS maison non optimisés",
    "advice": "Utiliser networkx ou bibliothèques optimisées.",
    "exemple_avant": "def bfs(g, s):\n    ...  # implémentation Python pure\nres = bfs(graph, start)",
    "exemple_apres": "import networkx as nx\nG = nx.from_dict_of_lists(graph)\nres = list(nx.bfs_tree(G, start))",
    "gain_cpu": "-50%",
    "gain_ram": "-15%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.22g CO₂ économisés pour 1M arêtes"
  },
  {
    "id": "rule_088",
    "langage": "Python",
    "contexte": "NumPy (reshape)",
    "anti_pattern": "Reformatage de tableaux via boucles et affectations manuelles",
    "advice": "Utiliser numpy.reshape/ravel pour réorganiser sans copier si possible.",
    "exemple_avant": "b = np.empty((n*m,))\nk = 0\nfor i in range(n):\n    for j in range(m):\n        b[k] = a[i,j]; k += 1",
    "exemple_apres": "b = a.ravel()",
    "gain_cpu": "-80%",
    "gain_ram": "-20%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.30g CO₂ économisés pour 1M éléments"
  },
  {
    "id": "rule_089",
    "langage": "Python",
    "contexte": "JSON (parsing)",
    "anti_pattern": "json.loads appelé de manière répétée sur les mêmes chaînes",
    "advice": "Parser une fois, réutiliser les objets ou pré-décoder en amont.",
    "exemple_avant": "for s in strings:\n    obj = json.loads(s)\n    use(obj)\n# plus tard\nfor s in strings:\n    obj = json.loads(s)\n    use2(obj)",
    "exemple_apres": "decoded = [json.loads(s) for s in strings]\nfor obj in decoded:\n    use(obj)\nfor obj in decoded:\n    use2(obj)",
    "gain_cpu": "-45%",
    "gain_ram": "-10%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.18g CO₂ économisés pour 1M objets"
  },
  {
    "id": "rule_090",
    "langage": "Python",
    "contexte": "SQLAlchemy et bases de données",
    "anti_pattern": "Plusieurs requêtes dans une boucle pour simuler une jointure",
    "advice": "Utiliser des join/outerjoin SQLAlchemy et récupérer en une fois.",
    "exemple_avant": "for u in session.query(User).all():\n    addr = session.query(Address).filter(Address.user_id==u.id).first()\n    use(u, addr)",
    "exemple_apres": "q = session.query(User, Address).join(Address, User.id==Address.user_id)\nfor u, addr in q:\n    use(u, addr)",
    "gain_cpu": "-55%",
    "gain_ram": "-20%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.24g CO₂ économisés pour 1M lignes jointes"
  },
    {
    "id": "rule_091",
    "langage": "Python",
    "contexte": "XML (sérialisation)",
    "anti_pattern": "Sérialisation XML lente avec ElementTree par défaut",
    "advice": "Utiliser cElementTree ou lxml pour une sérialisation plus rapide.",
    "exemple_avant": "import xml.etree.ElementTree as ET\nroot = ET.Element('root'); ET.SubElement(root,'x').text='1'\nxml = ET.tostring(root)",
    "exemple_apres": "from lxml import etree as ET\nroot = ET.Element('root'); ET.SubElement(root,'x').text='1'\nxml = ET.tostring(root, encoding='utf-8')",
    "gain_cpu": "-30%",
    "gain_ram": "-10%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.12g CO₂ économisés pour 1M nœuds"
  },
  {
    "id": "rule_092",
    "langage": "Python",
    "contexte": "pandas (indexation)",
    "anti_pattern": "Indexation chaînée (chained indexing) provoquant des copies cachées",
    "advice": "Utiliser .loc/.iloc pour des sélections explicites et sans copie inutile.",
    "exemple_avant": "df[df['a']>0]['b'] = 1",
    "exemple_apres": "df.loc[df['a']>0, 'b'] = 1",
    "gain_cpu": "-25%",
    "gain_ram": "-40%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.18g CO₂ économisés pour 1M lignes"
  },
  {
    "id": "rule_093",
    "langage": "Python",
    "contexte": "pandas (élément par élément)",
    "anti_pattern": "Utilisation excessive de DataFrame.applymap pour des transformations élémentaires",
    "advice": "Privilégier les opérations vectorisées ou .where/.mask.",
    "exemple_avant": "df = df.applymap(lambda x: x*2)",
    "exemple_apres": "df = df * 2",
    "gain_cpu": "-70%",
    "gain_ram": "-20%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.26g CO₂ économisés pour 1M cellules"
  },
  {
    "id": "rule_094",
    "langage": "Python",
    "contexte": "Données creuses (sparse)",
    "anti_pattern": "Utilisation de matrices denses pour des données clairsemées",
    "advice": "Utiliser scipy.sparse (CSR/CSC) pour stocker et calculer.",
    "exemple_avant": "import numpy as np\nA = np.zeros((n,n)); A[i,j] = 1",
    "exemple_apres": "from scipy.sparse import csr_matrix\nA = csr_matrix((data, (row, col)), shape=(n,n))",
    "gain_cpu": "-50%",
    "gain_ram": "-80%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "Forte baisse de mémoire → CO₂ ↓ significativement"
  },
  {
    "id": "rule_095",
    "langage": "Python",
    "contexte": "Filtrage de données",
    "anti_pattern": "Boucles manuelles pour filtrer des listes/DataFrames",
    "advice": "Utiliser des compréhensions de liste ou pandas.query.",
    "exemple_avant": "out = []\nfor x in xs:\n    if cond(x):\n        out.append(x)",
    "exemple_apres": "out = [x for x in xs if cond(x)]  # ou df.query('a>0 & b==1')",
    "gain_cpu": "-30%",
    "gain_ram": "-10%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.12g CO₂ économisés pour 1M éléments"
  },
  {
    "id": "rule_096",
    "langage": "Python",
    "contexte": "Dictionnaires",
    "anti_pattern": "Fusion manuelle de dictionnaires via boucles imbriquées",
    "advice": "Utiliser dict.update() ou l’unpacking {**d1, **d2}.",
    "exemple_avant": "merged = {}\nfor d in dicts:\n    for k,v in d.items():\n        merged[k] = v",
    "exemple_apres": "merged = {}\nfor d in dicts:\n    merged.update(d)\n# ou merged = {k:v for d in dicts for k,v in d.items()}",
    "gain_cpu": "-20%",
    "gain_ram": "-5%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.08g CO₂ économisés pour 1M clés"
  },
  {
    "id": "rule_097",
    "langage": "Python",
    "contexte": "CSV (lecture)",
    "anti_pattern": "Laisser auto-détecter le séparateur sur de gros fichiers",
    "advice": "Définir explicitement le délimiteur pour éviter le surcoût.",
    "exemple_avant": "import pandas as pd\ndf = pd.read_csv('data.txt')",
    "exemple_apres": "import pandas as pd\ndf = pd.read_csv('data.txt', sep='\\t')",
    "gain_cpu": "-25%",
    "gain_ram": "-0%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.10g CO₂ économisés pour 1M lignes"
  },
  {
    "id": "rule_098",
    "langage": "Python",
    "contexte": "pandas (mémoire)",
    "anti_pattern": "Colonnes string répétitives stockées en dtype object",
    "advice": "Utiliser le dtype 'category' pour réduire la mémoire.",
    "exemple_avant": "df['city'] = df['city'].astype(object)",
    "exemple_apres": "df['city'] = df['city'].astype('category')",
    "gain_cpu": "-5%",
    "gain_ram": "-60%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "Réduction forte des allocations mémoire"
  },
  {
    "id": "rule_099",
    "langage": "Python",
    "contexte": "Flux de données",
    "anti_pattern": "Matérialiser de grosses listes intermédiaires au lieu de générer à la volée",
    "advice": "Utiliser des générateurs (yield) et l’itération paresseuse.",
    "exemple_avant": "rows = [read_record(f) for f in files]\nfor r in rows:\n    use(r)",
    "exemple_apres": "def iter_rows(files):\n    for f in files:\n        yield read_record(f)\nfor r in iter_rows(files):\n    use(r)",
    "gain_cpu": "-15%",
    "gain_ram": "-50%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "Moins de pics mémoire → CO₂ ↓"
  },
  {
    "id": "rule_100",
    "langage": "Python",
    "contexte": "NumPy (précision)",
    "anti_pattern": "Utiliser float64/int64 par défaut sans nécessité",
    "advice": "Réduire la précision (float32/int32) quand c’est acceptable.",
    "exemple_avant": "a = np.array(vals, dtype=np.float64)",
    "exemple_apres": "a = np.array(vals, dtype=np.float32)",
    "gain_cpu": "-20%",
    "gain_ram": "-50%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "Moitié de mémoire sur colonnes flottantes"
  },
  {
    "id": "rule_101",
    "langage": "Python",
    "contexte": "CSV (écriture)",
    "anti_pattern": "Quoting par défaut coûteux ou incohérent",
    "advice": "Contrôler explicitement quoting pour limiter le travail sur les chaînes.",
    "exemple_avant": "import csv\nw = csv.writer(open('f.csv','w', newline=''))\nw.writerows(rows)",
    "exemple_apres": "import csv\nw = csv.writer(open('f.csv','w', newline=''), quoting=csv.QUOTE_MINIMAL)\nw.writerows(rows)",
    "gain_cpu": "-15%",
    "gain_ram": "-5%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.06g CO₂ économisés pour 1M lignes"
  },
  {
    "id": "rule_102",
    "langage": "Python",
    "contexte": "Chaînes de caractères",
    "anti_pattern": "Assemblage de très nombreuses chaînes via '+' dans une boucle",
    "advice": "Utiliser str.join sur une liste d’éléments.",
    "exemple_avant": "s = ''\nfor p in parts:\n    s += p",
    "exemple_apres": "s = ''.join(parts)",
    "gain_cpu": "-65%",
    "gain_ram": "-20%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.24g CO₂ économisés pour 1M concaténations"
  },
  {
    "id": "rule_103",
    "langage": "Python",
    "contexte": "pandas (tri)",
    "anti_pattern": "Appels de tri répétés ou sans clé précise",
    "advice": "Utiliser sort_values avec clés précises et éviter les tris multiples.",
    "exemple_avant": "df = df.sort_values(by=list(df.columns))\n# puis retrié plusieurs fois",
    "exemple_apres": "df = df.sort_values(by=['date','id'])  # une seule fois",
    "gain_cpu": "-50%",
    "gain_ram": "-15%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.20g CO₂ économisés pour 1M lignes"
  },
  {
    "id": "rule_104",
    "langage": "Python",
    "contexte": "Algèbre linéaire",
    "anti_pattern": "Inversion explicite de matrice avec np.linalg.inv pour résoudre Ax=b",
    "advice": "Utiliser np.linalg.solve (plus stable et plus rapide).",
    "exemple_avant": "x = np.linalg.inv(A) @ b",
    "exemple_apres": "x = np.linalg.solve(A, b)",
    "gain_cpu": "-35%",
    "gain_ram": "-10%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.14g CO₂ économisés pour 1M systèmes"
  },
  {
    "id": "rule_105",
    "langage": "Python",
    "contexte": "Sérialisation de données",
    "anti_pattern": "Utiliser CSV pour des datasets volumineux",
    "advice": "Préférer des formats binaires colonnes (parquet, feather).",
    "exemple_avant": "df.to_csv('data.csv', index=False)",
    "exemple_apres": "df.to_parquet('data.parquet')  # ou .to_feather(...)",
    "gain_cpu": "-30%",
    "gain_ram": "-25%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "I/O réduit → CO₂ ↓ notable sur gros volumes"
  },
  {
    "id": "rule_106",
    "langage": "Python",
    "contexte": "Téléchargements HTTP",
    "anti_pattern": "Télécharger un gros fichier en une seule fois en mémoire",
    "advice": "Utiliser requests avec stream=True et iter_content par chunks.",
    "exemple_avant": "r = requests.get(url)\nopen('f.bin','wb').write(r.content)",
    "exemple_apres": "with requests.get(url, stream=True) as r, open('f.bin','wb') as f:\n    for chunk in r.iter_content(chunk_size=1<<20):\n        if chunk: f.write(chunk)",
    "gain_cpu": "-5%",
    "gain_ram": "-80%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "Pics mémoire réduits, I/O plus stables"
  },
  {
    "id": "rule_107",
    "langage": "Python",
    "contexte": "Ensembles",
    "anti_pattern": "Intersection manuelle par boucles",
    "advice": "Utiliser set.intersection pour calculer l’intersection efficacement.",
    "exemple_avant": "common = []\nfor x in A:\n    if x in B:\n        common.append(x)",
    "exemple_apres": "common = set(A).intersection(B)",
    "gain_cpu": "-70%",
    "gain_ram": "-20%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.26g CO₂ économisés pour 1M comparaisons"
  },
  {
    "id": "rule_108",
    "langage": "Python",
    "contexte": "Logging en production",
    "anti_pattern": "Handlers de log sans rotation menant à des fichiers énormes",
    "advice": "Utiliser RotatingFileHandler/TimedRotatingFileHandler et niveaux adaptés.",
    "exemple_avant": "logging.basicConfig(filename='app.log', level=logging.INFO)",
    "exemple_apres": "from logging.handlers import RotatingFileHandler\nh = RotatingFileHandler('app.log', maxBytes=10_000_000, backupCount=5)\nlogger = logging.getLogger('app'); logger.addHandler(h)",
    "gain_cpu": "-5%",
    "gain_ram": "-5%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "I/O disque mieux maîtrisé sur longue durée"
  },
  {
    "id": "rule_109",
    "langage": "Python",
    "contexte": "CSV → pandas",
    "anti_pattern": "Laisser pandas créer un index par défaut inutile",
    "advice": "Spécifier index_col pour éviter une colonne supplémentaire.",
    "exemple_avant": "df = pd.read_csv('data.csv')",
    "exemple_apres": "df = pd.read_csv('data.csv', index_col='id')",
    "gain_cpu": "-5%",
    "gain_ram": "-15%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "Moins de colonnes → mémoire et I/O réduites"
  },
  {
    "id": "rule_110",
    "langage": "Python",
    "contexte": "Lisibilité/maintenabilité",
    "anti_pattern": "Lambdas complexes et répétées au lieu de fonctions nommées",
    "advice": "Définir une fonction nommée réutilisable (possiblement compilée/cachée).",
    "exemple_avant": "df['y'] = df['x'].apply(lambda v: (v*v + 1)/3 if v>0 else 0)",
    "exemple_apres": "def transform(v):\n    return (v*v + 1)/3 if v>0 else 0\ndf['y'] = df['x'].apply(transform)",
    "gain_cpu": "-5%",
    "gain_ram": "-0%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "Impact faible; robustesse accrue"
  },
  {
    "id": "rule_111",
    "langage": "Python",
    "contexte": "Multiprocessing (plateformes Unix)",
    "anti_pattern": "Utiliser le start method par défaut inadapté provoquant surcoûts et copies",
    "advice": "Privilégier 'forkserver' ou 'spawn' selon le besoin de stabilité/mémoire.",
    "exemple_avant": "import multiprocessing as mp\n# start method implicite\np = mp.Pool().map(f, xs)",
    "exemple_apres": "import multiprocessing as mp\nmp.set_start_method('forkserver', force=True)\nwith mp.Pool() as p:\n    p.map(f, xs)",
    "gain_cpu": "-10%",
    "gain_ram": "-20%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "Moins de copies inattendues → CO₂ ↓"
  },
  {
    "id": "rule_112",
    "langage": "Python",
    "contexte": "Système/OS",
    "anti_pattern": "Accès répétés à os.environ dans les boucles",
    "advice": "Mettre en cache les valeurs d’environnement en variables locales.",
    "exemple_avant": "for _ in range(n):\n    mode = os.environ.get('APP_MODE','dev')\n    do(mode)",
    "exemple_apres": "mode = os.environ.get('APP_MODE','dev')\nfor _ in range(n):\n    do(mode)",
    "gain_cpu": "-15%",
    "gain_ram": "-0%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "≈0.06g CO₂ économisés pour 1M accès"
  },
  {
    "id": "rule_113",
    "langage": "Python",
    "contexte": "pandas (dtypes)",
    "anti_pattern": "Types numériques non optimisés (int64/float64) par défaut",
    "advice": "Downcaster les colonnes (to_numeric(..., downcast=...), astype) pour réduire la mémoire.",
    "exemple_avant": "df['count'] = df['count'].astype('int64')",
    "exemple_apres": "df['count'] = pd.to_numeric(df['count'], downcast='integer')",
    "gain_cpu": "-5%",
    "gain_ram": "-40%",
    "outil_mesure": ["memory_profiler", "timeit"],
    "estimation_co2": "Réduction mémoire substantielle sur colonnes volumineuses"
  }
]

# ------------------------------------------------------------
# 2. Brique 2 – Structuration & Analyse des pratiques
# Embed KB entries for semantic search  (sanitized – no raw vectors exposed)
# ------------------------------------------------------------
from functools import lru_cache
import numpy as np

# Toggle: keep False to avoid storing giant vectors inside the KB
STORE_KB_VECTORS = False

@lru_cache(maxsize=4096)
def _embed_cached(text: str, input_type: str = "query") -> tuple:
    emb = client.embeddings.create(
        model="nvidia/nv-embedqa-e5-v5",
        input=text,
        extra_body={"input_type": input_type}  # must be "query" or "passage"
    )
    # tuple for cacheability; we won't print or store these in the KB
    return tuple(emb.data[0].embedding)

def embed_text(text: str, input_type: str = "query") -> np.ndarray:
    return np.asarray(_embed_cached(text, input_type=input_type), dtype=np.float32)

# Embed KB entries as "passages" (kept for compatibility, but gated)
# NOTE: we do NOT store real vectors unless STORE_KB_VECTORS=True.
for rule in knowledge_base:
    if STORE_KB_VECTORS:
        rule["vector"] = embed_text(rule.get("advice", ""), input_type="passage")
    else:
        # keep the key present but lightweight to respect existing code expectations
        rule["vector"] = None

# Naive similarity function (dot product)
def cosine_similarity(vec1, vec2):
    v1, v2 = np.asarray(vec1), np.asarray(vec2)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) or 1.0
    return float(v1.dot(v2) / denom)

# keep only light rule fields in outputs
LIGHT_RULE_KEYS = ("id", "contexte", "anti_pattern", "advice", "gain_cpu", "gain_ram", "estimation_co2")
def _sanitize_rules(rules: list[dict]) -> list[dict]:
    out = []
    for r in rules:
        out.append({k: r[k] for k in LIGHT_RULE_KEYS if k in r})
    return out

def retrieve_best_practices(code: str, top_k: int = 1) -> List[Dict]:
    # Compute similarity on-the-fly; don't rely on stored vectors
    q = embed_text(code, input_type="query")
    scored = []
    for r in knowledge_base:
        pv = embed_text(r.get("advice", ""), input_type="passage")
        scored.append((cosine_similarity(q, pv), r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return _sanitize_rules([r for _, r in scored[:top_k]])

# ------------------------------------------------------------
# 3. Brique 3 – Analyse du code soumis
# Simple AST-based check for inefficiencies
# ------------------------------------------------------------
def analyze_code(code: str) -> Dict:
    analysis = {"language": "python", "patterns": []}
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # check for += str() concatenation inside loop
                for n in ast.walk(node):
                    if isinstance(n, ast.AugAssign) and isinstance(n.op, ast.Add):
                        analysis["patterns"].append("concat string in loop")
    except Exception as e:
        analysis["error"] = str(e)
    return analysis

# ------------------------------------------------------------
# 3bis. Analyse AST native + Hybride (AST + Embeddings)
# ------------------------------------------------------------
LANGUAGE_REGEX = {
    "Python": [r"\bdef\b", r"\bimport\b", r":\s*(#.*)?$", r"\bprint\(", r"\bNone\b", r"^\s*#"],
    "JavaScript": [r"\bfunction\b", r"=>", r"\bconsole\.log\b", r"\b(var|let|const)\b", r";\s*$", r"\bimport\s"],
    "Java": [r"\bpublic\s+class\b", r"\bpublic\s+static\s+void\s+main\b", r"\bSystem\.out\.println\b", r";\s*$"],
    "C++": [r"#include\s*<", r"\bstd::", r"\bcout\s*<<", r";\s*$"],
}

def _language_by_regex(code: str):
    scores = {}
    for lang, pats in LANGUAGE_REGEX.items():
        s = sum(1 for p in pats if re.search(p, code, flags=re.MULTILINE))
        scores[lang] = float(s)
    best = max(scores.items(), key=lambda x: x[1])[0] if scores else "Unknown"
    maxs = max(scores.values()) or 1.0
    return best, {k: v / maxs for k, v in scores.items()}

def _detect_python_patterns(code: str):
    patterns, tags = [], []
    try:
        tree = ast.parse(code)
    except Exception:
        tree = None
    if tree:
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                for n in ast.walk(node):
                    if isinstance(n, ast.AugAssign) and isinstance(n.op, ast.Add):
                        patterns.append("concat string in loop")
                        tags.append("strings:concat_in_loop")
                        break
    if re.search(r'(^|\s)open\s*\(', code) and not re.search(r'with\s+open\s*\(', code):
        patterns.append("file open without context manager"); tags.append("io:missing_with_open")
    if "import re" in code and re.search(r'for\s+.+:\s*[\s\S]*?re\.(search|match|findall)\(', code, re.MULTILINE):
        if "re.compile(" not in code: patterns.append("regex not compiled inside loop"); tags.append("regex:missing_compile")
    if re.search(r'for\s+\w+\s+in\s+range\(\s*len\(', code):
        patterns.append("len(list) inside loop range"); tags.append("loops:len_in_loop")
    if re.search(r'\.apply\(', code) or re.search(r'\.iterrows\(', code):
        patterns.append("row-wise pandas apply/iterrows"); tags.append("pandas:row_wise")
    if re.search(r'for\s+\w+\s+in\s+.+:\s*[\s\S]*?\.append\(', code, re.MULTILINE):
        patterns.append("append in large loop"); tags.append("lists:append_loop")
    return patterns, tags

def _detect_language(code: str) -> str:
    lang, _ = _language_by_regex(code)
    return lang

def _detect_libraries(code: str, language: str) -> List[str]:
    libs = []
    if language == "Python":
        try:
            tree = ast.parse(code)
            for n in ast.walk(tree):
                if isinstance(n, ast.Import):
                    libs += [a.name.split(".")[0] for a in n.names]
                elif isinstance(n, ast.ImportFrom) and n.module:
                    libs.append(n.module.split(".")[0])
        except SyntaxError:
            pass
    elif language == "JavaScript":
        libs += re.findall(r"from\s+['\"]([^'\"]+)['\"]", code)
        libs += re.findall(r"require\(\s*['\"]([^'\"]+)['\"]\s*\)", code)
    return sorted(set(libs))

def analyze_native_ast(code: str) -> Dict:
    lang = _detect_language(code)
    libs = _detect_libraries(code, lang)
    _, tags = _detect_python_patterns(code) if lang == "Python" else ([], [])
    type_map = {
        "strings": "Chaînes de caractères",
        "loops": "Algorithmes et boucles",
        "regex": "Regex et parsing",
        "io": "Fichiers et I/O",
        "pandas": "pandas",
        "numpy": "NumPy",
        "lists": "Structures de données",
    }
    opti_types = sorted({v for k, v in type_map.items() if any(t.startswith(k + ":") for t in tags)})
    return {
        "approach": "Analyse AST native",
        "language": lang,
        "libraries": libs,
        "optimization_types": opti_types,
        "motifs": sorted(set(tags)),
    }

def analyze_hybrid_ast_embeddings(code: str, top_k: int = 3) -> Dict:
    base = analyze_native_ast(code)
    recs = retrieve_best_practices(code, top_k=top_k)  # sanitized rules only
    base.update({
        "approach": "Modèle hybride AST + Embeddings",
        "recommendations": recs
    })
    return base

# ------------------------------------------------------------
# 4. Brique 4 – LLM Suggestion & Optimization (NVIDIA NIM)
# (no streaming, no 'thinking')
# ------------------------------------------------------------
def suggest_optimization(user_code: str) -> str:
    # Step A: Analyze code
    analysis = analyze_code(user_code)
    # Step B: Retrieve KB rules (sanitized)
    best_rules = retrieve_best_practices(user_code, top_k=1)
    # Step C: Build RAG prompt
    rag_prompt = f"""
    User code:
    {user_code}

    Detected patterns: {analysis.get("patterns")}

    Relevant Green practices:
    {json.dumps(best_rules, indent=2, ensure_ascii=False)}

    Task: Rewrite the code in a more sustainable way,
    explain the optimizations and their environmental benefits.
    """
    # Step D: Query NVIDIA NIM model
    completion = client.chat.completions.create(
        model="qwen/qwen3-235b-a22b",
        messages=[
            {"role": "system", "content": "You are GreenCoder, an assistant that improves code for sustainability."},
            {"role": "user", "content": rag_prompt}
        ],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024
    )
    # Get output
    final_answer = completion.choices[0].message.content
    print(final_answer)
    return final_answer

# ==================================================================================================
# MÉTHODE 1 : Regex + Heuristiques (kept; sanitize kb_rules)
# ==================================================================================================
def _kb_match_rules_keyword(knowledge_base: List[Dict], code: str, max_rules: int = 5) -> List[Dict]:
    signals = {
        "concat": ["+=", "''.join", "join("],
        "regex": ["re.", "regex", "re.compile("],
        "pandas": ["pandas", "pd.", ".apply(", ".iterrows("],
        "numpy": ["numpy", "np."],
        "io": ["open(", "write(", "read("],
        "set": [" set(", "dict(", "defaultdict("],
        "cache": ["lru_cache", "@lru_cache"],
        "copy": ["deepcopy(", ".copy("],
    }
    low_code = code.lower()
    scored = []
    for r in knowledge_base:
        text = " ".join([str(r.get("advice") or r.get("advice") or ""),
                         str(r.get("anti_pattern") or ""),
                         str(r.get("contexte") or "")]).lower()
        s = 0
        for k, keys in signals.items():
            if any(kx.lower() in low_code for kx in keys) and k in text: s += 2
            if k in text: s += 1
        if any(w in text for w in ["boucle","vectoris","pandas","numpy","json","io","regex"]) and \
           any(w in low_code for w in ["for ", "while ", "apply(", "np.", "pd.", "open("]):
            s += 1
        scored.append((s, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return _sanitize_rules([r for s, r in scored if s > 0][:max_rules])

def method1_regex_heuristics(code: str, knowledge_base: List[Dict]) -> Dict:
    language, lang_scores = _language_by_regex(code)
    patterns, tags = _detect_python_patterns(code)
    matched_rules = _kb_match_rules_keyword(knowledge_base, code, max_rules=6)
    type_map = {
        "strings": "Chaînes de caractères",
        "loops": "Algorithmes et boucles",
        "regex": "Regex et parsing",
        "io": "Fichiers et I/O",
        "pandas": "pandas",
        "numpy": "NumPy",
        "lists": "Structures de données",
    }
    opti_types = sorted({v for k, v in type_map.items() if any(t.startswith(k + ":") for t in tags)})
    return {
        "language": language,
        "language_scores": lang_scores,
        "optimization_types": opti_types,
        "tags": sorted(set(tags + patterns)),
        "kb_rules": matched_rules,   # sanitized
        "inconsistencies": []
    }

# ==================================================================================================
# MÉTHODE 4 : Classification supervisée (RandomForest)  (kept; sanitize kb_rules)
# ==================================================================================================
def _get_rf_models():
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception as e:
        raise RuntimeError("Installe scikit-learn: pip install scikit-learn") from e

    LANG_DOCS = [
        "def f(x):\n    import math\n    return x+1\nprint('ok')",
        "function f(x){ console.log(x+1); }",
        "public class A { public static void main(String[] a){ System.out.println(1); } }",
        "#include <iostream>\nint main(){ std::cout<<1; }",
    ]
    LANG_LBLS = ["Python", "JavaScript", "Java", "C++"]

    OPTI_DOCS = [
        "s='';\nfor x in data:\n    s += str(x)",
        "with open(p,'w') as f:\n    f.write('x')",
        "for i in range(len(lst)):\n    total+=lst[i]",
        "import pandas as pd\ndf.apply(lambda r: r.x+1, axis=1)",
        "import re\nfor l in lines:\n    if re.search('[0-9]+', l): pass",
        "import numpy as np\nout = a * 2",
    ]
    OPTI_LBLS = ["Chaînes de caractères", "Fichiers et I/O", "Algorithmes et boucles",
                 "pandas", "Regex et parsing", "NumPy"]

    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer

    lang_model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=1)),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))
    ]).fit(LANG_DOCS, LANG_LBLS)

    vocab = [
        "for", "while", "append(", "apply(", "iterrows(", "join(", "re.compile(", "open(", "heapq",
        "np.", "pandas", "set(", "dict(", "lru_cache", "deepcopy(", ".copy(", "len(", "json.loads(", "csv."
    ]
    opti_model = Pipeline([
        ("tfidf", TfidfVectorizer(vocabulary=vocab, lowercase=False)),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))
    ]).fit(OPTI_DOCS, OPTI_LBLS)

    return lang_model, opti_model

def method4_supervised_rf(code: str, knowledge_base: List[Dict]) -> Dict:
    lang_model, opti_model = _get_rf_models()
    lang_pred = lang_model.predict([code])[0]
    if hasattr(lang_model, "predict_proba"):
        classes = list(lang_model.classes_)
        probs = lang_model.predict_proba([code])[0]
        lang_scores = {c: float(p) for c, p in zip(classes, probs)}
    else:
        lang_scores = {lang_pred: 1.0}
    opti_pred = opti_model.predict([code])[0]
    _, tags = _detect_python_patterns(code)
    matched_rules = _kb_match_rules_keyword(knowledge_base, code, max_rules=8)
    return {
        "language": lang_pred,
        "language_scores": lang_scores,
        "optimization_types": [opti_pred],
        "tags": sorted(set(tags)),
        "kb_rules": matched_rules,   # sanitized
        "inconsistencies": []
    }

# ==================================================================================================
# MÉTHODE 5 : Embedding + Clustering (SBERT + KMeans)  (kept; sanitized kb_rules)
# ==================================================================================================
def method5_embed_cluster(code: str, knowledge_base: List[Dict], top_k: int = 12, n_clusters: int = 3) -> Dict:
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    except Exception as e:
        raise RuntimeError("Installe: pip install sentence-transformers scikit-learn") from e

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    kb_texts = [
        f"{r.get('contexte','')} | {r.get('anti_pattern','')} | {(r.get('advice') or r.get('advice') or '')}"
        for r in knowledge_base
    ]
    kb_vecs = model.encode(kb_texts, normalize_embeddings=True)
    code_vec = model.encode([code], normalize_embeddings=True)

    sims = cos_sim(code_vec, kb_vecs).ravel()
    idx = np.argsort(-sims)[:top_k]
    top_rules = [knowledge_base[i] for i in idx]
    top_vecs = kb_vecs[idx]

    k = max(1, min(n_clusters, len(top_rules)))
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(top_vecs)

    ctx = []
    tags = []
    for lab, rule in zip(labels, top_rules):
        if rule.get("contexte"): ctx.append(rule["contexte"])
        if rule.get("anti_pattern"): tags.append(f"kb:{rule['id']}:{rule['anti_pattern'][:40]}")
        else: tags.append(f"kb:{rule['id']}")

    language, lang_scores = _language_by_regex(code)
    return {
        "language": language,
        "language_scores": lang_scores,
        "optimization_types": sorted(set(ctx)),
        "tags": sorted(set(tags)),
        "kb_rules": _sanitize_rules(top_rules),   # sanitized
        "inconsistencies": []
    }

# ==================================================================================================
# Dispatch pour appeler la méthode voulue
# ==================================================================================================
def run_green_analyzers(code: str, method: str = "regex") -> Dict:
    method = method.lower()
    if method in ("regex", "heuristics", "m1"):
        return method1_regex_heuristics(code, knowledge_base)
    if method in ("rf", "randomforest", "supervised", "m4"):
        return method4_supervised_rf(code, knowledge_base)
    if method in ("sbert", "kmeans", "embed", "m5"):
        return method5_embed_cluster(code, knowledge_base)
    raise ValueError("Unknown method. Use 'regex', 'rf', or 'sbert'.")

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if __name__ == "__main__":
    user_code = """
result = ""
for i in range(0, len(data)):
    result += str(data[i])
"""

    print("\n=== Optimizing User Code (Green LLM) ===\n")
    suggestion = suggest_optimization(user_code)
    print("\n\n=== Final Suggestion ===\n")
    print(suggestion)

    print("\n\n=== Méthode 1 : Regex + Heuristiques ===")
    print(json.dumps(run_green_analyzers(user_code, "regex"), indent=2, ensure_ascii=False))

    print("\n=== Méthode 4 : RandomForest supervisé ===")
    try:
        print(json.dumps(run_green_analyzers(user_code, "rf"), indent=2, ensure_ascii=False))
    except RuntimeError as e:
        print("Méthode 4 indisponible :", e)

    print("\n=== Méthode 5 : SBERT + KMeans ===")
    try:
        print(json.dumps(run_green_analyzers(user_code, "sbert"), indent=2, ensure_ascii=False))
    except RuntimeError as e:
        print("Méthode 5 indisponible :", e)

    # --- NEW: Show Native AST & Hybrid like in your second slide ---
    print("\n=== Analyse AST native ===")
    print(json.dumps(analyze_native_ast(user_code), indent=2, ensure_ascii=False))

    print("\n=== Modèle hybride AST + Embeddings ===")
    print(json.dumps(analyze_hybrid_ast_embeddings(user_code, top_k=3), indent=2, ensure_ascii=False))

    # --- Mini-benchmark + charts w just 2 ---
    try:
        import matplotlib.pyplot as plt

        # Small demo set (replace with your labeled eval set)
        SAMPLES = [
            {
                "code": "import re\ns=''\nfor l in lines:\n    if re.search(r'\\d+', l): s += l\n",
                "lang": "Python",
            },
            {
                "code": "import pandas as pd\ndf['y']=df['x'].apply(lambda v: v*2)\n",
                "lang": "Python",
            },
            {
                "code": "function f(x){ console.log(x+1) }\nimport lib from 'mylib'\n",
                "lang": "JavaScript",
            },
        ]

        def evaluate_methods(samples):
            methods = {
                "Analyse AST native": analyze_native_ast,
                "Modèle hybride AST + Embeddings": lambda c: analyze_hybrid_ast_embeddings(c, top_k=3),
            }
            tasks = ["Identification Langage"]
            scores = {t: {m: [] for m in methods} for t in tasks}
            for s in samples:
                for name, fn in methods.items():
                    out = fn(s["code"])
                    scores["Identification Langage"][name].append(1.0 if out["language"] == s["lang"] else 0.0)
            # average %
            return {t: {m: round(100*sum(v)/max(1,len(v)),1) for m, v in d.items()} for t, d in scores.items()}

        res = evaluate_methods(SAMPLES)
        # plot single chart example (language id)
        vals = list(res["Identification Langage"].values())
        labels = list(res["Identification Langage"].keys())
        fig, ax = plt.subplots(figsize=(8,4))
        y = np.arange(len(labels))
        ax.barh(y, vals)
        ax.set_yticks(y, labels)
        ax.set_xlim(0,100)
        ax.set_xlabel("Taux de précision (%)")
        ax.set_title("Identification Langage (%)")
        plt.tight_layout(); plt.show()
    except Exception:
        pass
