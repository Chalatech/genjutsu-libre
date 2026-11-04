# Plan: Genjustsu con Kie (Seedance 2.5) y GPU propia, sin Enhancor ni Replicate

Rama: `feat/kie-gpu-local` del fork `Chalatech/genjutsu-kie` (upstream: `sirioberati/Genjustsu-Open-Source-Workflow`, base `40fd661`).
Fecha: 2026-11-04. Este repo es **público**: nada de nombres de máquinas, IPs, usuarios u organizaciones del host GPU en commits; se le llama «GPU host» y se configura por `.env`.

## Por qué

El flujo de upstream reemplaza al sujeto de un video conservando cámara, fondo y audio original: separa voz y música, saca profundidad en color y máscaras del sujeto, compone el sujeto en profundidad sobre el fondo real y manda eso a Seedance con las imágenes de los personajes nuevos. Quita la identidad del actor original antes de generar, que es justo lo que falló con Wan 3 (el guardia original reaparecía).

Upstream depende de dos servicios externos:

| Etapa | Upstream | Este fork |
|---|---|---|
| Voz y música | Replicate `cjwbw/demucs` (htdemucs) | `demucs` htdemucs en el GPU host |
| Profundidad en color | Replicate `lucataco/depth-anything-video` | Video-Depth-Anything **Small** (Apache-2.0) en el GPU host, colormap inferno |
| Máscaras | Replicate `lucataco/sam3-video` | SAM 3 (`facebook/sam3`, acceso restringido en HF) o, sin acceso, SAM 2.1 + detector de personas, en el GPU host |
| Transporte | n/a | `rsync`/`ssh` a un alias del GPU host |
| Generación | Enhancor Seedance 2.5 (`edit` / `multi_reference`, `pass_faces`) | Kie.ai `bytedance/seedance-2-5` (referencia multimodal, `draft: true`) y cierre HD con `bytedance/seedance-2-5-draft` |
| Hosting de medios | tmpfiles / catbox (URLs públicas) | subida de archivos de Kie (expira en 24 h) |
| Webhook | cloudflared obligatorio | no hace falta: Kie se consulta por polling |

El comportamiento de upstream queda como **default**: los backends nuevos se activan por `.env`.

## Estado de partida (medido 2026-11-04)

- Setup de upstream en macOS ARM64, Python 3.15.2, Rubber Band 4.0.0: dependencias OK.
- `test_composite.py`: PASS. `python -m unittest discover -s tests`: **13 tests OK** (todos con mocks).
- `doctor.py --require-keys`: UI lista, generación bloqueada (faltan `ENHANCOR_API_KEY`, `REPLICATE_API_TOKEN`, webhook).
- UI local responde 200 en `/`, `/docs` y en las imágenes de `static/`.

## Configuración nueva (`.env`, nunca se commitea)

```
GEN_BACKEND=enhancor        # enhancor | kie
PREP_BACKEND=replicate      # replicate | gpu
KIE_API_KEY=
KIE_UNIT_CREDITS=17         # créditos por segundo facturable; medir y corregir en F6
KIE_MAX_USD=3               # tope por request; se rechaza por encima
GPU_HOST=                   # alias de ~/.ssh/config
GPU_DIR=                    # carpeta de trabajo en el host (solo ahí se escribe)
GPU_PYTHON=                 # python del venv del host
GPU_MIN_FREE_GB=30          # memoria libre mínima antes de lanzar un worker
SEGMENT_MODEL=sam3          # sam3 | sam2
HF_TOKEN=                   # solo para SAM 3 (cuenta propia, nunca una compartida)
```

## Contratos que el código existente ya exige (no romperlos)

- `composite()` exige el mismo número de cuadros en `original.mp4`, `depth.mp4` y las máscaras, y nombres de máscara `mask_<n>.png` con `n` = 0..N-1 consecutivos (usar `mask_00000.png`). `normalize()` deja 24 fps y lado mayor ≤ 960.
- `separate()` deja `original-audio.wav`, `vocals.wav`, `music.wav`, un `.wav` por stem y `audio-manifest.json` con las mismas claves; el backend GPU produce lo mismo (con `prediction_id` = id local del trabajo).
- `mask_review` liga la aprobación a hashes de los artefactos: cualquier backend nuevo debe escribir los mismos archivos antes de la revisión.
- Seedance recibe `@Video1` con las voces aisladas (+3 semitonos) incrustadas y devuelve video cuyo audio se descarta; el audio final es siempre el original.

### Restricciones de Kie Seedance 2.5 (docs.kie.ai/market/bytedance/seedance-2-5, leídas 2026-11-04)

- Escenarios excluyentes: texto, primer/último cuadro, o **referencia multimodal** (hasta 30 imágenes, 10 videos y 10 audios).
- Video de referencia: mp4/mov/mkv, 24–60 fps, 2–30 s cada uno y ≤ 30 s en total, lados 300–6000 px, aspecto 0.4–2.5, **ancho×alto entre 409 600 y 927 408**, ≤ 200 MB. (960×540 = 518 400 cumple.)
- Imágenes: jpeg/png/webp, lados 300–6000 px, aspecto 0.4–2.5, < 30 MB.
- `duration` 4–30 o `-1` (en tareas de edición iguala el largo del video de entrada). `resolution` 480p/720p/1080p. `generate_audio` (default `true`). `draft: true` solo admite 480p; el id del draft vale 7 días y se completa con el modelo `bytedance/seedance-2-5-draft` y `draft_task_id`.
- Cobro con video de entrada: precio unitario × (segundos de video de referencia + segundos de salida). Precio publicado «desde $0.085/s (17 créditos)»; el precio por variante (resolución, draft) **no está publicado**: se mide en F6.
- Kie responde HTTP 200 aunque falle: mirar el campo `code`. Estado: `GET /api/v1/jobs/recordInfo?taskId=`; `creditsConsumed` es 0 en tareas fallidas. Las tareas pueden quedarse en `waiting` 30+ min y fallar por timeout sin cobro.

## Reglas para todas las fases

1. Solo la rama `feat/kie-gpu-local`. Nunca `main`.
2. Repo público: nunca commitear medios, caras, llaves, `.env`, `runs/`, ni datos del GPU host (nombre, IP, usuario, organización). Ejemplos y tests con datos sintéticos.
3. Los 13 tests originales y `test_composite.py` pasan al cierre de cada fase. Cada fase agrega sus tests.
4. **Ninguna llamada paga** (Kie `createTask`, Replicate, Enhancor) en fases desatendidas: solo mocks. Las llamadas reales son de F6, con Carlos presente y su visto bueno por cada una.
5. En el GPU host: solo se escribe dentro de `GPU_DIR`; sin `sudo`; sin tocar procesos ni archivos ajenos; antes de cada worker se mide la memoria libre y se aborta si es menor a `GPU_MIN_FREE_GB`; workers con `nice -n 10`, sin demonios que queden vivos. Detalles del host en las notas locales del trabajo (fuera del repo).
6. Código y docs nuevos en inglés (igual que upstream). Commits convencionales.
7. Video-Depth-Anything **Large** es CC-BY-NC: no se usa por defecto.

## Fases

### F0 (hecha): fork, rama y línea base

Fork, remotos (`origin` = fork, `upstream` = original), rama, setup y tests en verde, este plan.

### F1: entorno del GPU host

Archivos: `gpu_workers/setup_gpu.sh`, `gpu_workers/requirements-gpu.txt`, `gpu_workers/check.py`.

- `setup_gpu.sh` idempotente y en espacio de usuario: instala `uv` para el usuario si falta, crea `$GPU_DIR/.venv` con Python 3.12, instala torch con CUDA para la arquitectura del host (probar primero el índice oficial de PyTorch para CUDA 13 / sbsa; si no hay wheel compatible con la GPU, documentar la alternativa con el contenedor NGC de PyTorch, solo si el usuario ya puede usar docker), `demucs`, Video-Depth-Anything (git en commit fijo), SAM 3 o SAM 2.1, `opencv-python-headless`, `av`. Pesos en `$GPU_DIR/models`.
- `check.py`: versión de torch, `cuda.is_available()`, nombre y capability de la GPU, tiempo de un matmul 4096², memoria libre del sistema; sale con código 1 si no hay CUDA.

Aceptación: en el GPU host `check.py` sale 0 con `cuda=True` y la capability de la GPU soportada por el torch instalado; cifras y disco usado (< 25 GB) anotados en el estado del trabajo.

### F2: workers GPU

Archivos: `gpu_workers/separate.py`, `gpu_workers/depth.py`, `gpu_workers/segment.py`, `gpu_workers/common.py`. Cada uno es CLI `--in <dir> --out <dir>` y escribe `report-<worker>.json` con tiempos y pico de memoria (`torch.cuda.max_memory_allocated`).

- `separate.py`: htdemucs → `vocals.wav`, stems y `music.wav` con las mismas convenciones que `separate()`.
- `depth.py`: Video-Depth-Anything Small sobre `original.mp4` → `depth-raw.mp4` en color (inferno), mismo número de cuadros, 24 fps.
- `segment.py`: personas con seguimiento. SAM 3 con texto (`--prompt person`) si hay acceso; si no, SAM 2.1 sembrado con cajas de un detector de personas cada N cuadros para capturar entradas tardías. Salida: `instances/<id>/mask_00000.png…`, `instances.json` (id, primer/último cuadro, área media) y `instances-contact.jpg` (cuadros muestra con cada id coloreado y rotulado). `--select 1,3` escribe la unión en `raw-masks/mask_00000.png…`.

Aceptación, sobre un clip de prueba local (no se commitea): número de cuadros idéntico entre original, profundidad y máscaras; profundidad no constante; ≥ 2 instancias detectadas en un clip con dos personas; máscara no vacía en ≥ 90 % de los cuadros donde la instancia está; pico de memoria ≤ 20 GB por worker; tiempos anotados. Tests en el Mac (sin GPU) para la unión de instancias y el naming.

### F3: backend remoto e integración con el pipeline

Archivos: `gpu_backend.py`, cambios en `pipeline.py`, `audio_workflow.py`, `raw_masks.py`, `workflow.py`, `app.py`.

- `gpu_backend.py`: `push(job)`, `run(worker, args)`, `pull(job)` con `rsync`/`ssh` al alias `GPU_HOST`; carpeta remota `$GPU_DIR/runs/<job>`; chequeo de memoria libre antes de correr; `nice -n 10`; timeout; logs remotos copiados al job local. Nunca mata procesos.
- `PREP_BACKEND=gpu` enruta `separate()`, la profundidad (`provider='gpu'`) y `segment_raw()`; `workflow.py prepare … --provider gpu --instances 1,3`; `app.py` acepta `provider=gpu`; la revisión de máscara puede elegir instancias y regenera la unión (lo que invalida la aprobación anterior por hash).

Aceptación: `workflow.py prepare <clip> --provider gpu` corre de punta a punta **sin** `REPLICATE_API_TOKEN` y deja `composite.mp4`, `mask.mp4`, `raw-masks/`, `audio-manifest.json`, `seedance-input` listo y fingerprint de revisión; tests con `ssh`/`rsync` simulados (líneas de comando, rechazo por memoria baja, error remoto propagado); tests viejos en verde.

### F4: puente Kie

Archivos: `kie_bridge.py`, despacho en `seedance_bridge.py` por `GEN_BACKEND`, `doctor.py`.

- Mismas firmas: `submit(folder, prompt, sirio, tein=None, status=print)`, `wait_and_finish(folder, request_id, status=print)`, `submit_hd(folder)`.
- Valida localmente las restricciones de Kie (sección de contratos) **antes** de subir nada; sube con la API de archivos de Kie.
- Request: modelo `bytedance/seedance-2-5`, `reference_video_urls=[composite con voces]`, `reference_image_urls=[referencias]`, `draft: true`, `resolution: 480p`, `duration: -1`, `generate_audio: false`, prompt con `@Video1`/`@Image1…`.
- Costo estimado = `KIE_UNIT_CREDITS` × (segundos de referencia + salida) × $0.005; se imprime y se rechaza si supera `KIE_MAX_USD`.
- Escribe `seedance-request.json` y `kie-task.json` (taskId) **antes** de hacer polling; si `kie-task.json` existe, `submit` se niega (sin duplicados). Nunca reintenta un `createTask`. `fail` → error con el mensaje del proveedor.
- HD: `bytedance/seedance-2-5-draft` con `draft_task_id`, resolución pedida 720p o 1080p; misma protección de duplicados en `hd/`.
- `doctor.py`: con `GEN_BACKEND=kie` exige `KIE_API_KEY` y no exige webhook; con `PREP_BACKEND=gpu` verifica `ssh` y `check.py` remoto en vez de `REPLICATE_API_TOKEN`.

Aceptación: tests con HTTP simulado para el cuerpo del draft y del HD, el tope de costo, la negativa por duplicado, el manejo de `fail`/`waiting` y la validación de restricciones (píxeles, fps, duración, tamaño de imagen); cero llamadas reales; tests viejos en verde.

### F5: UI, doctor y documentación

- UI: selector de backend de preparación y de generación, hoja de instancias en la revisión de máscara con casillas por id, costo estimado antes de enviar.
- `docs/LOCAL-GPU.md`, `docs/KIE.md`, sección «Fork notes» en el README.

Aceptación: con `PREP_BACKEND=gpu`, `GEN_BACKEND=kie` y sin llaves de Enhancor ni Replicate, `doctor.py --require-keys` reporta listo; la UI responde 200 y muestra las opciones nuevas; tests en verde.

### F6 (con Carlos): prueba real de la toma 2

Clip: segmento 12.05–27.0 s del video de prueba (local). Personajes: protagonista y guardia, cada uno con retrato y cuerpo sobre fondo blanco.

1. Preparar en el GPU host. Carlos revisa la máscara completa y elige las instancias: protagonista y guardia; los bailarines del fondo quedan fuera.
2. Sonda de caras reales (llamada paga, con visto bueno): la salida más corta posible con referencia de 4 s, 480p draft (≈ (4+4) × $0.085 ≈ $0.68). Mide si Kie acepta caras reales y el precio real por segundo; corregir `KIE_UNIT_CREDITS`.
3. Draft de la toma completa a 480p (≈ (14.9 + 15) × precio medido; ≈ $2.5 con el precio publicado), con visto bueno.
4. Carlos juzga. Solo con su aprobación, HD a 720p con `draft_task_id`.

Aceptación: veredicto de Carlos (el guardia es el de las referencias en toda la toma; la ropa no cambia; el fondo es el del video original); costos reales anotados.

### Auditoría (Opus)

Re-corre tests y doctor, revisa cada criterio y las reglas en el diff (en especial: nada del GPU host ni medios en el repo, ninguna llamada paga fuera de F6), arregla solo defectos chicos y escribe `INFORME.md` en español.

## Riesgos y preguntas abiertas

- **Caras reales en Kie:** Enhancor usa `pass_faces: true`; Kie no documenta un equivalente y menciona `asset://{assetId}`. Si Kie rechaza caras reales, F6 se detiene y se evalúa la librería de assets de Kie o volver a Enhancor solo para la generación.
- **Precio por variante** de Seedance 2.5 en Kie: desconocido hasta F6.
- **Lip-sync con `generate_audio: false`:** no está confirmado que Seedance use las voces incrustadas en `@Video1` para sincronizar la boca si no genera audio.
- **torch en el GPU host:** depende de que exista un wheel con CUDA para su arquitectura y GPU; si no, contenedor NGC (solo si el usuario ya tiene docker sin `sudo`).
- **SAM 3** está restringido en Hugging Face: requiere que Carlos pida acceso con su cuenta. Sin acceso, SAM 2.1 + detector, con más trabajo de selección de instancias.
- **GPU host compartido:** memoria variable; los workers se abortan si no hay margen.
- **Red:** el GPU host solo es accesible desde la red donde está; el runner de fases corre en la máquina que lo alcanza.
