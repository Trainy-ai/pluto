import os
import platform
import time

import httpx

from .args import parse, read_sets_compat

TAG = "test-audio"
ITEM_PER_EPOCH = 10
WAIT = ITEM_PER_EPOCH * 0.01
if platform.system() == "Linux":  # or platform.machine() == "x86_64"
    NUM_EPOCHS = 2  # for actions
else:
    NUM_EPOCHS = 10
TOTAL = NUM_EPOCHS * ITEM_PER_EPOCH
INIT = time.time()

FILE_NAME = ".mlop/files/audio"
os.makedirs(f"{os.path.dirname(FILE_NAME)}", exist_ok=True)
r = httpx.get(
    "https://actions.google.com/sounds/v1/alarms/digital_watch_alarm_long.ogg"
)
with open(f"{FILE_NAME}.ogg", "wb") as f:
    f.write(r.content)


args = parse(TAG)
mlop, settings = read_sets_compat(args, TAG)
run = mlop.init(dir=".mlop/", project=TAG, settings=settings)

for e in range(NUM_EPOCHS):
    examples = []
    RUN = time.time()
    for i in range(ITEM_PER_EPOCH):
        file = mlop.Audio(f"{FILE_NAME}.ogg", caption=f"random-field-{e}-{i}")
        examples.append(file)
        run.log({"A/iter": file})
    run.log({"B/0": examples})
    print(
        f"{TAG}: Epoch {e + 1} / {NUM_EPOCHS} took {time.time() - RUN:.4f}s, now waiting {WAIT}s"
    )
    time.sleep(WAIT)

print(f"{TAG}: Script time ({TOTAL}): {time.time() - INIT:.4f}s")
