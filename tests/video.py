import os
import time

import httpx
import numpy as np

from .args import get_prefs, init_test, timer

TAG = "video"


def get_video(FILE_NAME=f".mlop/files/{TAG}"):
    os.makedirs(f"{os.path.dirname(FILE_NAME)}", exist_ok=True)
    r = httpx.get(
        "https://raw.githubusercontent.com/chromium/chromium/master/media/test/data/bear-1280x720.mp4"
    )
    with open(f"{FILE_NAME}.mp4", "wb") as f:
        f.write(r.content)


@timer
def test_video(
    mlop, run, FILE_NAME=f".mlop/files/{TAG}", NUM_EPOCHS=None, ITEM_PER_EPOCH=None
):
    if not os.path.exists(f"{FILE_NAME}.ogg"):
        get_video(FILE_NAME)
    if NUM_EPOCHS is None or ITEM_PER_EPOCH is None:
        NUM_EPOCHS = get_prefs(TAG)["NUM_EPOCHS"]
        ITEM_PER_EPOCH = get_prefs(TAG)["ITEM_PER_EPOCH"]
    WAIT = ITEM_PER_EPOCH * 0.01
    for e in range(NUM_EPOCHS):
        epoch_time = time.time()
        examples = []
        for i in range(ITEM_PER_EPOCH):
            file = mlop.Video(f"{FILE_NAME}.mp4", caption=f"{TAG}-{e}-{i}")
            ndarray = mlop.Video(
                # np.random.randint(low=0, high=256, size=(10, 3, 100, 100), dtype=np.uint8),
                np.random.randint(
                    low=0, high=256, size=(10, 3, 112, 112), dtype=np.uint8
                ),
                caption=f"{TAG}-{e}-{i}-ndarray",
                fps=4,
            )
            gif = mlop.Video(
                np.random.randint(
                    low=0, high=256, size=(10, 3, 112, 112), dtype=np.uint8
                ),
                caption=f"{TAG}-{e}-{i}-gif",
                fps=8,
                format="gif",
            )
            examples.append(file)
            examples.append(gif)
            examples.append(ndarray)
            run.log({f"{TAG}/file": file})
            run.log({f"{TAG}/gif": gif})
            run.log({f"{TAG}/ndarray": ndarray})
        run.log({f"{TAG}/all": examples})
        print(
            f"{TAG}: Epoch {e + 1} / {NUM_EPOCHS} took {time.time() - epoch_time:.4f}s, sleeping {WAIT}s"
        )
        time.sleep(WAIT)


if __name__ == "__main__":
    mlop, run = init_test(TAG)
    test_video(mlop, run)
