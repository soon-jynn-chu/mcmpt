import sys
import time
from argparse import ArgumentParser

import yaml

from utils import SingleCamera, Models, Utilities


def parse_opt():
    parser = ArgumentParser()
    parser.add_argument(
        "--configs", type=str, default="configs.yaml", help="Path to configs"
    )
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    total_start_time = time.time()

    with open(opt.configs, "r") as stream:
        try:
            configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise exc

    videos = configs["sources"]

    models = Models(configs)
    utilities = Utilities(configs)

    # Create SingleCamera object for each video
    cameras = [
        SingleCamera(
            input=video,
            models=models,
            utilities=utilities,
            configs=configs,
        )
        for video in videos
    ]

    max_frames = max([camera.length for camera in cameras])
    count = 0

    while len(cameras) != 0:
        t = time.time()
        count += 1

        dones = [camera.update_single_frame() for camera in cameras]

        # Remove completed videos
        for i in reversed(range(len(dones))):
            if dones[i]:
                cameras.pop(i)

        if len(cameras) != 0:
            total_time = time.time() - t
            average_time = total_time / len(dones)
            print(
                f"Elapsed time: {time.time() - total_start_time:.2f}   Cameras: {len(cameras)}   Frame: {count}/{max_frames}   Total: {total_time:.2f}s   Average: {average_time:.2f}s/camera"
            )

    # Save a copy of configs
    with open(f"{utilities.save_path}/configs.yml", "w") as f:
        yaml.dump(configs, f, default_flow_style=False)

    print(f"Results save to {utilities.save_path}")
    print(f"Total time: {time.time() - total_start_time:.2f}s")
