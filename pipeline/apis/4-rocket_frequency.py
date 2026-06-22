#!/usr/bin/env python3
"""Display SpaceX launch counts by rocket."""

import requests


if __name__ == "__main__":
    launches_url = "https://api.spacexdata.com/v4/launches"
    rockets_url = "https://api.spacexdata.com/v4/rockets"

    launches = requests.get(launches_url).json()
    rockets = requests.get(rockets_url).json()
    rocket_names = {rocket.get("id"): rocket.get("name") for rocket in rockets}
    counts = {}

    for launch in launches:
        rocket_id = launch.get("rocket")
        name = rocket_names.get(rocket_id)
        counts[name] = counts.get(name, 0) + 1

    for name, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        print("{}: {}".format(name, count))
