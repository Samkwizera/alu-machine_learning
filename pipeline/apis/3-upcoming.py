#!/usr/bin/env python3
"""Display the next upcoming SpaceX launch."""

import requests


if __name__ == "__main__":
    launches_url = "https://api.spacexdata.com/v4/launches/upcoming"
    rockets_url = "https://api.spacexdata.com/v4/rockets/{}"
    launchpads_url = "https://api.spacexdata.com/v4/launchpads/{}"

    launches = requests.get(launches_url).json()
    launch = sorted(launches, key=lambda x: x.get("date_unix"))[0]

    rocket = requests.get(rockets_url.format(launch.get("rocket"))).json()
    launchpad = requests.get(
        launchpads_url.format(launch.get("launchpad"))
    ).json()

    print(
        "{} ({}) {} - {} ({})".format(
            launch.get("name"),
            launch.get("date_local"),
            rocket.get("name"),
            launchpad.get("name"),
            launchpad.get("locality"),
        )
    )
