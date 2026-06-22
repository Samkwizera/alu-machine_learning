#!/usr/bin/env python3
"""Print the location of a GitHub user."""

import sys
import time
import requests


if __name__ == "__main__":
    response = requests.get(sys.argv[1])

    if response.status_code == 404:
        print("Not found")
    elif response.status_code == 403:
        reset = int(response.headers.get("X-Ratelimit-Reset", 0))
        minutes = int((reset - time.time()) / 60)
        if minutes < 0:
            minutes = 0
        print("Reset in {} min".format(minutes))
    else:
        print(response.json().get("location"))
