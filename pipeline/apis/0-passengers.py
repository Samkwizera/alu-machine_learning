#!/usr/bin/env python3
"""Find Star Wars ships that can hold a passenger count."""

import requests


def availableShips(passengerCount):
    """Return ships that can hold at least passengerCount passengers."""
    url = "https://swapi-api.alx-tools.com/api/starships/"
    ships = []

    while url is not None:
        response = requests.get(url)
        data = response.json()

        for ship in data.get("results", []):
            passengers = ship.get("passengers", "").replace(",", "")
            if passengers.isdigit() and int(passengers) >= passengerCount:
                ships.append(ship.get("name"))

        url = data.get("next")

    return ships
