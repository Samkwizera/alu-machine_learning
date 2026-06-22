#!/usr/bin/env python3
"""Find the home planets of all sentient Star Wars species."""

import requests


def sentientPlanets():
    """Return home planet names for all sentient species."""
    url = "https://swapi-api.alx-tools.com/api/species/"
    planets = []

    while url is not None:
        response = requests.get(url)
        data = response.json()

        for species in data.get("results", []):
            if species.get("designation") == "sentient":
                homeworld = species.get("homeworld")
                if homeworld is None:
                    planets.append("unknown")
                else:
                    planet = requests.get(homeworld).json()
                    planets.append(planet.get("name"))

        url = data.get("next")

    return planets
