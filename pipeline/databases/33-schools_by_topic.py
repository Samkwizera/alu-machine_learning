#!/usr/bin/env python3
"""Find schools by topic in a MongoDB collection."""


def schools_by_topic(mongo_collection, topic):
    """Return schools that include topic in their topics list."""
    return list(mongo_collection.find({"topics": topic}))
