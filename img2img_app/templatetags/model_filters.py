from django import template
import json

register = template.Library()


@register.filter
def get_item(dictionary, key):
    """Get an item from a dictionary in templates."""
    if dictionary is None:
        return None
    return dictionary.get(key)


@register.filter
def to_json(value):
    """Convert a Python object to JSON for use in JavaScript."""
    return json.dumps(value)
