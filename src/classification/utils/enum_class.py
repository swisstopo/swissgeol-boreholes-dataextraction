"""Module to declare the union of class enum types."""

from classification.utils.lithology_classes import LithologyClasses
from classification.utils.uscs_classes import USCSClasses

ClassEnum = USCSClasses | LithologyClasses
