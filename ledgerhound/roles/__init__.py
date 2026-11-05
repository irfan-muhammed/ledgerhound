"""
ledgerhound.roles
=================

Column roles (design doc, "Column roles"): guardrails key off what a column
IS, never what it is called.

    model   ROLES, RoleMap, Assignment, PERSON_ONLY, NOT_A_FEATURE
    infer   infer_roles(df): generic name-token + value-shape inference
    store   RoleStore: confirmed maps, per dataset content hash
"""

from ledgerhound.roles.infer import infer_roles, tokens
from ledgerhound.roles.model import (
    ENTITY_ROLES,
    NOT_A_FEATURE,
    PERSON_ONLY,
    ROLES,
    Assignment,
    RoleMap,
)
from ledgerhound.roles.store import RoleStore

__all__ = ["ROLES", "PERSON_ONLY", "NOT_A_FEATURE", "ENTITY_ROLES", "Assignment",
           "RoleMap", "RoleStore", "infer_roles", "tokens"]
