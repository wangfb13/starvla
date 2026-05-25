#!/usr/bin/env python

from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.pi05_full import PI05FullConfig, PI05FullPolicy


def test_make_policy_config_pi05_full():
    config = make_policy_config("pi05_full")

    assert isinstance(config, PI05FullConfig)
    assert config.type == "pi05_full"


def test_get_policy_class_pi05_full():
    policy_cls = get_policy_class("pi05_full")

    assert policy_cls is PI05FullPolicy
