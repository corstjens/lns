# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 01:01:06 2014

@author: Benoit
"""

from alns.problem import Request


def test_equal_requests(pickups, deliveries):
    req1 = Request(pickups[1], deliveries[1])
    req2 = Request(pickups[1], deliveries[1])
    assert req1 == req2


def test_unequal_requests1(pickups, deliveries):
    req1 = Request(pickups[1], deliveries[1])
    req2 = Request(pickups[1], deliveries[2])
    req1 != req2


def test_unequal_requests2(pickups, deliveries):
    req1 = Request(pickups[1], deliveries[2])
    req2 = Request(pickups[2], deliveries[2])
    req1 != req2


def test_unequal_requests3(pickups, deliveries):
    req1 = Request(pickups[1], deliveries[2])
    req2 = Request(pickups[2], deliveries[1])
    req1 != req2
